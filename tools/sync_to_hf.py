#!/usr/bin/env python3
"""同步 meme-generator 的表情目录到 HuggingFace Space（增量同步）。

工作原理：
  1. 扫描本地 SYNC_DIRS / SYNC_FILES，计算每个文件的内容哈希。
     - 普通文件: git blob sha1（sha1("blob <len>\\0" + 内容)），与 HF API 返回的 oid 对比
     - LFS 文件: sha256，与 HF API 返回的 lfs.oid 对比
     - 文本文件先做 CRLF→LF 规范化（Windows checkout / autocrlf 场景）
  2. 从 HF Space API 拉取远端文件清单，逐文件对比。
  3. 只上传「新增或内容变化」的文件；HF 按内容哈希去重，未变化的文件不会重复占用空间。
  4. 删除远端存在、但本地已不存在的表情文件（上游移除表情时自动跟进）。
  5. HF Space 的专有文件（根目录 app.py、README.md 的 gradio front matter、
     requirements.txt 等）永不触碰。

用法（GitHub Actions 自动运行，也可本地手动运行）：
  HF_TOKEN=hf_xxx python tools/sync_to_hf.py            # 正常同步
  HF_TOKEN=hf_xxx python tools/sync_to_hf.py --dry-run  # 只打印差异，不做任何修改
"""

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

HF_REPO = os.environ.get("HF_REPO", "donotthink/meme")
HF_REPO_TYPE = "space"

# 需要同步到 HF Space 的表情目录（相对仓库根）
SYNC_DIRS = [
    "meme_generator/memes",
    "meme_generator/memes_jj",
    "meme_generator/memes_emoji",
    "meme_generator/memes_other",
    "meme_generator/memes_emoji_nsfw",
]

# meme_generator 核心代码也同步（HF Space 直接用这份代码跑 FastAPI）
SYNC_FILES = [
    "meme_generator/__init__.py",
    "meme_generator/app.py",
    "meme_generator/cli.py",
    "meme_generator/compat.py",
    "meme_generator/config.py",
    "meme_generator/dirs.py",
    "meme_generator/download.py",
    "meme_generator/exception.py",
    "meme_generator/log.py",
    "meme_generator/manager.py",
    "meme_generator/meme.py",
    "meme_generator/tags.py",
    "meme_generator/utils.py",
    "meme_generator/version.py",
    "meme_generator/resources/resource_list.json",
]

# HF Space 上的专有文件：永不覆盖、永不删除（不参与对比即可）
HF_ONLY_FILES = {
    "app.py",            # HF 入口：拉取 Xet 指针图片 + ZeroGPU 预热 + 启动 FastAPI
    "README.md",         # 带 gradio front matter（title/sdk/app_file 等）
    "requirements.txt",
    "packages.txt",
    ".gitattributes",
    ".gitignore",
}

# 本地扫描时跳过的目录/文件名
SKIP_NAMES = {"__pycache__", ".DS_Store"}

# 需要做 CRLF→LF 规范化的文本扩展名（git autocrlf 检出时会把 LF 变 CRLF）
TEXT_EXTS = {".py", ".md", ".json", ".toml", ".txt", ".yaml", ".yml", ".lock", ".cfg"}


def file_hashes(path: Path) -> tuple[str, str]:
    """返回 (git_blob_sha1, sha256)，内容统一为 LF 规范化后的字节。"""
    data = path.read_bytes()
    if path.suffix.lower() in TEXT_EXTS and b"\r\n" in data:
        data = data.replace(b"\r\n", b"\n")
    blob_sha1 = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    sha256 = hashlib.sha256(data).hexdigest()
    return blob_sha1, sha256


def collect_local_files(repo_root: Path) -> dict[str, tuple[str, str]]:
    """返回 {仓库相对路径: (blob_sha1, sha256)}。"""
    result: dict[str, tuple[str, str]] = {}
    for rel_dir in SYNC_DIRS:
        dir_path = repo_root / rel_dir
        if not dir_path.exists():
            print(f"[warn] 目录不存在，跳过: {rel_dir}")
            continue
        for p in dir_path.rglob("*"):
            if not p.is_file():
                continue
            if any(part in SKIP_NAMES for part in p.parts):
                continue
            rel = p.relative_to(repo_root).as_posix()
            result[rel] = file_hashes(p)
    for rel_file in SYNC_FILES:
        p = repo_root / rel_file
        if p.exists():
            result[rel_file] = file_hashes(p)
    return result


def collect_remote_files(api: HfApi) -> dict[str, tuple[str, str]]:
    """返回 {仓库路径: ("sha1"|"sha256", oid)}。只保留同步范围内的文件。"""
    prefixes = tuple(d + "/" for d in SYNC_DIRS)
    result: dict[str, tuple[str, str]] = {}
    for info in api.list_repo_tree(HF_REPO, repo_type=HF_REPO_TYPE, recursive=True):
        if info.type != "file":
            continue
        path = info.path
        if path in HF_ONLY_FILES:
            continue
        if not (path.startswith(prefixes) or path in SYNC_FILES):
            continue
        lfs = getattr(info, "lfs", None)
        if lfs:
            result[path] = ("sha256", lfs.oid)
        else:
            result[path] = ("sha1", info.oid)
    return result


def file_matches(remote_entry: tuple[str, str], local_hashes: tuple[str, str]) -> bool:
    kind, oid = remote_entry
    blob_sha1, sha256 = local_hashes
    return oid == (sha256 if kind == "sha256" else blob_sha1)


def upload_one(api: HfApi, local: Path, rel: str) -> bool:
    for attempt in range(3):
        try:
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=rel,
                repo_id=HF_REPO,
                repo_type=HF_REPO_TYPE,
                commit_message=f"sync: {rel}",
            )
            return True
        except Exception as e:
            print(f"  [retry {attempt + 1}/3] {rel}: {e}")
            time.sleep(5 * (attempt + 1))
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="只对比差异，不做任何修改")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token and not args.dry_run:
        print("[error] 缺少 HF_TOKEN 环境变量（HuggingFace 的 write 权限 token）")
        return 1

    repo_root = Path(__file__).resolve().parent.parent
    print(f"[info] 本地仓库: {repo_root}")
    print(f"[info] 目标 Space: {HF_REPO}")

    api = HfApi(token=token)

    print("[1/4] 扫描本地文件 ...")
    local = collect_local_files(repo_root)
    print(f"      本地待同步文件: {len(local)}")

    print("[2/4] 获取 HF Space 远端清单 ...")
    remote = collect_remote_files(api)
    print(f"      远端文件(同步范围内): {len(remote)}")

    to_upload = sorted(p for p, h in local.items()
                       if p not in remote or not file_matches(remote[p], h))
    to_delete = sorted(p for p in remote if p not in local)

    print(f"[3/4] 差异: 需上传 {len(to_upload)} 个, 需删除 {len(to_delete)} 个")
    if not to_upload and not to_delete:
        print("[done] 已是最新，无需同步")
        return 0

    if args.dry_run:
        for p in to_upload:
            print(f"  + {p}")
        for p in to_delete:
            print(f"  - {p}")
        print("[dry-run] 未做任何修改")
        return 0

    print("[4/4] 开始上传 ...")
    ok, fail = 0, 0
    for i, rel in enumerate(to_upload, 1):
        print(f"  ({i}/{len(to_upload)}) {rel}", flush=True)
        if upload_one(api, repo_root / rel, rel):
            ok += 1
        else:
            fail += 1
    print(f"      上传完成: 成功 {ok}, 失败 {fail}")

    if to_delete:
        print(f"      删除 {len(to_delete)} 个上游已移除的文件 ...")
        try:
            api.delete_files(
                HF_REPO,
                repo_type=HF_REPO_TYPE,
                paths=to_delete,
                commit_message="sync: 删除上游已移除的表情文件",
            )
        except Exception as e:
            print(f"      [warn] 删除失败（不影响表情使用，下次运行会重试）: {e}")

    if fail:
        print(f"[warn] 有 {fail} 个文件上传失败，请重跑本 workflow")
        return 1
    print("[done] 同步完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
