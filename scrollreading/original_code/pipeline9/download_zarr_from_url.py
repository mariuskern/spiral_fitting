#!/usr/bin/env python3
"""
download_zarr.py

Recursively download a Zarr store (or any statically-served directory tree,
like the ones at dl.ash2txt.org) from a URL to a local destination folder.

Zarr stores are just directories full of small metadata files (.zarray,
.zgroup, .zattrs) and chunk files. When served over plain HTTP with
directory-listing enabled, this script walks the listing pages, recreates
the folder structure locally, and downloads every file it finds.

Usage:
    python download_zarr.py <URL> <DEST_FOLDER> [--workers 8] [--force]

Example:
    python download_zarr.py \\
        "https://dl.ash2txt.org/full-scrolls/Scroll4/PHerc1667.volpkg/volumes_zarr/20231117161658.zarr/2/" \\
        "./20231117161658_level2"
"""

import argparse
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, urlparse, unquote

import requests
from requests.adapters import HTTPAdapter, Retry

# Matches href="..." or href='...' in an HTML directory listing page
HREF_RE = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)


def make_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def list_directory(session: requests.Session, url: str):
    """Return (subdirs, files) as absolute URLs found on an autoindex page."""
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    base = resp.url  # follow redirects correctly when resolving relative links

    subdirs, files = [], []
    for href in HREF_RE.findall(resp.text):
        if href.startswith(("?", "#")) or href.startswith(("http://", "https://")) and not href.startswith(base):
            # skip sort-query links and links that jump to another host
            if not href.startswith(base):
                continue
        if href in ("../", "..", "/"):
            continue

        full = urljoin(base, href)

        # Only follow links that stay within the directory we're crawling
        if not full.startswith(base) and not full.startswith(url):
            continue

        if full.rstrip("/") == url.rstrip("/"):
            continue

        if full.endswith("/"):
            subdirs.append(full)
        else:
            files.append(full)

    return subdirs, files


def walk_remote(session: requests.Session, root_url: str):
    """Recursively walk the directory tree, yielding file URLs."""
    if not root_url.endswith("/"):
        root_url += "/"

    stack = [root_url]
    all_files = []

    while stack:
        current = stack.pop()
        try:
            subdirs, files = list_directory(session, current)
        except requests.HTTPError as e:
            print(f"  [warn] could not list {current}: {e}", file=sys.stderr)
            continue

        all_files.extend(files)
        stack.extend(subdirs)
        print(f"  [scan] {current}  (+{len(files)} files, {len(subdirs)} subdirs)")

    return all_files


def local_path_for(root_url: str, file_url: str, dest_root: str) -> str:
    rel = file_url[len(root_url):] if file_url.startswith(root_url) else urlparse(file_url).path.lstrip("/")
    rel = unquote(rel)
    return os.path.join(dest_root, rel)


def download_file(session: requests.Session, url: str, dest_path: str, force: bool) -> str:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    if not force and os.path.exists(dest_path):
        try:
            head = session.head(url, timeout=30, allow_redirects=True)
            remote_size = int(head.headers.get("Content-Length", -1))
            if remote_size >= 0 and os.path.getsize(dest_path) == remote_size:
                return f"skip (exists): {dest_path}"
        except requests.RequestException:
            pass  # fall through and re-download if HEAD fails

    with session.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        tmp_path = dest_path + ".part"
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
        os.replace(tmp_path, dest_path)

    return f"downloaded: {dest_path}"


def main():
    parser = argparse.ArgumentParser(description="Download a Zarr store (or static HTTP directory) to a local folder.")
    parser.add_argument("url", help="Source URL of the zarr directory (must be listable over HTTP)")
    parser.add_argument("dest", help="Destination local folder")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel download threads (default: 8)")
    parser.add_argument("--force", action="store_true", help="Re-download files even if they already exist locally")
    args = parser.parse_args()

    root_url = args.url if args.url.endswith("/") else args.url + "/"
    dest_root = os.path.abspath(args.dest)
    os.makedirs(dest_root, exist_ok=True)

    session = make_session()

    print(f"Scanning remote directory tree at:\n  {root_url}\n")
    files = walk_remote(session, root_url)
    total = len(files)
    print(f"\nFound {total} files. Downloading with {args.workers} workers into:\n  {dest_root}\n")

    done = 0
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_file, session, url, local_path_for(root_url, url, dest_root), args.force
            ): url
            for url in files
        }
        for future in as_completed(futures):
            url = futures[future]
            done += 1
            try:
                result = future.result()
                print(f"[{done}/{total}] {result}")
            except Exception as e:
                errors.append((url, str(e)))
                print(f"[{done}/{total}] ERROR downloading {url}: {e}", file=sys.stderr)

    print(f"\nDone. {total - len(errors)}/{total} files downloaded successfully.")
    if errors:
        print(f"{len(errors)} file(s) failed:", file=sys.stderr)
        for url, err in errors:
            print(f"  {url}: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()