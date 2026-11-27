"""Print a local or remote Zarr hierarchy without reading array chunks."""

import argparse
import json
import os
from pathlib import Path
import sys
from urllib.parse import unquote, urlencode, urlsplit, urlunsplit
import xml.etree.ElementTree as ET

import fsspec
import numpy as np


def is_s3_http_url(url):
    parts = urlsplit(url)
    host = parts.hostname or ""
    return parts.scheme in ("http", "https") and host.endswith(".amazonaws.com") and (".s3." in host or ".s3-" in host)


def normalize_url(url):
    """Resolve local paths, or turn S3 browser links into object URLs."""
    url = os.fspath(url)
    if "://" not in url:
        # Local names may contain # and must not be parsed as URL fragments.
        return Path(url).expanduser().resolve().as_posix()
    parts = urlsplit(url)
    if is_s3_http_url(url) and parts.path.rstrip("/").endswith("/index.html") and parts.fragment:
        return urlunsplit((parts.scheme, parts.netloc, "/" + parts.fragment.lstrip("/"), "", ""))
    if parts.fragment:
        raise ValueError("Use a direct Zarr data URL, without a browser-page #fragment.")
    return url


def join_path(*parts):
    return "/".join(part.strip("/") for part in parts if part.strip("/"))


class MetadataReader:
    """Read raw metadata so inspecting a store never requires its codecs."""

    def __init__(self, source):
        self.fs, self.base = fsspec.core.url_to_fs(normalize_url(source))
        self.cache = {}
        self.consolidated_groups = set()

    def url(self, path):
        return self.base.rstrip("/") + "/" + path

    def read_json(self, path, name):
        key = join_path(path, name)
        if key not in self.cache:
            try:
                content = self.fs.cat_file(self.url(key))
            except FileNotFoundError:
                return None
            try:
                self.cache[key] = json.loads(content)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise ValueError(
                    f"Expected JSON metadata at {self.url(key)}. "
                    "Check that the URL points to Zarr data, not an HTML browser page."
                ) from error
        return self.cache[key]

    def metadata(self, path):
        data = self.read_json(path, "zarr.json")
        if data is not None:
            consolidated = data.get("consolidated_metadata")
            if consolidated and consolidated.get("kind") == "inline":
                for child, metadata in consolidated.get("metadata", {}).items():
                    self.cache[join_path(path, child, "zarr.json")] = metadata
                self.consolidated_groups.add(path)
            return data, data.get("attributes", {})

        consolidated = self.read_json(path, ".zmetadata")
        if consolidated is not None:
            for key, metadata in consolidated["metadata"].items():
                self.cache[join_path(path, key)] = metadata
            self.consolidated_groups.add(path)
        data = self.read_json(path, ".zarray")
        if data is None:
            data = self.read_json(path, ".zgroup")
        if data is None:
            raise FileNotFoundError(f"No Zarr metadata found at {self.url(path)}")
        return data, self.read_json(path, ".zattrs") or {}

    def s3_children(self, path):
        """List immediate prefixes, never the chunk objects inside arrays."""
        parts = urlsplit(self.url(path))
        prefix = unquote(parts.path.lstrip("/")).rstrip("/") + "/"
        endpoint = urlunsplit((parts.scheme, parts.netloc, "/", "", ""))
        names = set()
        token = None
        while True:
            params = {"list-type": "2", "prefix": prefix, "delimiter": "/"}
            if token:
                params["continuation-token"] = token
            listing = ET.fromstring(self.fs.cat_file(endpoint + "?" + urlencode(params)))
            for item in listing.findall("{*}CommonPrefixes/{*}Prefix"):
                names.add(item.text[len(prefix):].strip("/"))
            token = listing.findtext("{*}NextContinuationToken")
            if not token:
                return names

    def children(self, path, attrs):
        prefix = path.rstrip("/") + "/" if path else ""
        names = {key[len(prefix):].split("/", 1)[0] for key in self.cache
                 if key.startswith(prefix) and "/" in key[len(prefix):]}
        # OME datasets also allow traversal when directory listing is unavailable.
        multiscales = attrs.get("multiscales", []) + attrs.get("ome", {}).get("multiscales", [])
        for multiscale in multiscales:
            for dataset in multiscale.get("datasets", []):
                relative = dataset["path"].strip("/")
                if relative and relative != ".":
                    names.add(relative.split("/", 1)[0])
        names.update(attrs.get("labels", []))

        if any(parent == "" or path == parent or path.startswith(parent + "/")
               for parent in self.consolidated_groups):
            return sorted(names)
        try:
            if is_s3_http_url(self.base):
                names.update(self.s3_children(path))
            else:
                entries = self.fs.ls(self.url(path), detail=True)
                names.update(entry["name"].replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
                             for entry in entries if entry["type"] == "directory")
        except (OSError, ValueError) as error:
            if not names:
                raise ValueError(
                    f"Cannot list group {self.url(path)}. "
                    "Use --path with a known array path or a store with consolidated metadata."
                ) from error
            print(f"Warning: Cannot list {self.url(path)}; showing paths referenced in metadata.", file=sys.stderr)
        return sorted(names)

    def inspect(self, path="", include_attrs=False, inherited_axes=None):
        data, attrs = self.metadata(path)
        info = {"path": "/" + path if path else "/"}
        if data.get("node_type") == "array" or "shape" in data:
            dtype = data.get("data_type", data.get("dtype"))
            try:
                dtype = str(np.dtype(dtype))
            except (TypeError, ValueError):
                dtype = str(dtype)
            chunks = data.get("chunks")
            if chunks is None:
                chunks = data["chunk_grid"]["configuration"]["chunk_shape"]
            info.update(type="array", shape=data["shape"], ndim=len(data["shape"]), chunks=chunks, dtype=dtype)
            for codec in data.get("codecs", []):
                if codec["name"] == "sharding_indexed":
                    info["shards"] = chunks
                    info["chunks"] = codec["configuration"]["chunk_shape"]
                    break
            axes = data.get("dimension_names") or attrs.get("_ARRAY_DIMENSIONS") or inherited_axes
            if axes is not None and len(axes) == info["ndim"]:
                info["axes"] = axes
        else:
            info["type"] = "group"
            multiscales = attrs.get("multiscales", []) or attrs.get("ome", {}).get("multiscales", [])
            axes = inherited_axes
            if multiscales and multiscales[0].get("axes"):
                axes = [axis.get("name") if isinstance(axis, dict) else axis for axis in multiscales[0]["axes"]]
            info["children"] = []
            for name in self.children(path, attrs):
                try:
                    child = self.inspect(join_path(path, name), include_attrs, axes)
                except FileNotFoundError:
                    # A directory listing can also contain non-Zarr directories.
                    continue
                info["children"].append(child)
        if include_attrs:
            info["attrs"] = attrs
        return info


def print_tree(info, depth=0):
    indent = "  " * depth
    name = info["path"].rstrip("/").rsplit("/", 1)[-1] or "/"
    line = f"{indent}{name} [{info['type']}]"
    if info["type"] == "array":
        line += (
            f" shape={tuple(info['shape'])}"
            f" ndim={info['ndim']}"
            f" chunks={tuple(info['chunks'])}"
            f" dtype={info['dtype']}"
        )
        if "axes" in info:
            line += f" axes={tuple(info['axes'])}"
        if "shards" in info:
            line += f" shards={tuple(info['shards'])}"
    print(line)
    if "attrs" in info:
        print(f"{indent}  attrs: {json.dumps(info['attrs'], ensure_ascii=False)}")
    for child in info.get("children", []):
        print_tree(child, depth + 1)


def main():
    parser = argparse.ArgumentParser(
        description="Print local or remote Zarr groups and array dimensions without reading array data."
    )
    parser.add_argument("source", help="Local directory path or URL of a Zarr group or array (Zarr v2 or v3).")
    parser.add_argument("--path", default="", help="Inspect only this group or array inside the store.")
    parser.add_argument("--attrs", action="store_true", help="Also print group and array attributes.")
    parser.add_argument("--json", action="store_true", help="Print the metadata as JSON instead of a tree.")
    args = parser.parse_args()

    try:
        path = args.path.strip("/")
        info = MetadataReader(args.source).inspect("" if path == "." else path, args.attrs)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    if args.json:
        print(json.dumps(info, indent=2, ensure_ascii=False))
    else:
        print_tree(info)


if __name__ == "__main__":
    main()
