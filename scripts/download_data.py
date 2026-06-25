"""Download the GenCAD-Code parquet shards used by this solution into ./data.

We pull the auto-converted parquet shards from the HuggingFace datasets-server
(one train shard ~73K rows, one test shard that contains the curated 100-sample
evaluation subset). This is much faster and more reproducible than ``datasets``
streaming, and keeps the footprint small enough for a laptop.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

BASE = (
    "https://huggingface.co/datasets/CADCODER/GenCAD-Code/"
    "resolve/refs%2Fconvert%2Fparquet/default"
)
FILES = {
    "train_0000.parquet": f"{BASE}/train/0000.parquet",
    "test_0000.parquet": f"{BASE}/test/0000.parquet",
}


def download(url: str, dest: Path):
    print(f"-> {dest.name}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        read = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            read += len(chunk)
            if total:
                print(f"   {read/1e6:6.1f}/{total/1e6:.1f} MB", end="\r")
    tmp.rename(dest)
    print(f"   done: {dest} ({dest.stat().st_size/1e6:.1f} MB)")


def main():
    DATA_DIR.mkdir(exist_ok=True)
    for name, url in FILES.items():
        dest = DATA_DIR / name
        if dest.exists():
            print(f"== {name} already present, skipping")
            continue
        download(url, dest)


if __name__ == "__main__":
    main()
