"""Dataset loading and preprocessing for the GenCAD-Code image -> CadQuery task.

We read the HuggingFace parquet shards directly (downloaded once into ``data/``)
instead of using ``datasets`` streaming, which is far faster and reproducible on a
laptop. Each row contains:

    image          : a 448x448 rendered view of a CAD part (PNG/JPEG bytes)
    cadquery       : the ground-truth CadQuery python code that builds the part
    token_count    : length of the code in the dataset's own tokenizer
    deepcad_id     : unique id
    prompt         : the (constant) instruction string
    hundred_subset : bool flag marking a curated 100-sample evaluation subset
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


@dataclass
class Sample:
    """Holds the *raw* image bytes and decodes lazily.

    Keeping compressed bytes (~4-5 KB each) instead of decoded PIL images
    (~600 KB each) is the difference between ~40 MB and several GB of RAM for a
    few thousand samples -- essential on a 16 GB machine.
    """

    image_bytes: bytes
    code: str
    deepcad_id: str
    token_count: int
    hundred_subset: bool

    def load_image(self) -> Image.Image:
        return Image.open(io.BytesIO(self.image_bytes)).convert("RGB")

    # Backwards-friendly alias so callers can use ``sample.image``.
    @property
    def image(self) -> Image.Image:
        return self.load_image()


def _extract_bytes(value) -> bytes:
    """HF stores images either as a dict {'bytes':..., 'path':...} or raw bytes."""
    if isinstance(value, dict):
        value = value.get("bytes") or value.get("path")
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, str):
        with open(value, "rb") as f:
            return f.read()
    raise TypeError(f"Unsupported image value type: {type(value)}")


def load_samples(
    parquet_path: str | Path,
    limit: Optional[int] = None,
    max_token_count: Optional[int] = None,
    only_hundred_subset: bool = False,
) -> list[Sample]:
    """Load samples from a parquet shard with optional filtering.

    Args:
        parquet_path: path to a downloaded parquet shard.
        limit: keep at most this many samples (after filtering).
        max_token_count: drop samples whose ``token_count`` exceeds this (keeps
            training memory/time bounded and removes pathologically long codes).
        only_hundred_subset: keep only rows flagged as the curated 100-subset.
    """
    table = pq.read_table(parquet_path)
    cols = table.column_names
    n = table.num_rows

    images = table.column("image").to_pylist()
    codes = table.column("cadquery").to_pylist()
    ids = table.column("deepcad_id").to_pylist() if "deepcad_id" in cols else [str(i) for i in range(n)]
    token_counts = table.column("token_count").to_pylist() if "token_count" in cols else [0] * n
    hundred = (
        table.column("hundred_subset").to_pylist() if "hundred_subset" in cols else [False] * n
    )

    samples: list[Sample] = []
    for i in range(n):
        if only_hundred_subset and not hundred[i]:
            continue
        if max_token_count is not None and token_counts[i] is not None and token_counts[i] > max_token_count:
            continue
        samples.append(
            Sample(
                image_bytes=_extract_bytes(images[i]),
                code=codes[i],
                deepcad_id=str(ids[i]),
                token_count=int(token_counts[i] or 0),
                hundred_subset=bool(hundred[i]),
            )
        )
        if limit is not None and len(samples) >= limit:
            break
    return samples


class CadCodeDataset(Dataset):
    """Turns ``Sample``s into model-ready tensors (pixel_values + token labels).

    We append the EOS token so the decoder learns *when to stop* (critical for
    producing syntactically complete code), and use dynamic padding in the
    collate function instead of padding every example to ``max_target_length``
    (the median code is ~320 tokens, so this roughly halves compute).
    """

    def __init__(self, samples, image_processor, tokenizer, max_target_length: int = 512):
        self.samples = samples
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        pixel_values = self.image_processor(images=s.image, return_tensors="pt").pixel_values[0]
        text = s.code + self.tokenizer.eos_token
        ids = self.tokenizer(
            text,
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]
        return {"pixel_values": pixel_values, "labels": ids}


def make_collate(pad_token_id: int):
    def collate(batch):
        pixel_values = torch.stack([b["pixel_values"] for b in batch])
        max_len = max(b["labels"].size(0) for b in batch)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            n = b["labels"].size(0)
            labels[i, :n] = b["labels"]
        return {"pixel_values": pixel_values, "labels": labels}

    return collate
