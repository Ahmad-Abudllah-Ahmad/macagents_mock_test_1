"""Model factory for the image -> CadQuery code generator.

We use a ``VisionEncoderDecoderModel`` (ViT image encoder + GPT-2 text decoder).
Starting from the ``nlpconnect/vit-gpt2-image-captioning`` checkpoint gives us a
vision encoder and an autoregressive text decoder that already "talk" to each
other through cross-attention, so fine-tuning on (image, code) pairs only has to
adapt the model to the CadQuery "language" rather than learn captioning from
scratch. This is a pragmatic, laptop-friendly baseline (~240M params, runs on
Apple-silicon MPS).
"""

from __future__ import annotations

import torch
from transformers import (
    AutoTokenizer,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
)

BASE_CHECKPOINT = "nlpconnect/vit-gpt2-image-captioning"


def get_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(checkpoint: str = BASE_CHECKPOINT):
    """Return (model, image_processor, tokenizer) wired for generation."""
    image_processor = ViTImageProcessor.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = VisionEncoderDecoderModel.from_pretrained(checkpoint)

    # Make sure special tokens are consistent for training + generation.
    model.config.decoder_start_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id or tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    return model, image_processor, tokenizer
