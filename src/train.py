"""Minimal, dependency-light fine-tuning loop for the VisionEncoderDecoder model.

Written as an explicit PyTorch loop (rather than ``Trainer``) so the mechanics are
transparent and easy to tune for Apple-silicon MPS, where memory is the main
constraint. Supports gradient accumulation, linear warmup, gradient clipping and
periodic loss logging.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .data import make_collate


@dataclass
class TrainConfig:
    lr: float = 5e-5
    weight_decay: float = 0.01
    batch_size: int = 2
    grad_accum: int = 8
    max_steps: int = 600
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    log_every: int = 20
    num_workers: int = 0
    seed: int = 42
    freeze_encoder: bool = True


def train(model, dataset, tokenizer, device, cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    model.to(device)
    model.train()

    # Gradient checkpointing trades a little compute for a large activation-memory
    # saving -- essential to fit a 240M encoder-decoder on 16GB unified memory.
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # Optionally freeze the ViT encoder: removes its gradients + optimizer state
    # (~1 GB) and speeds each step. The decoder + cross-attention still adapt to
    # the CAD "language". A reasonable GPU-poor trade-off.
    if cfg.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable/1e6:.1f}M (freeze_encoder={cfg.freeze_encoder})", flush=True)

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=make_collate(tokenizer.pad_token_id),
        num_workers=cfg.num_workers,
        drop_last=True,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_updates = cfg.max_steps
    sched = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=int(cfg.warmup_ratio * total_updates),
        num_training_steps=total_updates,
    )

    history = []
    step = 0
    running = 0.0
    t0 = time.time()
    optim.zero_grad()

    done = False
    while not done:
        for i, batch in enumerate(loader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            # Be resilient to transient MPS OOM (this laptop shares 16 GB with the
            # rest of the OS): drop the offending micro-batch and keep going.
            try:
                out = model(pixel_values=pixel_values, labels=labels)
                loss = out.loss / cfg.grad_accum
                loss.backward()
                running += out.loss.item()
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    optim.zero_grad(set_to_none=True)
                    if device.type == "mps":
                        torch.mps.empty_cache()
                    print(f"  [oom] skipped micro-batch (len={labels.size(1)})", flush=True)
                    continue
                raise

            if (i + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optim.step()
                sched.step()
                optim.zero_grad()
                if device.type == "mps":
                    torch.mps.empty_cache()
                step += 1

                if step % cfg.log_every == 0:
                    avg = running / (cfg.log_every * cfg.grad_accum)
                    running = 0.0
                    lr_now = sched.get_last_lr()[0]
                    msg = (
                        f"step {step}/{cfg.max_steps} | loss {avg:.4f} | "
                        f"lr {lr_now:.2e} | {time.time()-t0:.0f}s"
                    )
                    print(msg, flush=True)
                    history.append({"step": step, "loss": avg, "lr": lr_now})

                if step >= cfg.max_steps:
                    done = True
                    break
    print(f"training done in {time.time()-t0:.0f}s", flush=True)
    return history
