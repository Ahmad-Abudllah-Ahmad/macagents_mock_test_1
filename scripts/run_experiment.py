"""End-to-end experiment runner: load data -> (train) -> evaluate -> save results.

Examples
--------
Baseline (short fine-tune, greedy decoding):
    uv run python scripts/run_experiment.py --name baseline \
        --train-limit 3000 --max-steps 500 --num-beams 1

Enhanced (more data + longer + beam/constrained decoding):
    uv run python scripts/run_experiment.py --name enhanced \
        --train-limit 9000 --max-steps 1800 --num-beams 5 \
        --no-repeat-ngram 3 --min-new-tokens 40
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# The xet downloader needs ~2x temp space; on a near-full disk the classic
# downloader is more reliable.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch

from src.data import DATA_DIR, CadCodeDataset, load_samples
from src.evaluate import compute_metrics, generate_predictions
from src.modeling import build_model, get_device
from src.train import TrainConfig, train

RESULTS_DIR = REPO_ROOT / "results"
CKPT_DIR = REPO_ROOT / "checkpoints"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--train-parquet", default=str(DATA_DIR / "train_0000.parquet"))
    p.add_argument("--test-parquet", default=str(DATA_DIR / "test_0000.parquet"))
    p.add_argument("--train-limit", type=int, default=3000)
    p.add_argument("--max-token-count", type=int, default=1100,
                   help="drop training codes whose dataset token_count exceeds this")
    p.add_argument("--max-target-length", type=int, default=512)
    # training
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--no-freeze-encoder", dest="freeze_encoder", action="store_false",
                   help="train the ViT encoder too (needs more memory)")
    p.set_defaults(freeze_encoder=True)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--load-checkpoint", default=None)
    p.add_argument("--save-checkpoint", action="store_true",
                   help="persist the fine-tuned weights (off by default to save disk)")
    # generation
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--num-beams", type=int, default=1)
    p.add_argument("--no-repeat-ngram", type=int, default=0)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--min-new-tokens", type=int, default=0)
    p.add_argument("--eval-limit", type=int, default=100)
    p.add_argument("--device", default=None, help="training device, e.g. cpu / mps")
    p.add_argument("--eval-device", default=None, help="generation device (default: mps if available)")
    return p.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    CKPT_DIR.mkdir(exist_ok=True)
    device = get_device(args.device)
    print(f"[{args.name}] device={device}", flush=True)

    # ---- model ----
    ckpt = args.load_checkpoint or None
    model, image_processor, tokenizer = build_model(ckpt) if ckpt else build_model()
    print(f"[{args.name}] params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

    # ---- data ----
    print(f"[{args.name}] loading test (hundred_subset)...", flush=True)
    test_samples = load_samples(
        args.test_parquet, only_hundred_subset=True, limit=args.eval_limit
    )
    print(f"[{args.name}] test samples: {len(test_samples)}", flush=True)
    gts = {s.deepcad_id: s.code for s in test_samples}

    history = []
    if not args.eval_only:
        print(f"[{args.name}] loading train (limit={args.train_limit})...", flush=True)
        train_samples = load_samples(
            args.train_parquet,
            limit=args.train_limit,
            max_token_count=args.max_token_count,
        )
        print(f"[{args.name}] train samples: {len(train_samples)}", flush=True)
        train_ds = CadCodeDataset(
            train_samples, image_processor, tokenizer,
            max_target_length=args.max_target_length,
        )
        cfg = TrainConfig(
            lr=args.lr,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            max_steps=args.max_steps,
            log_every=args.log_every,
            freeze_encoder=args.freeze_encoder,
        )
        history = train(model, train_ds, tokenizer, device, cfg)
        if args.save_checkpoint:
            out_ckpt = CKPT_DIR / args.name
            model.save_pretrained(out_ckpt)
            tokenizer.save_pretrained(out_ckpt)
            image_processor.save_pretrained(out_ckpt)
            print(f"[{args.name}] saved checkpoint -> {out_ckpt}", flush=True)
    else:
        model.to(device)

    # ---- evaluation ----
    # Re-enable the KV cache (disabled during training for grad-checkpointing) so
    # autoregressive generation runs at a sane speed.
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.config.use_cache = True

    import torch as _torch
    if args.eval_device:
        eval_device = _torch.device(args.eval_device)
    elif _torch.backends.mps.is_available():
        eval_device = _torch.device("mps")
    else:
        eval_device = device

    def _run_eval(dev):
        model.to(dev)
        return generate_predictions(
            model, image_processor, tokenizer, test_samples, dev,
            batch_size=args.eval_batch_size,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            no_repeat_ngram_size=args.no_repeat_ngram,
            repetition_penalty=args.repetition_penalty,
            min_new_tokens=args.min_new_tokens,
        )

    print(f"[{args.name}] generating predictions on {eval_device}...", flush=True)
    t0 = time.time()
    try:
        preds = _run_eval(eval_device)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() and eval_device.type != "cpu":
            print(f"[{args.name}] eval OOM on {eval_device}, falling back to cpu", flush=True)
            if eval_device.type == "mps":
                _torch.mps.empty_cache()
            preds = _run_eval(_torch.device("cpu"))
        else:
            raise
    gen_time = time.time() - t0
    print(f"[{args.name}] computing metrics (VSR + IoU)...", flush=True)
    metrics = compute_metrics(preds, gts, verbose=False)
    metrics["gen_time_s"] = gen_time
    metrics["config"] = vars(args)

    (RESULTS_DIR / f"{args.name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    (RESULTS_DIR / f"{args.name}_predictions.json").write_text(json.dumps(preds, indent=2))
    if history:
        (RESULTS_DIR / f"{args.name}_train_history.json").write_text(json.dumps(history, indent=2))

    print("=" * 60, flush=True)
    print(f"[{args.name}] RESULTS", flush=True)
    print(f"  VSR             : {metrics['vsr']:.3f}", flush=True)
    print(f"  Mean IoU (all)  : {metrics['mean_iou_all']:.3f}", flush=True)
    print(f"  Mean IoU (valid): {metrics['mean_iou_valid']:.3f} (n={metrics['n_iou_pairs_valid']})", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
