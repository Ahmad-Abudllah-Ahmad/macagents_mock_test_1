# MecAgent Technical Test — Image → CadQuery Code Generator

This repo contains my solution to the technical test: **generate CadQuery code from an image of a
CAD part**, build a baseline, enhance it, and evaluate both with the provided metrics
(Valid Syntax Rate + Best IoU).

- **Read the write-up:** [`SOLUTION.md`](SOLUTION.md)
- **Notebook (narrative + live results):** [`solution.ipynb`](solution.ipynb)
- Original task description: [`good_luck.ipynb`](good_luck.ipynb)

## Quickstart

```bash
uv sync
uv add torch torchvision transformers accelerate pillow matplotlib
uv run python scripts/download_data.py            # fetch dataset shards into ./data

# Baseline (short fine-tune + greedy decoding)
uv run python scripts/run_experiment.py --name baseline --device cpu --eval-device mps \
    --train-limit 4000 --max-steps 200 --num-beams 1 --max-new-tokens 384

# Enhanced (more data + steps, same greedy decoding)
uv run python scripts/run_experiment.py --name enhanced --device cpu --eval-device mps \
    --train-limit 5000 --max-steps 220 --num-beams 1 --max-new-tokens 384

uv run python scripts/compare_results.py          # table + plots into ./results
```

## Layout

| Path | Purpose |
|---|---|
| `src/data.py` | parquet loading, filtering, dataset/collate (lazy images) |
| `src/modeling.py` | ViT-GPT2 `VisionEncoderDecoder` factory |
| `src/train.py` | MPS/CPU-friendly training loop (grad checkpointing, accumulation, OOM-resilient) |
| `src/evaluate.py` | generation + VSR/IoU metrics |
| `scripts/` | `download_data.py`, `run_experiment.py`, `compare_results.py` |
| `metrics/` | provided metrics (unchanged) |
| `results/` | metrics JSON, predictions, plots |

Developed and run end-to-end on an **Apple M4 laptop (16 GB, MPS)** — see `SOLUTION.md` for the
memory/compute trade-offs this required.
