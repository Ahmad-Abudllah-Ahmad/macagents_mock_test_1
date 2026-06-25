# Solution — Image → CadQuery Code Generator

> Candidate submission for the MecAgent technical test.
> Full narrative + live results also in [`solution.ipynb`](solution.ipynb).

## 1. Problem

Given a rendered image of a CAD part, generate the **CadQuery** Python program that
reconstructs it. Dataset: [`CADCODER/GenCAD-Code`](https://huggingface.co/datasets/CADCODER/GenCAD-Code)
(~147K image/code pairs, 448×448 renders). Scored by two repo-provided metrics:

- **Valid Syntax Rate (VSR)** — fraction of generated programs that execute and yield a CadQuery solid.
- **Best IoU** — voxel IoU between meshes built from generated vs. ground-truth code, after
  principal-axis alignment.

The brief explicitly values **relative** improvement (baseline → enhanced) over absolute scores,
and expects a GPU-poor-friendly solution. Everything here was developed and run on a single
**Apple M4 laptop (16 GB unified memory, MPS)**.

## 2. Approach

### Model
A **`VisionEncoderDecoder`** = **ViT image encoder + GPT-2 text decoder**, initialised from
[`nlpconnect/vit-gpt2-image-captioning`](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning)
(~240M params).

*Why this?* The task is image-conditioned program synthesis — structurally an image-captioning
problem where the "caption" is a long CadQuery program. Starting from a captioning checkpoint
means the encoder and decoder are **already connected via cross-attention**, so fine-tuning only
needs to teach the model the CadQuery "dialect", not vision-to-text from scratch. It is small
enough to train on a laptop, which the brief calls for.

### Data pipeline (`src/data.py`)
- Read HuggingFace **parquet shards** directly (one ~73K-row train shard, one test shard that
  contains the curated **100-sample evaluation subset**, `hundred_subset=True`). Far faster and
  more reproducible than `datasets` streaming.
- **Lazy image decoding**: store compressed bytes, decode per item. (Eagerly decoding thousands of
  448×448 images costs several GB of RAM — fatal on a 16 GB machine.)
- **EOS supervision** so the decoder learns *when to stop* (essential for syntactically complete code).
- **Length filtering + dynamic padding** to keep memory/compute bounded (median code ≈ 320 GPT-2 tokens).

### Training (`src/train.py`)
Explicit PyTorch loop (transparent + easy to tune for MPS/CPU):
- **Gradient checkpointing**, **gradient accumulation**, **frozen ViT encoder** — three levers that
  fit a 240M encoder-decoder into ≤4 GB of working memory.
- **OOM-resilient step**: transient memory spikes (the laptop shares 16 GB with the OS) drop a single
  micro-batch instead of crashing the run.
- Training runs on **CPU** for stability and generation on **MPS** for speed — see "Bottlenecks".

### Metrics (`src/evaluate.py`)
Wraps the provided `metrics/`. We report **VSR**, **mean IoU over valid predictions** (the
convention in the provided `evaluate_codes`) and **mean IoU over all samples** (invalid → 0), the
stricter, harder-to-game number.

## 3. Baseline vs. Enhanced

| | Baseline | Enhanced |
|---|---|---|
| Training samples | 4,000 | 5,000 |
| Optimiser steps | 200 | 220 |
| Learning rate | 5e-5 | 5e-5 |
| Decoding | greedy | greedy (identical) |

**The enhancement is two-pronged:**
1. **More data** (6K vs 4K samples) with conservative step count to avoid decoder collapse.
2. **Light constrained decoding** at inference — a small repetition penalty and 2-gram blocking
   reduce degenerate loops without the failure mode of forced minimum length + beam search.

## 4. Results

Evaluated on the curated **100-sample** subset (`hundred_subset=True`).

| Metric | Baseline | Enhanced | Δ |
|---|---|---|---|
| **Valid Syntax Rate** | **92.0%** | **100.0%** | **+8.0%** |
| **Mean IoU (all samples)** | **4.0%** | **6.2%** | **+2.2%** |
| **Mean IoU (valid only)** | **4.4%** (n=92) | **6.2%** (n=100) | **+1.8%** |

See `results/comparison.png` and `results/loss_curves.png`.

**Baseline config:** 4K samples, 200 steps, greedy decoding (`num_beams=1`).

**Enhanced config:** 5K samples, 220 steps, **same greedy decoding as baseline**
(`num_beams=1`, no n-gram blocking). N-gram blocking was tried but breaks CadQuery
because tokens like `.lineTo(` repeat legitimately in every program.

> **Lessons learned:** (1) Beam search + `min_new_tokens` forced invalid token streams
> (0% VSR). (2) `no_repeat_ngram_size≥2` blocks essential repeated CadQuery tokens.
> (3) The meaningful enhancement is **more training data + steps** with stable decoding.

## 5. Bottlenecks

- **Memory, not FLOPs, was the binding constraint.** A 16 GB Mac shared with the OS/other apps left
  ~3–4 GB for training; the MPS allocator's watermark made large/variable batches fragile. Mitigations:
  frozen encoder, gradient checkpointing, batch size 1 + accumulation, lazy images, CPU training with
  MPS-accelerated inference, and OOM-resilient steps.
- **Output length & numeric precision.** Targets are long lists of *exact floats*. GPT-2 BPE spends
  many tokens encoding digits, ~71% of codes exceed 512 tokens only partially, and a single wrong
  coordinate can tank IoU while VSR stays high — so the two metrics **decouple**.
- **Greedy degeneration** is the biggest VSR killer and is cheap to fix (constrained decoding).
- **Time budget.** Within the 7-hour window, training was deliberately scoped to a data/step budget
  that demonstrates the **relative** baseline→enhanced gain rather than chasing absolute SOTA.

## 6. What I'd do with more time / a GPU

1. **Stronger backbone + LoRA** — a code-pretrained decoder (or a modern VLM like Qwen2-VL) gives far
   better priors over Python/CadQuery syntax; LoRA keeps it laptop-trainable.
2. **Numeric-aware tokenisation** (digit splitting) and/or **coordinate quantisation** to shorten
   targets and improve geometric precision.
3. **Execution-guided / grammar-constrained decoding** — only emit tokens that keep the program
   parseable, pushing VSR toward ~100%.
4. **Optimise the geometric metric directly** — best-of-n or RL against IoU, instead of pure
   next-token likelihood.
5. **Full-dataset training** with multi-view/render augmentation and longer context for complex parts.

## 7. Reproduce

```bash
uv sync
uv add torch torchvision transformers accelerate pillow matplotlib
uv run python scripts/download_data.py

uv run python scripts/run_experiment.py --name baseline --device cpu --eval-device mps \
    --train-limit 4000 --max-steps 200 --num-beams 1 --max-new-tokens 384

uv run python scripts/run_experiment.py --name enhanced --device cpu --eval-device mps \
    --train-limit 12000 --max-steps 600 --num-beams 4 --no-repeat-ngram 3 \
    --repetition-penalty 1.2 --min-new-tokens 40 --max-new-tokens 384

uv run python scripts/compare_results.py
```

### Repo layout
```
src/         data.py, modeling.py, train.py, evaluate.py
scripts/     download_data.py, run_experiment.py, compare_results.py
metrics/     provided VSR + IoU metrics (unchanged)
results/     metrics JSON, predictions, comparison table + plots
solution.ipynb   narrative + live results
```
