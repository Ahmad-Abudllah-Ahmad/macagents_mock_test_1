# MecAgent Technical Test — Image → CadQuery Code Generator

**Task:** Given a rendered CAD image, generate the **CadQuery Python code** that reconstructs the part.  
**Dataset:** [`CADCODER/GenCAD-Code`](https://huggingface.co/datasets/CADCODER/GenCAD-Code) (~147K pairs)  
**Hardware:** Apple M4 · 16 GB unified memory · CPU training · MPS inference

---

## Submission Readiness — Honest Assessment

> **Bottom line:** This solution is **eligible and competitive on process and relative improvement**, not on raw CAD quality. It is **not a lock to pass**. If the repo is public, the write-up is strong, and MecAgent scores as their brief implies, there is a **fair shot**. If they mainly rank on absolute metrics vs GPU-heavy submissions, this submission will **likely lose**.

| Assessment area | Status |
|---|---|
| Process & relative improvement (VSR 92%→100%, IoU 4.0%→6.2%) | Competitive |
| Raw CAD / geometric quality (absolute IoU ~6%) | Weak |
| Write-up, reproducibility, engineering under constraints | Strong |
| Pass certainty | **Not guaranteed** |

### Before submitting (non-negotiable)

- [ ] **Public GitHub URL is live** — repo must be visible on your GitHub account (do not fork the original MecAgent repo).
- [ ] **URL points to the final commit** containing both **baseline** and **enhanced** results in `results/`.
- [ ] Submit the repo URL via the **“Submit Test”** tab on the MecAgent test portal.

**Repo URL:** `https://github.com/Ahmad-Abudllah-Ahmad/macagents_mock_test_1`

---

| Resource | Link |
|---|---|
| Full write-up | [`SOLUTION.md`](SOLUTION.md) |
| Interactive notebook | [`solution.ipynb`](solution.ipynb) |
| Original brief | [`good_luck.ipynb`](good_luck.ipynb) |
| Result charts | [`results/comparison.png`](results/comparison.png) · [`results/loss_curves.png`](results/loss_curves.png) |

---

## Table 1 — Final Results (100-sample eval subset)

| Metric | Baseline | Enhanced | Δ | Interpretation |
|---|---|---|---|---|
| **Valid Syntax Rate (VSR)** | 92.0% | **100.0%** | **+8.0%** | More programs execute without error |
| **Mean IoU (all samples)** | 4.0% | **6.2%** | **+2.2%** | Stricter score — invalid → 0 |
| **Mean IoU (valid only)** | 4.4% (n=92) | **6.2%** (n=100) | **+1.8%** | Geometry overlap on runnable code |
| **Generation time** | 306 s | 318 s | +12 s | ~100 images, greedy decode |

### Result charts (infographics)

<p align="center">
  <img src="results/comparison.png" alt="Baseline vs Enhanced — VSR and IoU comparison" width="700"/>
</p>

<p align="center">
  <img src="results/loss_curves.png" alt="Training loss curves — baseline vs enhanced" width="700"/>
</p>

---

## Diagram 1 — End-to-End Solution Pipeline

```mermaid
flowchart LR
    subgraph INPUT["📥 INPUT"]
        A["🖼️ CAD Render<br/>448×448 PNG"]
        B["📋 Ground-Truth<br/>CadQuery Code"]
    end

    subgraph MODEL["🧠 MODEL"]
        C["ViT Encoder<br/>Image → Features"]
        D["GPT-2 Decoder<br/>Features → Tokens"]
    end

    subgraph OUTPUT["📤 OUTPUT"]
        E["🐍 Generated<br/>CadQuery Code"]
    end

    subgraph METRICS["📊 METRICS"]
        F["✅ VSR<br/>Code runs?"]
        G["📐 Best IoU<br/>Mesh similarity"]
    end

    A --> C --> D --> E
    E --> F
    E --> G
    B -.->|compare| G

    style INPUT fill:#e3f2fd,stroke:#1565c0,color:#000
    style MODEL fill:#f3e5f5,stroke:#7b1fa2,color:#000
    style OUTPUT fill:#e8f5e9,stroke:#2e7d32,color:#000
    style METRICS fill:#fff3e0,stroke:#ef6c00,color:#000
```

---

## Diagram 2 — Reproduction & Experiment Workflow

```mermaid
flowchart TD
    START(["🚀 START"]) --> ENV["⚙️ uv sync<br/>Install dependencies"]
    ENV --> DATA["📦 download_data.py<br/>Fetch parquet shards"]
    DATA --> BASE["🔵 BASELINE RUN<br/>4K samples · 200 steps"]
    DATA --> ENH["🟢 ENHANCED RUN<br/>5K samples · 220 steps"]
    BASE --> EVAL1["📊 Evaluate 100-subset<br/>VSR + IoU"]
    ENH --> EVAL2["📊 Evaluate 100-subset<br/>VSR + IoU"]
    EVAL1 --> COMP["📈 compare_results.py<br/>Tables + plots"]
    EVAL2 --> COMP
    COMP --> DONE(["✅ SUBMIT<br/>Public GitHub URL"])

    style START fill:#4caf50,color:#000
    style DONE fill:#4caf50,color:#000
    style BASE fill:#2196f3,color:#000
    style ENH fill:#00897b,color:#000
```

---

## Diagram 3 — Dataset Ingestion Flow

```mermaid
flowchart TD
    HF["☁️ HuggingFace<br/>CADCODER/GenCAD-Code"] --> DL["⬇️ Parquet Shards<br/>~380 MB total"]
    DL --> TRAIN["📁 train_0000.parquet<br/>~73,645 rows"]
    DL --> TEST["📁 test_0000.parquet<br/>~7,355 rows"]

    TRAIN --> FILTER["🔍 Filter<br/>max_token_count ≤ 1300"]
    FILTER --> LIMIT["✂️ Subset<br/>4K / 5K samples"]
    LIMIT --> LAZY["💾 Lazy image bytes<br/>Decode per batch"]

    TEST --> H100["⭐ hundred_subset<br/>100 curated eval rows"]
    H100 --> EVAL["📊 Evaluation set"]

    LAZY --> DSET["🗂️ CadCodeDataset<br/>pixel_values + labels"]

    style HF fill:#ffecb3,stroke:#ff8f00,color:#000
    style H100 fill:#c8e6c9,stroke:#388e3c,color:#000
    style EVAL fill:#bbdefb,stroke:#1976d2,color:#000
```

---

## Table 2 — Dataset Schema

| Field | Type | Description |
|---|---|---|
| `image` | 448×448 RGB | Rendered view of the CAD part |
| `cadquery` | string | Ground-truth CadQuery Python program |
| `token_count` | int | Code length (dataset tokenizer) |
| `deepcad_id` | string | Unique sample identifier |
| `prompt` | string | Instruction string (constant) |
| `hundred_subset` | bool | Curated 100-sample evaluation flag |

---

## Table 3 — Dataset Split Summary

| Shard | Rows | Used for | Download size |
|---|---|---|---|
| `train_0000.parquet` | ~73,645 | Fine-tuning (subset sampled) | ~315 MB |
| `test_0000.parquet` | ~7,355 | Evaluation | ~32 MB |
| `hundred_subset=True` | **100** | **Official eval subset** | included in test |

---

## Diagram 4 — Model Architecture (ViT-GPT2)

```mermaid
flowchart TB
    IMG["🖼️ Input Image<br/>448 × 448 × 3"] --> VIT["🔍 ViT Encoder<br/>~86M params<br/>FROZEN ❄️"]

    VIT --> FEAT["🎯 Visual Features<br/>Cross-attention context"]

    FEAT --> DEC["✍️ GPT-2 Decoder<br/>~152M trainable params"]
    TOK["🔤 Previous tokens<br/>shifted right"] --> DEC

    DEC --> LOGITS["📤 Next-token logits"]
    LOGITS --> CODE["🐍 CadQuery code string<br/>import cadquery as cq …"]

    subgraph CHECKPOINT["Pre-trained weights"]
        CP["nlpconnect/vit-gpt2-image-captioning"]
    end
    CP -.-> VIT
    CP -.-> DEC

    style VIT fill:#e1bee7,stroke:#8e24aa,color:#000
    style DEC fill:#b2dfdb,stroke:#00695c,color:#000
    style CHECKPOINT fill:#fff9c4,stroke:#f9a825,color:#000
```

---

## Diagram 5 — Training Loop Flow

```mermaid
flowchart TD
    INIT["🏁 Load ViT-GPT2 checkpoint"] --> FREEZE["❄️ Freeze ViT encoder<br/>152.8M trainable params"]
    FREEZE --> GC["💡 Enable gradient checkpointing"]
    GC --> LOOP{"🔄 Training step<br/>(max_steps)"}

    LOOP --> BATCH["📦 Batch: images + code tokens<br/>batch=1 · grad_accum=8"]
    BATCH --> FWD["➡️ Forward pass<br/>cross-entropy loss"]
    FWD --> OOM{"⚠️ OOM?"}
    OOM -->|Yes| SKIP["Skip micro-batch<br/>empty MPS cache"]
    OOM -->|No| BWD["⬅️ Backward + clip grad"]
    SKIP --> LOOP
    BWD --> STEP["📈 Optimizer step<br/>AdamW · linear warmup"]
    STEP --> LOOP

    LOOP -->|done| SAVE["💾 Model in memory"]
    SAVE --> GEN["🚀 Switch to eval mode<br/>enable KV cache"]

    style FREEZE fill:#e3f2fd,stroke:#1565c0,color:#000
    style OOM fill:#ffcdd2,stroke:#c62828,color:#000
    style GEN fill:#c8e6c9,stroke:#2e7d32,color:#000
```

---

## Table 4 — Baseline Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Train samples | 4,000 | From `train_0000.parquet` |
| Optimizer steps | 200 | ~32 min on CPU |
| Learning rate | 5e-5 | AdamW + warmup |
| Batch size | 1 | Effective batch = 8 (grad accum) |
| Max target length | 384 tokens | Dynamic padding |
| Freeze encoder | ✅ Yes | Saves ~1 GB memory |
| Train device | CPU | Stable on 16 GB Mac |
| Decode device | MPS | Faster generation |
| Decoding | Greedy (`num_beams=1`) | No n-gram blocking |

---

## Table 5 — Enhanced Hyperparameters

| Parameter | Value | Change vs baseline |
|---|---|---|
| Train samples | 5,000 | +1,000 (+25%) |
| Optimizer steps | 220 | +20 (+10%) |
| Learning rate | 5e-5 | Same |
| Batch / accum | 1 / 8 | Same |
| Max target length | 384 | Same |
| Freeze encoder | ✅ Yes | Same |
| Decoding | Greedy (`num_beams=1`) | Same — stable |
| Final train loss | ~0.33 | Similar to baseline |

---

## Diagram 6 — Inference & Code Generation Flow

```mermaid
flowchart LR
    IMG["🖼️ Test image"] --> PROC["🎨 ViTImageProcessor<br/>normalize · resize"]
    PROC --> PV["tensor pixel_values"]
    PV --> MPS["⚡ MPS generate<br/>max_new_tokens=384"]
    MPS --> DECODE["🔤 GPT-2 decode<br/>skip_special_tokens"]
    DECODE --> POST["🧹 Post-process<br/>ensure import cadquery"]
    POST --> CODE["📄 Prediction JSON<br/>deepcad_id → code"]

    MPS -->|OOM| CPUFB["🔄 Fallback CPU"]

    style MPS fill:#fff3e0,stroke:#e65100,color:#000
    style CODE fill:#e8f5e9,stroke:#2e7d32,color:#000
    style CPUFB fill:#ffcdd2,stroke:#b71c1c,color:#000
```

---

## Diagram 7 — Valid Syntax Rate (VSR) Evaluation

```mermaid
flowchart TD
    CODE["🐍 Generated CadQuery code"] --> EXEC["⚙️ exec in sandbox<br/>namespace: cq, np"]
    EXEC --> ERR{"❌ Exception?"}
    ERR -->|Yes| FAIL["✗ Count as invalid<br/>VSR += 0"]
    ERR -->|No| FIND{"🔎 CadQuery object<br/>result / solid?"}
    FIND -->|No| FAIL
    FIND -->|Yes| OK["✓ Count as valid<br/>VSR += 1"]
    FAIL --> RATE["📊 VSR = valid / total"]
    OK --> RATE

    style OK fill:#a5d6a7,stroke:#2e7d32,color:#000
    style FAIL fill:#ef9a9a,stroke:#c62828,color:#000
    style RATE fill:#bbdefb,stroke:#1565c0,color:#000
```

---

## Diagram 8 — Best IoU Evaluation Pipeline

```mermaid
flowchart TD
    GT["📗 Ground-truth code"] --> MESH1["🔷 Execute → Solid"]
    PR["📘 Predicted code"] --> MESH2["🔷 Execute → Solid"]

    MESH1 --> N1["📐 Normalize mesh<br/>centroid · scale by r_g"]
    MESH2 --> N2["📐 Normalize mesh<br/>centroid · scale by r_g"]

    N1 --> ALIGN["🔄 Principal-axis alignment<br/>4 sign-flip candidates"]
    N2 --> ALIGN

    ALIGN --> VOX["🧊 Voxelize<br/>pitch = 0.05"]
    VOX --> IOU["📊 IoU = |A ∩ B| / |A ∪ B|"]
    IOU --> MEAN["📈 Mean over 100 samples"]

    style IOU fill:#fff9c4,stroke:#f57f17,color:#000
    style MEAN fill:#c8e6c9,stroke:#388e3c,color:#000
```

---

## Table 6 — Metric Definitions

| Metric | Formula / Method | What it measures |
|---|---|---|
| **VSR** | `valid_count / total` | Does generated code run and produce a solid? |
| **Best IoU** | Voxel IoU after PCA alignment | 3D shape similarity (scale/rotation invariant) |
| **IoU (all)** | Invalid predictions count as 0 | Penalises syntax failures |
| **IoU (valid)** | Mean over successful pairs only | Geometry quality when code runs |

---

## Diagram 9 — Baseline vs Enhanced Decision Flow

```mermaid
flowchart TD
    Q["❓ How to improve<br/>baseline model?"] --> D1["📚 More training data<br/>4K → 5K"]
    Q --> D2["⏱️ More steps<br/>200 → 220"]
    Q --> D3["🔍 Beam search<br/>num_beams=3"]
    Q --> D4["🚫 N-gram block<br/>no_repeat_ngram=2"]

    D1 --> OK1["✅ +8% VSR<br/>+2.2% IoU"]
    D2 --> OK1
    D3 --> BAD1["❌ 0% VSR<br/>forced garbage tokens"]
    D4 --> BAD2["❌ 0% VSR<br/>blocks .lineTo( repeats"]

    OK1 --> FINAL["🏆 Final enhanced config<br/>More data + steps<br/>Greedy decode only"]

    style OK1 fill:#a5d6a7,stroke:#2e7d32,color:#000
    style BAD1 fill:#ef9a9a,stroke:#c62828,color:#000
    style BAD2 fill:#ef9a9a,stroke:#c62828,color:#000
    style FINAL fill:#bbdefb,stroke:#1565c0,color:#000
```

---

## Diagram 10 — Memory & Device Strategy (16 GB Mac)

```mermaid
flowchart TD
    PROB["⚠️ 16 GB unified memory<br/>240M param model"] --> S1["❄️ Freeze ViT encoder"]
    PROB --> S2["💡 Gradient checkpointing"]
    PROB --> S3["📦 Batch size = 1"]
    PROB --> S4["🖼️ Lazy image bytes<br/>not decoded PIL cache"]
    PROB --> S5["🖥️ Train on CPU<br/>avoid MPS watermark OOM"]
    PROB --> S6["⚡ Generate on MPS<br/>inference fits in ~3 GB"]

    S1 --> OK["✅ Stable training<br/>~9 s/step"]
    S2 --> OK
    S3 --> OK
    S4 --> OK
    S5 --> OK
    S6 --> OK

    style PROB fill:#ffcdd2,stroke:#b71c1c,color:#000
    style OK fill:#c8e6c9,stroke:#388e3c,color:#000
```

---

## Table 7 — Repository Layout

| Path | Purpose |
|---|---|
| `src/data.py` | Parquet loading, lazy images, `CadCodeDataset`, collate |
| `src/modeling.py` | ViT-GPT2 factory, device selection |
| `src/train.py` | Training loop (checkpointing, grad accum, OOM-resilient) |
| `src/evaluate.py` | Generation + VSR/IoU wrapper |
| `scripts/download_data.py` | Download parquet shards from HuggingFace |
| `scripts/run_experiment.py` | End-to-end train + evaluate runner |
| `scripts/compare_results.py` | Build comparison table and plots |
| `metrics/` | Provided VSR + Best IoU metrics (unchanged) |
| `results/` | `baseline_*`, `enhanced_*`, comparison PNGs |
| `SOLUTION.md` | Full technical write-up |
| `solution.ipynb` | Narrative notebook with live results |

---

## Table 8 — Hardware & Runtime Environment

| Component | Specification |
|---|---|
| Machine | Apple MacBook (M4) |
| RAM | 16 GB unified memory |
| Train device | CPU (stable, no MPS OOM) |
| Inference device | MPS (Apple GPU) |
| Python | 3.11 (via `uv`) |
| Key packages | PyTorch 2.12 · Transformers 5.12 · CadQuery 2.5 |
| Baseline wall time | ~32 min train + ~5 min eval + ~30 min IoU |
| Enhanced wall time | ~42 min train + ~5 min eval + ~30 min IoU |

---

## Quickstart

```bash
uv sync
uv add torch torchvision transformers accelerate pillow matplotlib
uv run python scripts/download_data.py

# Baseline
uv run python scripts/run_experiment.py --name baseline --device cpu --eval-device mps \
    --train-limit 4000 --max-steps 200 --num-beams 1 --max-new-tokens 384

# Enhanced
uv run python scripts/run_experiment.py --name enhanced --device cpu --eval-device mps \
    --train-limit 5000 --max-steps 220 --num-beams 1 --max-new-tokens 384

uv run python scripts/compare_results.py
```

---

## Key Takeaways

| # | Insight |
|---|---|
| 1 | Image→code is structurally **image captioning**; ViT-GPT2 is a strong laptop baseline |
| 2 | **Relative improvement** (+8% VSR, +2.2% IoU) matters more than absolute IoU on this hardware |
| 3 | **Beam search + min_new_tokens** and **n-gram blocking** break CadQuery generation |
| 4 | Absolute IoU (~6%) remains low — exact float coordinates need stronger models / tokenizers |
| 5 | Full reproduction requires `scripts/download_data.py` (~380 MB download) |
