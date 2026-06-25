"""Aggregate baseline vs enhanced results into a table + plots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"


def load(name):
    p = RESULTS / f"{name}_metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    baseline = load("baseline")
    enhanced = load("enhanced")
    if not baseline or not enhanced:
        print("Need both baseline and enhanced metrics first.")
        return

    rows = [
        ("Valid Syntax Rate", baseline["vsr"], enhanced["vsr"]),
        ("Mean IoU (all samples)", baseline["mean_iou_all"], enhanced["mean_iou_all"]),
        ("Mean IoU (valid only)", baseline["mean_iou_valid"], enhanced["mean_iou_valid"]),
    ]

    print(f"{'Metric':<26}{'Baseline':>12}{'Enhanced':>12}{'Delta':>12}")
    print("-" * 62)
    md = ["| Metric | Baseline | Enhanced | Δ |", "|---|---|---|---|"]
    for name, b, e in rows:
        delta = e - b
        print(f"{name:<26}{pct(b):>12}{pct(e):>12}{('+' if delta>=0 else '')+pct(delta):>12}")
        md.append(f"| {name} | {pct(b)} | {pct(e)} | {('+' if delta>=0 else '')+pct(delta)} |")
    (RESULTS / "comparison_table.md").write_text("\n".join(md) + "\n")

    # ---- bar chart ----
    labels = ["VSR", "IoU (all)", "IoU (valid)"]
    bvals = [baseline["vsr"], baseline["mean_iou_all"], baseline["mean_iou_valid"]]
    evals = [enhanced["vsr"], enhanced["mean_iou_all"], enhanced["mean_iou_valid"]]
    x = range(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([i - w / 2 for i in x], bvals, w, label="Baseline", color="#9aa0a6")
    ax.bar([i + w / 2 for i in x], evals, w, label="Enhanced", color="#1a73e8")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Image -> CadQuery: Baseline vs Enhanced")
    for i, (b, e) in enumerate(zip(bvals, evals)):
        ax.text(i - w / 2, b + 0.01, f"{b:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, e + 0.01, f"{e:.2f}", ha="center", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "comparison.png", dpi=130)
    print(f"\nsaved {RESULTS/'comparison.png'}")

    # ---- loss curves ----
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    for name, color in [("baseline", "#9aa0a6"), ("enhanced", "#1a73e8")]:
        hp = RESULTS / f"{name}_train_history.json"
        if hp.exists():
            hist = json.loads(hp.read_text())
            ax2.plot([h["step"] for h in hist], [h["loss"] for h in hist], label=name, color=color)
    ax2.set_xlabel("optimizer step")
    ax2.set_ylabel("training loss")
    ax2.set_title("Training loss")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(RESULTS / "loss_curves.png", dpi=130)
    print(f"saved {RESULTS/'loss_curves.png'}")


if __name__ == "__main__":
    main()
