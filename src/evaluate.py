"""Generation + evaluation using the repo-provided metrics.

Two metrics (from ``metrics/``):
  * Valid Syntax Rate (VSR): does the generated code execute and yield a solid?
  * Best IoU: voxel intersection-over-union between the meshes built from the
    generated code and the ground-truth code (after principal-axis alignment).

We report VSR, mean IoU over the *valid* predictions (the convention used by the
provided ``evaluate_codes``) and mean IoU over *all* samples (invalid -> 0),
which is the stricter, harder-to-game number.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.best_iou import get_iou_best
from metrics.valid_syntax_rate import evaluate_syntax_rate_simple


def postprocess(text: str) -> str:
    """Light cleanup of decoded text into runnable CadQuery code."""
    text = text.strip()
    # Ensure the cadquery import is present (decoders sometimes drop the header).
    if "import cadquery" not in text:
        text = "import cadquery as cq\n" + text
    return text


@torch.no_grad()
def generate_predictions(
    model,
    image_processor,
    tokenizer,
    samples,
    device,
    batch_size: int = 8,
    max_new_tokens: int = 512,
    num_beams: int = 1,
    no_repeat_ngram_size: int = 0,
    repetition_penalty: float = 1.0,
    min_new_tokens: int = 0,
    verbose: bool = True,
):
    model.eval()
    preds: dict[str, str] = {}
    t0 = time.time()
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        images = [s.image for s in batch]
        pixel_values = image_processor(images=images, return_tensors="pt").pixel_values.to(device)
        gen_kwargs = dict(
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if min_new_tokens > 0:
            gen_kwargs["min_new_tokens"] = min_new_tokens
        if no_repeat_ngram_size > 0:
            gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
        if repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = repetition_penalty
        if num_beams > 1:
            gen_kwargs["early_stopping"] = True
        gen = model.generate(**gen_kwargs)
        texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for s, txt in zip(batch, texts):
            preds[s.deepcad_id] = postprocess(txt)
        if verbose:
            done = min(start + batch_size, len(samples))
            print(f"  generated {done}/{len(samples)} ({time.time()-t0:.0f}s)", flush=True)
    return preds


def compute_metrics(preds: dict[str, str], gts: dict[str, str], verbose: bool = True):
    """VSR over preds + IoU (valid-only and all-samples conventions)."""
    vsr = evaluate_syntax_rate_simple(preds)

    ious_valid = []
    ious_all = []
    for _id, gt_code in gts.items():
        pred_code = preds.get(_id)
        iou = 0.0
        ok = False
        if pred_code is not None:
            try:
                iou = float(get_iou_best(gt_code, pred_code))
                ok = True
            except Exception:
                ok = False
        ious_all.append(iou)
        if ok:
            ious_valid.append(iou)
        if verbose:
            print(f"  {_id}: iou={iou:.3f} ok={ok}", flush=True)

    n = len(gts)
    mean_iou_all = sum(ious_all) / n if n else 0.0
    mean_iou_valid = sum(ious_valid) / len(ious_valid) if ious_valid else 0.0
    return {
        "vsr": vsr,
        "mean_iou_all": mean_iou_all,
        "mean_iou_valid": mean_iou_valid,
        "n_iou_pairs_valid": len(ious_valid),
        "n_total": n,
    }
