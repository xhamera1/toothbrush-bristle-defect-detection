"""
tune_multiscale_tta.py - Grid search ROI multi-scale factors for submission5/model.py

Evaluates mean IoU on the same defective + good subsets as test_model.py (first 15 each).
Prefer configs with mean good IoU == 1.0 (no false positives), then maximize mean
defective IoU and overall mean.

Usage:
    python tune_multiscale_tta.py
"""

import os
import sys
import time

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SUBMISSION_DIR = os.path.join(PROJECT_ROOT, "submission5")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

DEFECTIVE_IMG_DIR = os.path.join(DATA_DIR, "train", "defective")
GOOD_IMG_DIR = os.path.join(DATA_DIR, "train", "good")
GT_DIR = os.path.join(DATA_DIR, "ground_truth", "defective")

sys.path.insert(0, SUBMISSION_DIR)


def compute_iou(pred_mask, gt_mask):
    pred_bin = (pred_mask > 127).astype(np.uint8)
    gt_bin = (gt_mask > 127).astype(np.uint8)
    intersection = np.sum(pred_bin & gt_bin)
    union = np.sum(pred_bin | gt_bin)
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def load_lists(limit=15):
    defective_files = sorted(
        f for f in os.listdir(DEFECTIVE_IMG_DIR) if f.endswith(".png")
    )[:limit]
    good_files = sorted(f for f in os.listdir(GOOD_IMG_DIR) if f.endswith(".png"))[
        :limit
    ]
    return defective_files, good_files


def evaluate_scales(roi_tta_scales, defective_files, good_files):
    from model import ToothbrushDefectDetector

    det = ToothbrushDefectDetector(roi_tta_scales=roi_tta_scales)
    di, gi = [], []
    for fn in defective_files:
        p = os.path.join(DEFECTIVE_IMG_DIR, fn)
        gtp = os.path.join(GT_DIR, fn.replace(".png", "_mask.png"))
        im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        gt = cv2.imread(gtp, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            continue
        pred = det.predict(im)
        di.append(compute_iou(pred, gt))
    for fn in good_files:
        p = os.path.join(GOOD_IMG_DIR, fn)
        im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        z = np.zeros(im.shape[:2], dtype=np.uint8)
        pred = det.predict(im)
        gi.append(compute_iou(pred, z))
    return float(np.mean(di)), float(np.mean(gi)), float(np.mean(di + gi))


def main():
    defective_files, good_files = load_lists(15)
    presets = [
        (1.0,),
        (0.94, 1.0, 1.06),
        (0.92, 1.0, 1.08),
        (0.90, 1.0, 1.10),
        (0.88, 1.0, 1.12),
        (0.92, 1.0),
        (1.0, 1.08),
    ]

    print("ROI TTA scale presets vs mean IoU (defective / good / all)\n")
    rows = []
    for scales in presets:
        t0 = time.time()
        md, mg, ma = evaluate_scales(scales, defective_files, good_files)
        elapsed = time.time() - t0
        rows.append((scales, md, mg, ma, elapsed))
        print(
            f"  scales={scales!s:32} | "
            f"def={md:.4f} good={mg:.4f} all={ma:.4f} | {elapsed:.1f}s"
        )

    ok = [r for r in rows if r[2] >= 0.9999]
    pool = ok if ok else rows
    best = max(pool, key=lambda x: (x[3], x[1]))
    print("\nBest (preferring good IoU ~1.0):")
    print(f"  scales={best[0]!r}")
    print(f"  defective={best[1]:.4f}  good={best[2]:.4f}  overall={best[3]:.4f}")
    print(f"\nSet ROI_TTA_SCALES in submission5/model.py to: {best[0]!r}")


if __name__ == "__main__":
    main()
