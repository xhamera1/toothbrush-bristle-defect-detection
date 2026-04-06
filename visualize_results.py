"""
visualize_results.py - Save side-by-side visualizations for defective test images.

For each of the first N defective images (same subset as test_model.py):
  Left: original RGB
  Middle: predicted mask
  Right: ground truth mask

Each figure title includes IoU and defect percentage. Output folder is created
under the project root by default.

Usage:
    python visualize_results.py
    python visualize_results.py --out my_vis_folder
    python visualize_results.py --limit 5
"""

import argparse
import os
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

# -- Paths (aligned with test_model.py) ---------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SUBMISSION_DIR = os.path.join(PROJECT_ROOT, "submission5")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

DEFECTIVE_IMG_DIR = os.path.join(DATA_DIR, "train", "defective")
GT_DIR = os.path.join(DATA_DIR, "ground_truth", "defective")

DEFAULT_OUT_DIR = os.path.join(PROJECT_ROOT, "visualization_results_defective")

sys.path.insert(0, SUBMISSION_DIR)


def compute_iou(pred_mask, gt_mask):
    pred_bin = (pred_mask > 127).astype(np.uint8)
    gt_bin = (gt_mask > 127).astype(np.uint8)
    intersection = np.sum(pred_bin & gt_bin)
    union = np.sum(pred_bin | gt_bin)
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def defect_percentage(mask):
    total = mask.shape[0] * mask.shape[1]
    if total == 0:
        return 0.0
    return 100.0 * float(np.sum(mask == 255)) / total


def save_comparison_figure(
    image_rgb,
    pred_mask,
    gt_mask,
    out_path,
    title_line,
):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(title_line, fontsize=11, y=1.02)

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(pred_mask, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    axes[2].imshow(gt_mask, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("Ground truth")
    axes[2].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize defective-set predictions vs GT.")
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=15,
        help="Number of defective images to visualize (default: 15, same as test_model.py)",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Importing model...")
    t0 = time.time()
    from model import predict

    print(f"  Done ({time.time() - t0:.2f}s)\n")

    defective_files = sorted(
        f for f in os.listdir(DEFECTIVE_IMG_DIR) if f.endswith(".png")
    )[: args.limit]

    if not defective_files:
        print(f"No PNG files in {DEFECTIVE_IMG_DIR}")
        return

    print(f"Saving {len(defective_files)} figures to:\n  {args.out}\n")

    ious = []
    for img_name in defective_files:
        img_path = os.path.join(DEFECTIVE_IMG_DIR, img_name)
        gt_path = os.path.join(GT_DIR, img_name.replace(".png", "_mask.png"))

        image = cv2.imread(img_path)
        if image is None:
            print(f"  [SKIP] {img_name} (failed to read image)")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt_mask = (
            cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if os.path.exists(gt_path)
            else None
        )

        t1 = time.time()
        pred_mask = predict(image_rgb)
        elapsed = time.time() - t1

        if gt_mask is None:
            iou = float("nan")
            print(f"  [WARN] {img_name} | no GT mask | time={elapsed:.2f}s")
        else:
            iou = compute_iou(pred_mask, gt_mask)
            ious.append(iou)

        d_pct = defect_percentage(pred_mask)
        stem = os.path.splitext(img_name)[0]
        out_file = os.path.join(args.out, f"{stem}_comparison.png")

        title = (
            f"{img_name}  |  IoU={iou:.4f}  |  defect={d_pct:.2f}%  |  time={elapsed:.2f}s"
        )

        if gt_mask is None:
            gt_vis = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        else:
            gt_vis = gt_mask

        save_comparison_figure(image_rgb, pred_mask, gt_vis, out_file, title)
        iou_str = f"{iou:.4f}" if gt_mask is not None else "N/A"
        print(f"  [OK] {img_name} | IoU={iou_str} | defect={d_pct:.2f}% | -> {os.path.basename(out_file)}")

    if ious:
        print(f"\nMean IoU ({len(ious)} images with GT): {np.mean(ious):.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
