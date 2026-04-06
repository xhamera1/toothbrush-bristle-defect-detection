"""
tune_threshold.py - Grid search for optimal U-Net threshold.

Sweeps threshold values from 0.05 to 0.50 and evaluates Mean IoU
on defective images. Also tests good images for false positive check.
"""

import os
import sys
import cv2
import numpy as np
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SUBMISSION_DIR = os.path.join(PROJECT_ROOT, 'submission')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

DEFECTIVE_IMG_DIR = os.path.join(DATA_DIR, 'train', 'defective')
GOOD_IMG_DIR = os.path.join(DATA_DIR, 'train', 'good')
GT_DIR = os.path.join(DATA_DIR, 'ground_truth', 'defective')

sys.path.insert(0, SUBMISSION_DIR)


def compute_iou(pred_mask, gt_mask):
    pred_bin = (pred_mask > 127).astype(np.uint8)
    gt_bin = (gt_mask > 127).astype(np.uint8)
    intersection = np.sum(pred_bin & gt_bin)
    union = np.sum(pred_bin | gt_bin)
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def load_images():
    """Load all defective and good images + ground truth."""
    defective_files = sorted([f for f in os.listdir(DEFECTIVE_IMG_DIR) if f.endswith('.png')])
    good_files = sorted([f for f in os.listdir(GOOD_IMG_DIR) if f.endswith('.png')])
    
    defective_images = []
    gt_masks = []
    for img_name in defective_files:
        img_path = os.path.join(DEFECTIVE_IMG_DIR, img_name)
        gt_path = os.path.join(GT_DIR, img_name.replace('.png', '_mask.png'))
        
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(gt_path) else None
        
        defective_images.append((img_name, image_rgb, gt_mask))
    
    good_images = []
    for img_name in good_files:
        img_path = os.path.join(GOOD_IMG_DIR, img_name)
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        good_images.append((img_name, image_rgb))
    
    return defective_images, good_images


def test_threshold(detector, defective_images, good_images):
    """Test a detector with a specific threshold on all images."""
    defective_ious = []
    for img_name, image_rgb, gt_mask in defective_images:
        result = detector.predict(image_rgb)
        if gt_mask is not None:
            iou = compute_iou(result, gt_mask)
            defective_ious.append(iou)
    
    good_ious = []
    for img_name, image_rgb in good_images:
        result = detector.predict(image_rgb)
        gt_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        iou = compute_iou(result, gt_mask)
        good_ious.append(iou)
    
    return defective_ious, good_ious


def main():
    from model import ToothbrushDefectDetector
    
    print("Loading images...")
    defective_images, good_images = load_images()
    print(f"  {len(defective_images)} defective, {len(good_images)} good images loaded.\n")
    
    # Thresholds to test
    thresholds = [0.05, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 
                  0.22, 0.25, 0.28, 0.30, 0.35, 0.40, 0.45, 0.50]
    
    results = []
    
    for thresh in thresholds:
        t0 = time.time()
        detector = ToothbrushDefectDetector(threshold=thresh)
        defective_ious, good_ious = test_threshold(detector, defective_images, good_images)
        elapsed = time.time() - t0
        
        mean_def = np.mean(defective_ious)
        mean_good = np.mean(good_ious)
        mean_all = np.mean(defective_ious + good_ious)
        
        results.append((thresh, mean_def, mean_good, mean_all))
        
        print(f"  thresh={thresh:.2f} | def_IoU={mean_def:.4f} | good_IoU={mean_good:.4f} | "
              f"overall={mean_all:.4f} | time={elapsed:.1f}s")
    
    print("\n" + "=" * 70)
    
    # Find best threshold by defective IoU (with good IoU > 0.95 constraint)
    valid_results = [(t, d, g, a) for t, d, g, a in results if g > 0.95]
    if valid_results:
        best = max(valid_results, key=lambda x: x[1])
        print(f"  BEST threshold: {best[0]:.2f}")
        print(f"    Defective IoU: {best[1]:.4f}")
        print(f"    Good IoU:      {best[2]:.4f}")
        print(f"    Overall IoU:   {best[3]:.4f}")
    else:
        best = max(results, key=lambda x: x[1])
        print(f"  BEST threshold (unconstrained): {best[0]:.2f}")
        print(f"    Defective IoU: {best[1]:.4f}")
        print(f"    Good IoU:      {best[2]:.4f}")
    
    # Also find best by overall IoU
    best_overall = max(valid_results if valid_results else results, key=lambda x: x[3])
    print(f"\n  BEST by overall IoU: {best_overall[0]:.2f}")
    print(f"    Defective IoU: {best_overall[1]:.4f}")
    print(f"    Good IoU:      {best_overall[2]:.4f}")
    print(f"    Overall IoU:   {best_overall[3]:.4f}")
    
    print("=" * 70)


if __name__ == '__main__':
    main()
