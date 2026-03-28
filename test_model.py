"""
test_model.py - Local test script for the ToothbrushDefectDetector.

Tests the predict() function from submission/model.py against real images
from the data/ folder. Validates output format, computes IoU against ground
truth masks, and prints a clear pass/fail summary.

Usage:
    python test_model.py
"""

import os
import sys
import cv2
import numpy as np
import time

# -- Setup paths ---------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SUBMISSION_DIR = os.path.join(PROJECT_ROOT, 'submission')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

DEFECTIVE_IMG_DIR = os.path.join(DATA_DIR, 'train', 'defective')
GOOD_IMG_DIR = os.path.join(DATA_DIR, 'train', 'good')
GT_DIR = os.path.join(DATA_DIR, 'ground_truth', 'defective')

# Add submission directory to path so we can import model.py
sys.path.insert(0, SUBMISSION_DIR)


def compute_iou(pred_mask, gt_mask):
    """Compute Intersection over Union between two binary masks."""
    pred_bin = (pred_mask > 127).astype(np.uint8)
    gt_bin = (gt_mask > 127).astype(np.uint8)

    intersection = np.sum(pred_bin & gt_bin)
    union = np.sum(pred_bin | gt_bin)

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def validate_output(result, original_shape):
    """Validate that the output conforms to competition requirements."""
    errors = []

    if not isinstance(result, np.ndarray):
        errors.append(f"Output is {type(result).__name__}, expected numpy.ndarray")
        return errors

    expected_shape = original_shape[:2]
    if result.shape != expected_shape:
        errors.append(f"Shape mismatch: got {result.shape}, expected {expected_shape}")

    if result.dtype != np.uint8:
        errors.append(f"dtype is {result.dtype}, expected uint8")

    unique_vals = np.unique(result)
    invalid_vals = [v for v in unique_vals if v not in (0, 255)]
    if invalid_vals:
        errors.append(f"Invalid pixel values found: {invalid_vals} (only 0 and 255 allowed)")

    return errors


def run_tests():
    """Run all tests and print results."""
    print("=" * 70)
    print("  TOOTHBRUSH DEFECT DETECTOR -- LOCAL TEST SUITE")
    print("=" * 70)

    # -- Step 1: Import model --------------------------------------------------
    print("\n[1/4] Importing model.py...")
    t0 = time.time()
    try:
        from model import predict
        t_import = time.time() - t0
        print(f"  [OK] Model imported successfully ({t_import:.2f}s)")
    except Exception as e:
        print(f"  [FAIL] IMPORT FAILED: {e}")
        print("\n  This is the most likely cause of a 0.00 score on CodaBench.")
        return

    total_tests = 0
    passed_tests = 0
    ious = []

    # -- Step 2: Test defective images (with ground truth) ---------------------
    print("\n[2/4] Testing DEFECTIVE images (with IoU evaluation)...")
    print("-" * 70)

    defective_files = sorted([f for f in os.listdir(DEFECTIVE_IMG_DIR) if f.endswith('.png')])
    test_defective = defective_files[:15]  # test first 15

    for img_name in test_defective:
        total_tests += 1
        img_path = os.path.join(DEFECTIVE_IMG_DIR, img_name)
        gt_path = os.path.join(GT_DIR, img_name.replace('.png', '_mask.png'))

        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(gt_path) else None

        try:
            t0 = time.time()
            result = predict(image_rgb)
            t_pred = time.time() - t0

            errors = validate_output(result, image_rgb.shape)

            iou_str = "N/A"
            if gt_mask is not None and len(errors) == 0:
                iou = compute_iou(result, gt_mask)
                ious.append(iou)
                iou_str = f"{iou:.4f}"

            defect_pixels = np.sum(result == 255)
            total_pixels = result.shape[0] * result.shape[1]
            defect_pct = 100 * defect_pixels / total_pixels

            if errors:
                print(f"  [FAIL] {img_name} | ERRORS: {'; '.join(errors)}")
            else:
                passed_tests += 1
                print(f"  [OK]   {img_name} | IoU={iou_str} | "
                      f"defect={defect_pct:.2f}% | time={t_pred:.2f}s")

        except Exception as e:
            print(f"  [FAIL] {img_name} | EXCEPTION: {e}")

    # -- Step 3: Test good images (expect empty/near-empty mask) ---------------
    print(f"\n[3/4] Testing GOOD images (expecting minimal detections)...")
    print("-" * 70)

    good_files = sorted([f for f in os.listdir(GOOD_IMG_DIR) if f.endswith('.png')])
    test_good = good_files[:15]  # test first 15

    for img_name in test_good:
        total_tests += 1
        img_path = os.path.join(GOOD_IMG_DIR, img_name)

        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # For good images, ground truth is all zeros
        gt_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)

        try:
            t0 = time.time()
            result = predict(image_rgb)
            t_pred = time.time() - t0

            errors = validate_output(result, image_rgb.shape)

            iou = compute_iou(result, gt_mask)
            ious.append(iou)

            defect_pixels = np.sum(result == 255)
            total_pixels = result.shape[0] * result.shape[1]
            defect_pct = 100 * defect_pixels / total_pixels

            fp_ok = defect_pct <= 1.0

            if errors:
                print(f"  [FAIL] {img_name} | ERRORS: {'; '.join(errors)}")
            elif not fp_ok:
                passed_tests += 1
                print(f"  [WARN] {img_name} | IoU={iou:.4f} | "
                      f"false_positive={defect_pct:.2f}% (HIGH) | time={t_pred:.2f}s")
            else:
                passed_tests += 1
                print(f"  [OK]   {img_name} | IoU={iou:.4f} | "
                      f"false_positive={defect_pct:.2f}% | time={t_pred:.2f}s")

        except Exception as e:
            print(f"  [FAIL] {img_name} | EXCEPTION: {e}")

    # -- Step 4: Summary -------------------------------------------------------
    print(f"\n[4/4] SUMMARY")
    print("=" * 70)
    print(f"  Tests passed:   {passed_tests}/{total_tests}")

    if ious:
        mean_iou = np.mean(ious)
        print(f"  Mean IoU:       {mean_iou:.4f}")

        defective_ious = ious[:len(test_defective)]
        good_ious = ious[len(test_defective):]

        if defective_ious:
            print(f"  Mean IoU (defective): {np.mean(defective_ious):.4f}")
        if good_ious:
            print(f"  Mean IoU (good):      {np.mean(good_ious):.4f}")

    print()
    if passed_tests == total_tests:
        print("  ALL TESTS PASSED -- output format is correct for CodaBench!")
    else:
        print("  WARNING: Some tests failed -- fix issues before submitting.")

    print("=" * 70)


if __name__ == '__main__':
    run_tests()
