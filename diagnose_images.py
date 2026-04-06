"""
diagnose_images.py - Per-image breakdown of what each pipeline component contributes.
Shows IoU for: U-Net only, classical CV only, combined, and identifies where the loss comes from.
"""
import os, sys, cv2, numpy as np
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SUBMISSION_DIR = os.path.join(PROJECT_ROOT, 'submission5')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
DEFECTIVE_IMG_DIR = os.path.join(DATA_DIR, 'train', 'defective')
GT_DIR = os.path.join(DATA_DIR, 'ground_truth', 'defective')
sys.path.insert(0, SUBMISSION_DIR)

from model import ToothbrushDefectDetector
import torch

DEVICE = torch.device('cpu')

def compute_iou(pred_mask, gt_mask):
    pred_bin = (pred_mask > 127).astype(np.uint8)
    gt_bin = (gt_mask > 127).astype(np.uint8)
    intersection = np.sum(pred_bin & gt_bin)
    union = np.sum(pred_bin | gt_bin)
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

detector = ToothbrushDefectDetector()

defective_files = sorted([f for f in os.listdir(DEFECTIVE_IMG_DIR) if f.endswith('.png')])[:15]

print(f"{'Image':<10} {'Combined':>10} {'UNet-only':>10} {'CV-only':>10} {'GT-defect%':>10} {'UNet-pred%':>10} {'CV-pred%':>10}")
print("-" * 75)

for img_name in defective_files:
    img_path = os.path.join(DEFECTIVE_IMG_DIR, img_name)
    gt_path = os.path.join(GT_DIR, img_name.replace('.png', '_mask.png'))
    
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(gt_path) else None
    if gt_mask is None:
        continue
    
    original_h, original_w = image_rgb.shape[:2]
    total_pixels = original_h * original_w
    
    # Get components
    body_mask = detector._get_body_mask(image_rgb)
    external_defect_mask = detector._get_external_defects(body_mask)
    internal_dark_mask = detector._get_internal_dark_defects(image_rgb, body_mask)
    
    # Classical CV only
    cv_mask = cv2.bitwise_or(external_defect_mask, internal_dark_mask)
    body_dilated = cv2.dilate(body_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cv_mask = cv2.bitwise_and(cv_mask, body_dilated)
    
    # U-Net only
    unet_mask = np.zeros((original_h, original_w), dtype=np.uint8)
    contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        pad = 30
        x_start, y_start = max(0, x - pad), max(0, y - pad)
        x_end, y_end = min(original_w, x + w + pad), min(original_h, y + h + pad)
        
        cropped_rgb = image_rgb[y_start:y_end, x_start:x_end]
        crop_h, crop_w = cropped_rgb.shape[:2]
        
        if crop_h > 10 and crop_w > 10:
            prob_avg = detector._predict_with_tta(cropped_rgb)
            pred = (prob_avg > detector.threshold).astype(np.float32)
            pred_resized = cv2.resize(pred, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
            pred_binary = (pred_resized > 0).astype(np.uint8) * 255
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            pred_binary = cv2.dilate(pred_binary, dilate_kernel, iterations=1)
            unet_mask[y_start:y_end, x_start:x_end] = pred_binary
    unet_mask = cv2.bitwise_and(unet_mask, body_dilated)
    
    # Combined
    combined = detector.predict(image_rgb)
    
    gt_pct = 100 * np.sum(gt_mask > 127) / total_pixels
    unet_pct = 100 * np.sum(unet_mask > 127) / total_pixels
    cv_pct = 100 * np.sum(cv_mask > 127) / total_pixels
    
    iou_combined = compute_iou(combined, gt_mask)
    iou_unet = compute_iou(unet_mask, gt_mask)
    iou_cv = compute_iou(cv_mask, gt_mask)
    
    print(f"{img_name:<10} {iou_combined:>10.4f} {iou_unet:>10.4f} {iou_cv:>10.4f} "
          f"{gt_pct:>10.2f} {unet_pct:>10.2f} {cv_pct:>10.2f}")
