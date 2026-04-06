"""
tune_postprocess.py - Grid search for dilation size and morphological operations.
Tests different combinations on all defective images.
"""
import os, sys, cv2, numpy as np, time
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SUBMISSION_DIR = os.path.join(PROJECT_ROOT, 'submission5')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
DEFECTIVE_IMG_DIR = os.path.join(DATA_DIR, 'train', 'defective')
GOOD_IMG_DIR = os.path.join(DATA_DIR, 'train', 'good')
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

# Load all images
defective_files = sorted([f for f in os.listdir(DEFECTIVE_IMG_DIR) if f.endswith('.png')])
good_files = sorted([f for f in os.listdir(GOOD_IMG_DIR) if f.endswith('.png')])

defective_images = []
for fn in defective_files:
    img = cv2.cvtColor(cv2.imread(os.path.join(DEFECTIVE_IMG_DIR, fn)), cv2.COLOR_BGR2RGB)
    gt_path = os.path.join(GT_DIR, fn.replace('.png', '_mask.png'))
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(gt_path) else None
    defective_images.append((fn, img, gt))

good_images = []
for fn in good_files:
    img = cv2.cvtColor(cv2.imread(os.path.join(GOOD_IMG_DIR, fn)), cv2.COLOR_BGR2RGB)
    good_images.append((fn, img))

print(f"Loaded {len(defective_images)} defective, {len(good_images)} good images")

# Pre-compute raw probability maps for all defective images (saves time in sweep)
detector = ToothbrushDefectDetector()

print("Pre-computing probability maps...")
precomputed = []
for fn, img, gt in defective_images:
    original_h, original_w = img.shape[:2]
    body_mask = detector._get_body_mask(img)
    body_dilated = cv2.dilate(body_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    
    prob_map = None
    roi_info = None
    contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        pad = 30
        x_start, y_start = max(0, x - pad), max(0, y - pad)
        x_end, y_end = min(original_w, x + w + pad), min(original_h, y + h + pad)
        
        cropped = img[y_start:y_end, x_start:x_end]
        crop_h, crop_w = cropped.shape[:2]
        
        if crop_h > 10 and crop_w > 10:
            prob_map = detector._predict_with_tta(cropped)
            roi_info = (x_start, y_start, x_end, y_end, crop_w, crop_h)
    
    precomputed.append((fn, gt, body_mask, body_dilated, prob_map, roi_info, original_h, original_w))

# Also pre-compute for good images  
precomputed_good = []
for fn, img in good_images:
    original_h, original_w = img.shape[:2]
    body_mask = detector._get_body_mask(img)
    body_dilated = cv2.dilate(body_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    
    prob_map = None
    roi_info = None
    contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        pad = 30
        x_start, y_start = max(0, x - pad), max(0, y - pad)
        x_end, y_end = min(original_w, x + w + pad), min(original_h, y + h + pad)
        
        cropped = img[y_start:y_end, x_start:x_end]
        crop_h, crop_w = cropped.shape[:2]
        
        if crop_h > 10 and crop_w > 10:
            prob_map = detector._predict_with_tta(cropped)
            roi_info = (x_start, y_start, x_end, y_end, crop_w, crop_h)
    
    precomputed_good.append((fn, body_dilated, prob_map, roi_info, original_h, original_w))

print("Done pre-computing.\n")

# Sweep: threshold x dilation_size x closing_size
thresholds = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.30]
dilation_sizes = [0, 3, 5, 7]  # 0 = no dilation
closing_sizes = [0, 3, 5, 7]   # 0 = no closing

def apply_postprocess(prob_map, roi_info, body_dilated, original_h, original_w, threshold, dilate_sz, close_sz):
    mask = np.zeros((original_h, original_w), dtype=np.uint8)
    if prob_map is None or roi_info is None:
        return mask
    
    x_start, y_start, x_end, y_end, crop_w, crop_h = roi_info
    pred = (prob_map > threshold).astype(np.float32)
    pred_resized = cv2.resize(pred, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
    pred_binary = (pred_resized > 0).astype(np.uint8) * 255
    
    if close_sz > 0:
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_sz, close_sz))
        pred_binary = cv2.morphologyEx(pred_binary, cv2.MORPH_CLOSE, close_kernel)
    
    if dilate_sz > 0:
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_sz, dilate_sz))
        pred_binary = cv2.dilate(pred_binary, dilate_kernel, iterations=1)
    
    mask[y_start:y_end, x_start:x_end] = pred_binary
    mask = cv2.bitwise_and(mask, body_dilated)
    return mask

print(f"{'thresh':>6} {'dilate':>6} {'close':>6} | {'def_IoU':>8} {'good_IoU':>8} {'overall':>8}")
print("-" * 60)

best_def_iou = 0
best_params = None

for thresh in thresholds:
    for dilate_sz in dilation_sizes:
        for close_sz in closing_sizes:
            # Evaluate defective
            def_ious = []
            for fn, gt, body_mask, body_dilated, prob_map, roi_info, oh, ow in precomputed:
                if gt is None:
                    continue
                mask = apply_postprocess(prob_map, roi_info, body_dilated, oh, ow, thresh, dilate_sz, close_sz)
                def_ious.append(compute_iou(mask, gt))
            
            # Evaluate good
            good_ious = []
            for fn, body_dilated, prob_map, roi_info, oh, ow in precomputed_good:
                gt = np.zeros((oh, ow), dtype=np.uint8)
                mask = apply_postprocess(prob_map, roi_info, body_dilated, oh, ow, thresh, dilate_sz, close_sz)
                good_ious.append(compute_iou(mask, gt))
            
            mean_def = np.mean(def_ious)
            mean_good = np.mean(good_ious)
            mean_all = np.mean(def_ious + good_ious)
            
            if mean_good >= 0.98 and mean_def > best_def_iou:
                best_def_iou = mean_def
                best_params = (thresh, dilate_sz, close_sz, mean_def, mean_good, mean_all)
            
            print(f"{thresh:>6.2f} {dilate_sz:>6} {close_sz:>6} | {mean_def:>8.4f} {mean_good:>8.4f} {mean_all:>8.4f}"
                  f"{'  *** BEST' if best_params and best_params[:3] == (thresh, dilate_sz, close_sz) else ''}")

print("\n" + "=" * 60)
if best_params:
    print(f"BEST: thresh={best_params[0]}, dilate={best_params[1]}, close={best_params[2]}")
    print(f"  Defective IoU: {best_params[3]:.4f}")
    print(f"  Good IoU:      {best_params[4]:.4f}")
    print(f"  Overall IoU:   {best_params[5]:.4f}")
print("=" * 60)
