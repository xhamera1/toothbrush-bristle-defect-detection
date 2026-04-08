import os
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# Resolve the directory of THIS script file, so weights.pth is always
# found regardless of the current working directory (critical for CodaBench).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


DEVICE = torch.device('cpu')

# Tunable threshold for U-Net predictions (will be optimized via grid search)
UNET_THRESHOLD = 0.30

# Multi-scale ROI TTA: resize crop by these factors before Resize(256)->model, then warp
# probs back to crop size and average. Tune with tune_multiscale_tta.py on local data.
ROI_TTA_SCALES = (0.92, 1.0)


class ToothbrushDefectDetector:
    def __init__(self, weights_path=None, threshold=None, roi_tta_scales=None):
        """
        Initializes the hybrid detection system (OpenCV + U-Net).
        
        The weights_path is resolved relative to this file's directory
        to ensure compatibility with the CodaBench Docker environment,
        where the working directory may differ from the submission directory.
        """
        if weights_path is None:
            weights_path = os.path.join(SCRIPT_DIR, 'weights.pth')
        
        self.threshold = threshold if threshold is not None else UNET_THRESHOLD
        self.roi_tta_scales = (
            tuple(roi_tta_scales) if roi_tta_scales is not None else ROI_TTA_SCALES
        )

        self.model = smp.Unet(
            encoder_name="resnet34",        
            encoder_weights=None,
            in_channels=3,                  
            classes=1                       
        )
        
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found at {weights_path}")
            
        self.model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
        self.model.to(DEVICE)
        self.model.eval()
        
        self.transform = A.Compose([
            A.Resize(height=256, width=256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    # RGB->HSV, CLAHE, Gausian Blur, Otsu, Canny, OR: maska intensywnosci + krawedzie, mrofologia: close i open
    def _get_body_mask(self, image_rgb):
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        v = hsv[:, :, 2]

        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced_v = clahe.apply(v)
        blurred = cv2.GaussianBlur(enhanced_v, (3, 3), 0)

        otsu_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, intensity_mask = cv2.threshold(blurred, int(otsu_thresh * 0.85), 255, cv2.THRESH_BINARY)

        edges = cv2.Canny(blurred, threshold1=40, threshold2=120)
        edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, edge_kernel, iterations=1)

        fused_mask = cv2.bitwise_or(intensity_mask, edges)

        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        refined_mask = cv2.morphologyEx(fused_mask, cv2.MORPH_CLOSE, close_kernel)
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, open_kernel)

        contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        filled_mask = np.zeros_like(refined_mask)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(filled_mask, [largest_contour], -1, 255, -1)

        return filled_mask

    # otwarcie na masce ciala -> wykrywa odstajace elementy 
    # dylatacja rdzenia
    # deviation = body_mask AND NOT(core_expanded) -> to co jest poza rdzeniem traktujemy jako kandydata defektu zewnetrznego
    # otwarcie
    def _get_external_defects(self, body_mask):
        """Finds splaying bristles using morphological opening."""
        circular_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        core_body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_OPEN, circular_kernel)
        
        margin_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        core_body_expanded = cv2.dilate(core_body_mask, margin_kernel, iterations=1)
        
        deviation_mask = cv2.bitwise_and(body_mask, cv2.bitwise_not(core_body_expanded))
        noise_kernel = np.ones((7, 7), np.uint8)
        
        return cv2.morphologyEx(deviation_mask, cv2.MORPH_OPEN, noise_kernel)

    # RGB->HSV, Otsu na V-kanale, Binary Inverse, Erosion, AND z ciemnymi pikselami, otwarcie -> OTRZYMUJEMY MASKE CIEMNYCH DEFEKTOW WEWNETRZNYCH
    def _get_internal_dark_defects(self, image_rgb, body_mask):
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        v_channel = hsv[:, :, 2]
        
        otsu_thresh, _ = cv2.threshold(v_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        _, dark_pixels = cv2.threshold(v_channel, otsu_thresh * 0.5, 255, cv2.THRESH_BINARY_INV)
        
        erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
        safe_inner_body = cv2.erode(body_mask, erosion_kernel, iterations=1)
        
        internal_defects = cv2.bitwise_and(dark_pixels, safe_inner_body)
        
        noise_kernel = np.ones((5, 5), np.uint8)
        internal_defects = cv2.morphologyEx(internal_defects, cv2.MORPH_OPEN, noise_kernel)
        
        return internal_defects

    # TTA - Test Time Augmentation - pokazujemy kilka wersji jednego obrazka, np odbicie lustrzane, odbicie w poziomie itp, i bierzemy z tego średnią
    def _predict_with_tta(self, cropped_rgb):
        """Dihedral flips + multi-scale ROI TTA; mean fusion and light prob smoothing."""
        crop_h, crop_w = cropped_rgb.shape[:2]
        probs = []

        transforms = [
            (None, lambda x: x),
            (1, lambda x: cv2.flip(x, 1)),
            (0, lambda x: cv2.flip(x, 0)),
            (-1, lambda x: cv2.flip(x, -1)),
        ]

        for flip_code, inverse_fn in transforms:
            if flip_code is None:
                aug_img = cropped_rgb
            else:
                aug_img = cv2.flip(cropped_rgb, flip_code)

            for scale in self.roi_tta_scales:
                sh = max(8, int(round(crop_h * scale)))
                sw = max(8, int(round(crop_w * scale)))
                scaled_rgb = cv2.resize(
                    aug_img, (sw, sh), interpolation=cv2.INTER_LINEAR
                )
                augmented = self.transform(image=scaled_rgb)
                input_tensor = augmented['image'].unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    output = self.model(input_tensor)
                    prob = torch.sigmoid(output).cpu().numpy()[0, 0]

                prob = cv2.resize(
                    prob.astype(np.float32),
                    (sw, sh),
                    interpolation=cv2.INTER_LINEAR,
                )
                prob = cv2.resize(
                    prob, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR
                )
                probs.append(inverse_fn(prob))

        prob_avg = np.mean(probs, axis=0)
        return cv2.GaussianBlur(prob_avg, (3, 3), 0)

    # otwarcie i zamkniecie maski U-Net - usuwa małe artefakty i zle powstale obiekty
    def _postprocess_unet_mask(self, mask):
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return mask

        total_area = mask.shape[0] * mask.shape[1]
        min_area = max(20, int(total_area * 0.00012))
        cleaned = np.zeros_like(mask)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                cleaned[labels == i] = 255
        return cleaned

    def predict(self, image_rgb):
        """
        Main pipeline: Executes classical CV for external defects, 
        and Deep Learning for internal defects on the ROI.
        
        Pipeline:
        1. Classical CV: detect external splaying + internal dark defects
        2. U-Net with TTA: detect defects on cropped ROI
        3. Union all detections, then constrain to body region
        
        Args:
            image_rgb: numpy array of shape (H, W, 3), uint8 RGB image.
            
        Returns:
            Binary mask as numpy array of shape (H, W), uint8 with values 0 or 255.
        """
        original_h, original_w = image_rgb.shape[:2]
        final_mask = np.zeros((original_h, original_w), dtype=np.uint8)
        

        body_mask = self._get_body_mask(image_rgb) # HSV + CLAHE + Otsu + Canny + Morphology + Connected Components
        external_defect_mask = self._get_external_defects(body_mask) # morfologia na masce ciala -> wykrywa odstajace elementy 
        internal_dark_mask = self._get_internal_dark_defects(image_rgb, body_mask)# wykrycie ciemnych defektow wewnetrznych

        final_mask = cv2.bitwise_or(final_mask, external_defect_mask)
        final_mask = cv2.bitwise_or(final_mask, internal_dark_mask)
        
        # --- Step 2: U-Net prediction on ROI ---
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
                # TTA - Test Time Augmentation - pokazujemy kilka wersji jednego obrazka, np odbicie lustrzane, odbicie w poziomie itp, i bierzemy z tego średnią
                prob_avg = self._predict_with_tta(cropped_rgb)

                pred = (prob_avg > self.threshold).astype(np.float32)
                pred_resized = cv2.resize(pred, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
                pred_binary = (pred_resized > 0).astype(np.uint8) * 255
                pred_binary = self._postprocess_unet_mask(pred_binary) # otwarcie i zamkniecie maski U-Net - usuwa małe artefakty i zle powstale obiekty

                roi_mask = np.zeros((original_h, original_w), dtype=np.uint8)
                roi_mask[y_start:y_end, x_start:x_end] = pred_binary
                
                # UNION: combine U-Net predictions with classical CV detections
                final_mask = cv2.bitwise_or(final_mask, roi_mask)
        
        # --- Step 3: Constrain to body region (remove FPs outside toothbrush) ---
        body_dilated = cv2.dilate(body_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        final_mask = cv2.bitwise_and(final_mask, body_dilated)
                
        return final_mask


# Module-level initialization: create the detector once when model.py is imported.
# This is the pattern expected by CodaBench's scoring script.
detector = ToothbrushDefectDetector()


def predict(image):
    """
    Entry point called by the CodaBench scoring script.
    
    Args:
        image: numpy array of shape (H, W, 3), uint8 RGB image.
        
    Returns:
        Binary mask as numpy array of shape (H, W), uint8 with values 0 or 255.
    """
    return detector.predict(image)