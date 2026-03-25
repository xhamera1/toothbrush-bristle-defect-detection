import os
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

class ToothbrushDefectDetector:
    def __init__(self, weights_path='weights.pth', device=None):
        """
        Initializes the hybrid detection system (OpenCV + U-Net).
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = smp.Unet(
            encoder_name="resnet34",        
            encoder_weights=None,
            in_channels=3,                  
            classes=1                       
        )
        
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found at {weights_path}")
            
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = A.Compose([
            A.Resize(height=256, width=256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def _get_body_mask(self, image_rgb):
        """Extracts the base binary mask of the toothbrush using CLAHE and Otsu."""
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_v = clahe.apply(v)
        blurred = cv2.GaussianBlur(enhanced_v, (7, 7), 0)
        
        otsu_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, final_thresh = cv2.threshold(blurred, otsu_thresh * 0.7, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((21, 21), np.uint8)
        closed_mask = cv2.morphologyEx(final_thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = np.zeros_like(closed_mask)
        for cnt in contours:
            cv2.drawContours(filled_mask, [cnt], -1, 255, -1)
            
        return filled_mask

    def _get_external_defects(self, body_mask):
        """Finds splaying bristles using morphological opening."""
        circular_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
        core_body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_OPEN, circular_kernel)
        
        margin_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        core_body_expanded = cv2.dilate(core_body_mask, margin_kernel, iterations=1)
        
        deviation_mask = cv2.bitwise_and(body_mask, cv2.bitwise_not(core_body_expanded))
        noise_kernel = np.ones((5, 5), np.uint8)
        
        return cv2.morphologyEx(deviation_mask, cv2.MORPH_OPEN, noise_kernel)

    def predict(self, image_rgb):
        """
        Main pipeline: Executes classical CV for external defects, 
        and Deep Learning for internal defects on the ROI.
        """
        original_h, original_w = image_rgb.shape[:2]
        final_mask = np.zeros((original_h, original_w), dtype=np.uint8)
        
        body_mask = self._get_body_mask(image_rgb)
        external_defect_mask = self._get_external_defects(body_mask)
        
        final_mask = cv2.bitwise_or(final_mask, external_defect_mask)
        
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
                augmented = self.transform(image=cropped_rgb)
                input_tensor = augmented['image'].unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    output = self.model(input_tensor)
                    prob = torch.sigmoid(output)
                    pred = (prob > 0.5).float().cpu().numpy()[0, 0]
                
                pred_resized = cv2.resize(pred, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
                pred_binary = (pred_resized > 0).astype(np.uint8) * 255
                
                roi_mask = np.zeros((original_h, original_w), dtype=np.uint8)
                roi_mask[y_start:y_end, x_start:x_end] = pred_binary
                
                final_mask = cv2.bitwise_or(final_mask, roi_mask)
                
        return final_mask

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

detector = ToothbrushDefectDetector(weights_path='weights.pth', device=device)

def predict(image):
    return detector.predict(image)