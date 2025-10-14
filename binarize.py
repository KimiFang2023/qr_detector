import os
from pathlib import Path
from typing import Tuple, Optional, List, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block with batch normalization and ReLU activation."""
    
    def __init__(self, in_channels: int, out_channels: int):
        """Initialize the double convolution block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super(DoubleConv, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the double convolution block.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after convolution
        """
        return self.net(x)


class UNet(nn.Module):
    """U-Net architecture for image segmentation and binarization."""
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_c: int = 64):
        """Initialize the U-Net model.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            base_c: Base number of channels
        """
        super(UNet, self).__init__()
        self.inc = DoubleConv(in_channels, base_c)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_c, base_c * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_c * 2, base_c * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_c * 4, base_c * 8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_c * 8, base_c * 8))
        self.up1 = nn.ConvTranspose2d(base_c * 8, base_c * 8, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(base_c * 16, base_c * 4)
        self.up2 = nn.ConvTranspose2d(base_c * 4, base_c * 4, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(base_c * 8, base_c * 2)
        self.up3 = nn.ConvTranspose2d(base_c * 2, base_c * 2, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(base_c * 4, base_c)
        self.up4 = nn.ConvTranspose2d(base_c, base_c, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(base_c * 2, base_c)
        self.outc = nn.Conv2d(base_c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the U-Net.
        
        Args:
            x: Input tensor
            
        Returns:
            Output logits tensor
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        u1 = self.up1(x5)
        u1 = torch.cat([u1, x4], dim=1)
        u1 = self.conv1(u1)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, x3], dim=1)
        u2 = self.conv2(u2)
        u3 = self.up3(u2)
        u3 = torch.cat([u3, x2], dim=1)
        u3 = self.conv3(u3)
        u4 = self.up4(u3)
        u4 = torch.cat([u4, x1], dim=1)
        u4 = self.conv4(u4)
        logits = self.outc(u4)
        return logits


def _load_unet_model(model_paths: Tuple[str, ...]) -> Tuple[nn.Module, torch.device, bool]:
    """Load UNet model weights from the provided paths.
    
    Args:
        model_paths: Tuple of possible model weight paths
        
    Returns:
        Tuple of (model, device, loaded_flag)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=1)
    model = model.to(device)

    loaded = False
    for p in model_paths:
        if p and os.path.exists(p):
            try:
                state = torch.load(p, map_location=device)
                # Handle possible key name differences
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                model.load_state_dict(state, strict=False)
                print(f"Loaded UNet weights from: {p}")
                loaded = True
                break
            except Exception as e:
                print(f"Failed to load weights from ({p}): {e}")

    model.eval()
    return model, device, loaded


def _infer_mask(model: nn.Module, device: torch.device, gray_img: np.ndarray) -> np.ndarray:
    """Infer binary mask using the UNet model.
    
    Args:
        model: UNet model
        device: Device (CPU or GPU)
        gray_img: Grayscale input image
        
    Returns:
        Binary mask
    """
    h, w = gray_img.shape[:2]
    # Keep dimensions as multiples of 32 to reduce boundary artifacts
    new_h = (h + 31) // 32 * 32
    new_w = (w + 31) // 32 * 32
    resized = cv2.resize(gray_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 255.0
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        # Directly output logits without forcing sigmoid
        output = logits.squeeze().cpu().numpy()
    output = cv2.resize(output, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Adaptive threshold processing
    if output.min() < output.max():
        # Normalize to 0-255
        output = ((output - output.min()) / (output.max() - output.min()) * 255).astype(np.uint8)
        # Use Otsu threshold
        _, mask = cv2.threshold(output, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # If output is uniform, use traditional method
        mask = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 5)
    return mask


def binarize_image(image: np.ndarray, model: Optional[nn.Module] = None, 
                   device: Optional[torch.device] = None) -> np.ndarray:
    """Binarize an image using either UNet or traditional methods.
    
    Args:
        image: Input image (BGR or grayscale)
        model: Optional UNet model
        device: Optional device for model inference
        
    Returns:
        Binarized image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    if model is None:
        # Try multiple thresholding methods
        _, bw_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 31, 5)
        # Choose result with better contrast
        if np.std(bw_otsu) > np.std(bw_adaptive):
            bw = bw_otsu
        else:
            bw = bw_adaptive
        return bw
    try:
        mask = _infer_mask(model, device, gray)
        return mask
    except Exception as e:
        print(f"UNet binarization failed, falling back to adaptive threshold: {e}")
        # Use improved fallback method
        _, bw_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 31, 5)
        if np.std(bw_otsu) > np.std(bw_adaptive):
            bw = bw_otsu
        else:
            bw = bw_adaptive
        return bw


def process_binarization(input_dir: str, output_dir: str,
                         weight_paths: Tuple[str, ...] = (
                             "./models/binarize_unet_qr.pth",
                             "./models/unet_carvana_scale1.0_epoch2.pth",
                             "./models/unet_carvana_scale0.5_epoch2.pth",
                         )) -> None:
    """Process multiple images for binarization.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save binarized images
        weight_paths: Tuple of possible model weight paths
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model, device, loaded = _load_unet_model(weight_paths)
    if not loaded:
        print("Failed to load UNet weights, will use traditional adaptive threshold.")
        model = None
        device = None

    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_files = [f for f in os.listdir(input_dir)
                   if os.path.isfile(os.path.join(input_dir, f))
                   and Path(f).suffix.lower() in image_extensions]

    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to read image {img_file}, skipped")
            continue
        bw = binarize_image(image, model, device)
        out_path = os.path.join(output_dir, img_file)
        cv2.imwrite(out_path, bw)
        print(f"Saved binarized image: {out_path}")

    print("Binarization processing completed!")


if __name__ == "__main__":
    """Main entry point for the script."""
    corrected_input = "./process/output_corrected"
    bw_output = "./process/output_monochrome"
    process_binarization(corrected_input, bw_output)