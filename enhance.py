import os
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual block for U-Net architecture."""
    
    def __init__(self, channels: int):
        """Initialize residual block with given number of channels.
        
        Args:
            channels: Number of input/output channels
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the residual block.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor after residual connection and activation
        """
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class EnhanceUNet(nn.Module):
    """Enhancement U-Net for QR code image enhancement."""
    
    def __init__(self, in_channels: int = 1, base_c: int = 48):
        """Initialize EnhanceUNet with specified input channels and base filters.
        
        Args:
            in_channels: Number of input channels
            base_c: Base number of filters
        """
        super(EnhanceUNet, self).__init__()
        c = base_c
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            ResidualBlock(c)
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(c, c*2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c*2),
            nn.ReLU(inplace=True),
            ResidualBlock(c*2)
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(c*2, c*4, 3, padding=1, bias=False),
            nn.BatchNorm2d(c*4),
            nn.ReLU(inplace=True),
            ResidualBlock(c*4)
        )
        self.enc4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(c*4, c*8, 3, padding=1, bias=False),
            nn.BatchNorm2d(c*8),
            nn.ReLU(inplace=True),
            ResidualBlock(c*8)
        )

        self.up1 = nn.ConvTranspose2d(c*8, c*4, 2, 2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(c*8, c*4, 3, padding=1, bias=False),
            nn.BatchNorm2d(c*4),
            nn.ReLU(inplace=True),
            ResidualBlock(c*4)
        )
        self.up2 = nn.ConvTranspose2d(c*4, c*2, 2, 2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(c*4, c*2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c*2),
            nn.ReLU(inplace=True),
            ResidualBlock(c*2)
        )
        self.up3 = nn.ConvTranspose2d(c*2, c, 2, 2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(c*2, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            ResidualBlock(c)
        )
        self.outc = nn.Conv2d(c, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the U-Net.
        
        Args:
            x: Input tensor
            
        Returns:
            Enhanced output tensor
        """
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        u1 = self.up1(e4)
        u1 = torch.cat([u1, e3], dim=1)
        d1 = self.dec1(u1)
        u2 = self.up2(d1)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)
        u3 = self.up3(d2)
        u3 = torch.cat([u3, e1], dim=1)
        d3 = self.dec3(u3)
        res = self.outc(d3)
        return res


def _load_enhancer(weight_path: Optional[str] = None) -> Tuple[nn.Module, torch.device, bool]:
    """Load the enhancer model and prepare for inference.
    
    Args:
        weight_path: Optional path to model weights
        
    Returns:
        Tuple of (model, device, loaded_flag)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhanceUNet(in_channels=1).to(device)
    loaded = False
    
    if weight_path and os.path.exists(weight_path):
        try:
            state = torch.load(weight_path, map_location=device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)
            print(f"Loaded enhancement model weights: {weight_path}")
            loaded = True
        except Exception as e:
            print(f"Failed to load enhancement model: {e}")
    
    model.eval()
    return model, device, loaded


def _apply_clahe_unsharp(gray: np.ndarray) -> np.ndarray:
    """Apply CLAHE and unsharp masking for image enhancement.
    
    Args:
        gray: Grayscale image
        
    Returns:
        Enhanced grayscale image
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g1 = clahe.apply(gray)
    blur = cv2.GaussianBlur(g1, (0, 0), 1.0)
    sharp = cv2.addWeighted(g1, 1.5, blur, -0.5, 0)
    return sharp


def enhance_image(
    image: np.ndarray,
    model: Optional[nn.Module] = None,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """Enhance an image using either deep learning model or classical methods.
    
    Args:
        image: Input image (grayscale or BGR)
        model: Optional enhancement model
        device: Device to run model on
        
    Returns:
        Enhanced grayscale image
    """
    # Ensure image is grayscale
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use classical method if no model provided
    if model is None:
        return _apply_clahe_unsharp(gray)
    
    # Prepare image for model input
    h, w = gray.shape[:2]
    hh = (h + 31) // 32 * 32  # Ensure dimensions are multiples of 32
    ww = (w + 31) // 32 * 32
    resized = cv2.resize(gray, (ww, hh), interpolation=cv2.INTER_AREA)
    
    # Normalize and add batch/channel dimensions
    x = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 255.0
    x = x.to(device)
    
    # Forward pass
    with torch.no_grad():
        res = model(x)
        # Ensure output is within valid range
        out = (x + res).clamp(0, 1)
        out = out.squeeze().cpu().numpy()
        
        # Check for abnormal outputs and fallback if needed
        if out.min() == out.max() or np.isnan(out).any():
            print("Model output abnormal, falling back to CLAHE")
            return _apply_clahe_unsharp(gray)
            
        # Normalize to 0-255 and ensure uint8
        out = np.clip(out * 255, 0, 255).astype(np.uint8)
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
        return out


def process_enhancement(
    input_dir: str,
    output_dir: str,
    weight_path: Optional[str] = None
) -> None:
    """Process all images in a directory with enhancement.
    
    Args:
        input_dir: Input directory containing images
        output_dir: Output directory to save enhanced images
        weight_path: Optional path to model weights
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load model or prepare fallback method
    model, device, loaded = _load_enhancer(weight_path)
    if not loaded:
        print("Failed to load enhancement model, will use CLAHE+unsharp masking fallback.")

    # Get image files
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_files = [f for f in os.listdir(input_dir)
                   if os.path.isfile(os.path.join(input_dir, f))
                   and Path(f).suffix.lower() in image_extensions]

    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    # Process each image
    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Failed to read image {img_file}, skipped")
            continue
            
        enh = enhance_image(img, model if loaded else None, device if loaded else None)
        out_path = os.path.join(output_dir, img_file)
        cv2.imwrite(out_path, enh)
        print(f"Saved enhanced image: {out_path}")

    print("Enhancement processing completed!")


if __name__ == "__main__":
    # 这里可以修改模型路径参数
    # 例如，使用我们之前继续训练的模型：
    # process_enhancement("./process/output_monochrome", "./process/output_enhanced", "./models/enhance_unet_continued.pth")
    # 或者使用原始模型：
    process_enhancement("./process/output_monochrome", "./process/output_enhanced", "./models/enhance_unet.pth")


