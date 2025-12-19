import os
from pathlib import Path
import random
from typing import Tuple, Optional, List, Dict, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from enhance import EnhanceUNet


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CleanQRDataset(Dataset):
    """Dataset for clean QR code images with degradation simulation."""
    
    def __init__(self, clean_dir: str, img_size: int = 256):
        """Initialize dataset with clean QR code images.
        
        Args:
            clean_dir: Directory containing clean QR code images
            img_size: Target image size
        """
        self.clean_dir = clean_dir
        self.img_size = img_size
        exts = [".jpg", ".jpeg", ".png", ".bmp"]
        self.files = [f for f in os.listdir(clean_dir)
                      if os.path.isfile(os.path.join(clean_dir, f)) and os.path.splitext(f)[1].lower() in exts]

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.files)

    def _random_degrade(self, img: np.ndarray) -> np.ndarray:
        """Apply random degradations to simulate real-world QR code damage.
        
        Args:
            img: Original clean QR code image
            
        Returns:
            Degraded QR code image
        """
        h, w = img.shape
        # Perspective/affine transformation
        if random.random() < 0.7:
            src = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
            jitter = 0.08
            dst = src + np.float32([
                [random.uniform(-jitter, jitter) * w, random.uniform(-jitter, jitter) * h],
                [random.uniform(-jitter, jitter) * w, random.uniform(-jitter, jitter) * h],
                [random.uniform(-jitter, jitter) * w, random.uniform(-jitter, jitter) * h],
                [random.uniform(-jitter, jitter) * w, random.uniform(-jitter, jitter) * h],
            ])
            M = cv2.getPerspectiveTransform(src, dst)
            img = cv2.warpPerspective(img, M, (w, h), borderValue=255)

        # Blur
        if random.random() < 0.7:
            if random.random() < 0.5:
                k = random.choice([3, 5, 7])
                img = cv2.GaussianBlur(img, (k, k), random.uniform(0.5, 1.5))
            else:
                k = random.choice([5, 7, 9])
                img = cv2.medianBlur(img, k)

        # Noise
        if random.random() < 0.8:
            noise = np.random.normal(0, random.uniform(5, 15), img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Compression artifacts
        if random.random() < 0.8:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(30, 70)]
            _, enc = cv2.imencode('.jpg', img, encode_param)
            img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)

        # Brightness and contrast
        if random.random() < 0.7:
            alpha = random.uniform(0.7, 1.3)  # Contrast control
            beta = random.uniform(-20, 20)     # Brightness control
            img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

        # Small occlusions/stains
        if random.random() < 0.5:
            for _ in range(random.randint(1, 3)):
                x1 = random.randint(0, w-10)
                y1 = random.randint(0, h-10)
                x2 = min(w-1, x1 + random.randint(5, 30))
                y2 = min(h-1, y1 + random.randint(5, 30))
                color = random.choice([0, 255])
                cv2.rectangle(img, (x1, y1), (x2, y2), int(color), -1)
        
        # Special degradation for QR codes: missing black modules
        if random.random() < 0.6:
            # Binarize first to detect black modules
            _, binary = cv2.threshold(img.copy(), 127, 255, cv2.THRESH_BINARY_INV)
            # Find connected components
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Select medium-sized components (simulating QR code modules)
            valid_contours = [cnt for cnt in contours if 10 < cv2.contourArea(cnt) < 200]
            if valid_contours:
                # Randomly select some regions to damage
                num_to_remove = random.randint(1, min(3, len(valid_contours)))
                for i in range(num_to_remove):
                    cnt_idx = random.randint(0, len(valid_contours) - 1)
                    cnt = valid_contours[cnt_idx]
                    # Fill selected black module with white
                    cv2.drawContours(img, [cnt], -1, 255, -1)

        return img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Tuple of (degraded image, target image, black region mask)
        """
        path = os.path.join(self.clean_dir, self.files[idx])
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.full((self.img_size, self.img_size), 255, dtype=np.uint8)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        target = img.copy()
        degraded = self._random_degrade(img)
        
        # Normalize and add channel dimension
        x = torch.from_numpy(degraded).float().unsqueeze(0) / 255.0
        y = torch.from_numpy(target).float().unsqueeze(0) / 255.0
        
        # Create mask for black regions
        # Black parts of QR code are close to 0 after normalization
        black_mask = (y < 0.3).float()
        return x, y, black_mask


def custom_qr_loss(pred: torch.Tensor, target: torch.Tensor, black_mask: torch.Tensor, black_weight: float = 2.0) -> torch.Tensor:
    """Custom loss function that adds higher weight to black regions of QR codes.
    
    Args:
        pred: Predicted output tensor
        target: Target image tensor
        black_mask: Mask indicating black regions
        black_weight: Weight coefficient for black regions
        
    Returns:
        Computed loss value
    """
    # Base L1 loss
    base_loss = torch.abs(pred - target)
    
    # Add higher weight to black regions
    weighted_loss = base_loss * (black_mask * (black_weight - 1.0) + 1.0)
    
    return weighted_loss.mean()


def structure_consistency_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Structure consistency loss to help the model learn regular shapes of QR codes.
    
    Args:
        pred: Predicted output tensor
        target: Target image tensor
        
    Returns:
        Computed structure consistency loss value
    """
    # Calculate gradient maps (using safer method to handle dimension issues)
    # Vertical gradients
    pred_grad_v = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    target_grad_v = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
    # Horizontal gradients
    pred_grad_h = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    target_grad_h = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
    
    # Calculate average gradients
    pred_grad = (pred_grad_v.mean() + pred_grad_h.mean()) / 2.0
    target_grad = (target_grad_v.mean() + target_grad_h.mean()) / 2.0
    
    # Structure consistency loss
    return torch.abs(pred_grad - target_grad)


def train_enhancer(clean_dir: str = "./process/output_monochrome",
                   save_path: str = "./models/enhance_unet.pth",
                   img_size: int = 256,
                   epochs: int = 30,
                   batch_size: int = 16,
                   lr: float = 1e-4,
                   black_weight: float = 3.0,
                   structure_weight: float = 0.5,
                   pretrained_path: Optional[str] = None) -> None:
    """Train the EnhanceUNet model for QR code enhancement.
    
    Args:
        clean_dir: Directory containing clean QR code images
        save_path: Path to save the trained model weights
        img_size: Input image size
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        black_weight: Weight factor for black regions in loss calculation
        structure_weight: Weight factor for structure consistency loss
        pretrained_path: Path to pretrained model weights (optional)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)

    dataset = CleanQRDataset(clean_dir, img_size=img_size)
    if len(dataset) == 0:
        print(f"Dataset is empty: {clean_dir}")
        return
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Create model
    model = EnhanceUNet(in_channels=1).to(device)
    
    # Load pretrained weights if available
    if pretrained_path and os.path.exists(pretrained_path):
        try:
            state = torch.load(pretrained_path, map_location=device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)
            print(f"Loaded pretrained weights from: {pretrained_path}")
        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Learning rate scheduler
    # Removed verbose parameter as it might not be supported in some PyTorch versions
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_loss = 1e9
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_custom_loss = 0.0
        total_struct_loss = 0.0
        count = 0
        
        for x, y, black_mask in loader:
            x = x.to(device)
            y = y.to(device)
            black_mask = black_mask.to(device)
            
            optimizer.zero_grad()
            res = model(x)
            out = (x + res).clamp(0, 1)
            
            # Calculate custom loss
            custom_loss = custom_qr_loss(out, y, black_mask, black_weight)
            # Calculate structure consistency loss
            struct_loss = structure_consistency_loss(out, y) * structure_weight
            # Total loss
            loss = custom_loss + struct_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            total_custom_loss += custom_loss.item() * x.size(0)
            total_struct_loss += struct_loss.item() * x.size(0)
            count += x.size(0)
        
        avg_loss = total_loss / max(1, count)
        avg_custom_loss = total_custom_loss / max(1, count)
        avg_struct_loss = total_struct_loss / max(1, count)
        
        print(f"Epoch {epoch}/{epochs} - Total: {avg_loss:.6f}, Custom: {avg_custom_loss:.6f}, Struct: {avg_struct_loss:.6f}")
        
        # Update learning rate
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved best weights to {save_path}")

    print("Training completed!")
    print(f"Best loss: {best_loss:.6f}")


if __name__ == "__main__":
    set_seed(42)
    
    # Parameters can be adjusted as needed
    train_enhancer(
        clean_dir="./process/output_monochrome",
        save_path="./models/enhance_unet.pth",
        img_size=256,
        epochs=30,  # Increased training epochs for better learning of black region features
        batch_size=16,
        lr=1e-4,
        black_weight=3.0,  # Higher weight for black regions
        structure_weight=0.5,  # Weight for structure consistency loss
        pretrained_path="./enhance_unet.pth"  # Using existing weights as pretraining
    )


