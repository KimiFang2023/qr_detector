import os
from pathlib import Path
from typing import Optional, Tuple

import qrcode
import cv2
import numpy as np
import random
import string


def random_text(n: int = 32) -> str:
    """Generate random text with specified length.
    
    Args:
        n: Length of the random text
        
    Returns:
        Randomly generated text string
    """
    alphabet = string.ascii_letters + string.digits
    return ''.join(random.choice(alphabet) for _ in range(n))


def make_qr_image(text: str, size: int = 256, border: int = 1) -> np.ndarray:
    """Generate a QR code image with specified content and size.
    
    Args:
        text: Text content to encode in the QR code
        size: Size of the output image in pixels
        border: Border size around the QR code
        
    Returns:
        Grayscale numpy array representing the QR code image
    """
    qr = qrcode.QRCode(version=None, error_correction=qrcode.constants.ERROR_CORRECT_M,
                       box_size=10, border=border)
    qr.add_data(text)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert('L')
    img = np.array(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    return img


def degrade(img: np.ndarray) -> np.ndarray:
    """Apply random degradations to an image to simulate real-world conditions.
    
    Args:
        img: Input grayscale image
        
    Returns:
        Degraded image with various distortions applied
    """
    h, w = img.shape
    out = img.copy()
    
    # Perspective distortion
    if random.random() < 0.7:
        src = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
        jitter = 0.1
        dst = src + np.float32([
            [random.uniform(-jitter, jitter) * w, random.uniform(-jitter, jitter) * h],
            [random.uniform(-jitter, jitter) * w, random.uniform(-jitter, jitter) * h],
            [random.uniform(-jitter, jitter) * w, random.uniform(-jitter, jitter) * h],
            [random.uniform(-jitter, jitter) * w, random.uniform(-jitter, jitter) * h],
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        out = cv2.warpPerspective(out, M, (w, h), borderValue=255)

    # Blur
    if random.random() < 0.8:
        if random.random() < 0.5:
            k = random.choice([3, 5, 7])
            out = cv2.GaussianBlur(out, (k, k), random.uniform(0.5, 1.5))
        else:
            k = random.choice([5, 7, 9])
            out = cv2.medianBlur(out, k)

    # Noise
    if random.random() < 0.8:
        noise = np.random.normal(0, random.uniform(5, 20), out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Compression artifacts
    if random.random() < 0.9:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(25, 80)]
        _, enc = cv2.imencode('.jpg', out, encode_param)
        out = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)

    # Brightness and contrast
    if random.random() < 0.7:
        alpha = random.uniform(0.6, 1.4)
        beta = random.uniform(-30, 30)
        out = np.clip(alpha * out + beta, 0, 255).astype(np.uint8)

    # Stains/occlusions
    if random.random() < 0.6:
        for _ in range(random.randint(1, 3)):
            x1 = random.randint(0, w-10)
            y1 = random.randint(0, h-10)
            x2 = min(w-1, x1 + random.randint(5, 40))
            y2 = min(h-1, y1 + random.randint(5, 40))
            color = random.choice([0, 255])
            cv2.rectangle(out, (x1, y1), (x2, y2), int(color), -1)

    return out


def generate_dataset(root_dir: str = "./qr_dataset_auto", num_samples: int = 5000, 
                    size: int = 256) -> None:
    """Generate a dataset of clean and damaged QR code images.
    
    Args:
        root_dir: Root directory to store the dataset
        num_samples: Number of QR code pairs to generate
        size: Size of each QR code image in pixels
    """
    clean_dir = os.path.join(root_dir, "original")
    damaged_dir = os.path.join(root_dir, "damaged")
    Path(clean_dir).mkdir(parents=True, exist_ok=True)
    Path(damaged_dir).mkdir(parents=True, exist_ok=True)

    for i in range(num_samples):
        content = random_text(24)
        clean = make_qr_image(content, size=size)
        damaged = degrade(clean)
        name = f"qr_{i:06d}.png"
        cv2.imwrite(os.path.join(clean_dir, name), clean)
        cv2.imwrite(os.path.join(damaged_dir, name), damaged)
        if (i + 1) % 500 == 0:
            print(f"Generated {i+1}/{num_samples}")

    print(f"Dataset generated at {root_dir}, total {num_samples} pairs")


if __name__ == "__main__":
    """Main entry point for the script."""
    generate_dataset()


