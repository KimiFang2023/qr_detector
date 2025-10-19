import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import math


def correct_skew(image: np.ndarray) -> np.ndarray:
    """Detect image skew angle and correct it using affine transformation.
    
    Args:
        image: Input image in BGR format
        
    Returns:
        Corrected image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarization
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological operation to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image  # No contours found, return original image

    # Find the largest contour (assumed to be the QR code)
    largest_contour = max(contours, key=cv2.contourArea)

    # Fit minimum area rectangle
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[2]

    # Adjust angle (minAreaRect returns angle in a special range)
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90

    # Calculate rotation center and matrix
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate new image size after rotation to avoid cropping
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust translation part of the rotation matrix
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Perform affine transformation (rotation)
    corrected = cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)  # White background fill
    )

    return corrected


def process_cropped_qrcodes(input_dir: str, output_dir: str) -> None:
    """Batch process cropped QR code images for angle correction.
    
    Args:
        input_dir: Directory containing cropped QR code images
        output_dir: Directory to save corrected images
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(input_dir)
                   if os.path.isfile(os.path.join(input_dir, f))
                   and Path(f).suffix.lower() in image_extensions]

    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    # Process each image
    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        print(f"Processing: {img_path}")

        # Read image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to read image {img_file}, skipped")
            continue

        # Correct skew
        corrected_image = correct_skew(image)

        # Save corrected image
        output_path = os.path.join(output_dir, img_file)
        cv2.imwrite(output_path, corrected_image)
        print(f"Saved corrected image: {output_path}")

    print("All images processed!")


if __name__ == "__main__":
    """Main entry point for the script."""
    # Configure folder paths (corresponding to the QR code cropping folder)
    cropped_input = "./process/output_cropped"  # Input: folder with cropped QR code images
    corrected_output = "./process/output_corrected"  # Output: folder for angle-corrected images

    # Execute processing
    process_cropped_qrcodes(cropped_input, corrected_output)