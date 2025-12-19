import os
from pathlib import Path
from typing import Optional

import cv2
from decode import process_images as decode_dir
from repair import QRCodeRepair


def repair_if_needed(input_dir: str, output_dir: str, decode_threshold: float = 0.5) -> float:
    """Repair QR codes in images if the decoding success rate is below the threshold.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save processed images
        decode_threshold: Threshold for decoding success rate to trigger repair
        
    Returns:
        Final decoding success rate after potential repair
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # First try to decode binary results directly
    print("Evaluating current decode success rate...")
    success_rate = decode_dir(input_dir, os.path.join("./process/decode_results", Path(output_dir).name))

    if success_rate >= decode_threshold:
        print(f"Current decode success rate {success_rate:.2f}% has reached threshold {decode_threshold*100:.0f}%, skipping repair.")
        # Directly copy files to output (maintain pipeline consistency)
        for f in os.listdir(input_dir):
            src = os.path.join(input_dir, f)
            dst = os.path.join(output_dir, f)
            if os.path.isfile(src):
                img = cv2.imread(src)
                if img is not None:
                    cv2.imwrite(dst, img)
        return success_rate

    print("Decode rate does not meet standards, activating repair module...")
    qr = QRCodeRepair()

    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_files = [f for f in os.listdir(input_dir)
                   if os.path.isfile(os.path.join(input_dir, f))
                   and Path(f).suffix.lower() in image_extensions]

    for img_file in image_files:
        in_path = os.path.join(input_dir, img_file)
        out_path = os.path.join(output_dir, img_file)
        qr.auto_detect_and_repair(in_path, out_path)

    # Evaluate again after repair
    print("Repair completed, re-evaluating decode success rate...")
    success_rate = decode_dir(output_dir, os.path.join("./process/decode_results", Path(output_dir).name + "_repaired"))
    return success_rate


if __name__ == "__main__":
    """Main entry point for the script."""
    input_dir = "./process/output_monochrome"
    output_dir = "./process/output_restored"
    sr = repair_if_needed(input_dir, output_dir)
    print(f"Final decode success rate: {sr:.2f}%")


