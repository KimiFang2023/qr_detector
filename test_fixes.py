import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Union

from binarize import binarize_image
from enhance import enhance_image
from repair import QRCodeRepair

def test_single_image(image_path: str) -> None:
    """Test various image processing steps on a single image.
    
    Args:
        image_path: Path to the image file to test
    """
    print(f"Testing image: {image_path}")
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to read image")
        return
    
    # 1. Test binarization
    print("1. Testing binarization...")
    try:
        bw = binarize_image(img, model=None, device=None)  # Use traditional method first
        print(f"   Binarization output range: {bw.min()}-{bw.max()}, mean: {bw.mean():.1f}")
        
        # Save binarization result
        bw_path = image_path.replace('.jpg', '_bw.jpg').replace('.png', '_bw.png')
        cv2.imwrite(bw_path, bw)
        print(f"   Binarization result saved: {bw_path}")
        
    except Exception as e:
        print(f"   Binarization failed: {e}")
    
    # 2. Test enhancement
    print("2. Testing enhancement...")
    try:
        enhanced = enhance_image(bw, model=None, device=None)  # Use CLAHE first
        print(f"   Enhancement output range: {enhanced.min()}-{enhanced.max()}, mean: {enhanced.mean():.1f}")
        
        # Save enhancement result
        enh_path = image_path.replace('.jpg', '_enh.jpg').replace('.png', '_enh.png')
        cv2.imwrite(enh_path, enhanced)
        print(f"   Enhancement result saved: {enh_path}")
        
    except Exception as e:
        print(f"   Enhancement failed: {e}")
    
    # 3. Test repair (if enhancement failed)
    print("3. Testing repair...")
    try:
        repair = QRCodeRepair()
        # Use enhanced image as input if available
        input_for_repair = enhanced if 'enhanced' in locals() else bw
        restored = repair.inpaint(input_for_repair, np.ones_like(input_for_repair, dtype=np.uint8))
        
        if restored is not None:
            print(f"   Repair output range: {restored.min()}-{restored.max()}, mean: {restored.mean():.1f}")
            
            # Save repair result
            rep_path = image_path.replace('.jpg', '_rep.jpg').replace('.png', '_rep.png')
            cv2.imwrite(rep_path, restored)
            print(f"   Repair result saved: {rep_path}")
        else:
            print("   Repair returned None")
            
    except Exception as e:
        print(f"   Repair failed: {e}")
    
    print("Testing completed!")

if __name__ == "__main__":
    """Main entry point for the script."""
    # Test a single image
    test_image_dir = "./process/output_corrected"  # Start testing from corrected images
    
    # Find the first image
    image_files = list(Path(test_image_dir).glob("*.jpg")) + list(Path(test_image_dir).glob("*.png"))
    if image_files:
        test_single_image(str(image_files[0]))
    else:
        print(f"No image files found in {test_image_dir}")
        print("Please run angle.py first to generate corrected images")
