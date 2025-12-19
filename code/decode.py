import os
from pathlib import Path
from typing import Optional
from PIL import Image
from pyzbar.pyzbar import decode, ZBarSymbol


def process_images(
    input_dir: str,
    output_dir: str,
    supported_symbols: Optional[list] = None
) -> float:
    """Process all images in a directory and decode QR codes.
    
    Args:
        input_dir: Path to the input directory containing images
        output_dir: Path to the output directory for results
        supported_symbols: Optional list of barcode/QR code types to recognize
        
    Returns:
        float: Success rate percentage
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Supported image file extensions
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
    
    # Statistics
    total_files = 0
    successfully_decoded = 0
    
    # Walk through input directory
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            # Check if file is an image
            if file.lower().endswith(image_extensions):
                total_files += 1
                file_path = os.path.join(root, file)
                
                try:
                    # Open image
                    with Image.open(file_path) as img:
                        # Convert to grayscale to improve recognition rate
                        img_gray = img.convert('L')
                        
                        # Decode
                        if supported_symbols:
                            results = decode(img_gray, symbols=supported_symbols)
                        else:
                            results = decode(img_gray)
                        
                        # Check if decoding was successful
                        if results and len(results) > 0:
                            successfully_decoded += 1
                
                except Exception as e:
                    # Failed to decode
                    pass
    
    # Calculate and return success rate
    if total_files > 0:
        success_rate = (successfully_decoded / total_files) * 100
        print(f"Total files processed: {total_files}, successfully decoded: {successfully_decoded}")
        print(f"Decoding success rate: {success_rate:.2f}%")
        return success_rate
    else:
        print("No image files found")
        return 0.0

