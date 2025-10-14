import os
import sys
from pathlib import Path
from typing import Optional, List, Union, Dict, Any, Set, Tuple
from PIL import Image
from pyzbar.pyzbar import decode, ZBarSymbol



def create_output_directory(output_dir: str) -> None:
    """Create output directory if it doesn't exist.
    
    Args:
        output_dir: Path to the output directory
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")



def decode_image(
    image_path: str,
    supported_symbols: Optional[List[ZBarSymbol]] = None
) -> Optional[List[str]]:
    """Decode barcodes/QR codes from a single image.
    
    Args:
        image_path: Path to the image file
        supported_symbols: Optional list of barcode/QR code types to recognize
        
    Returns:
        List of decoded results as strings, or None if decoding failed
    """
    try:
        # Open image
        with Image.open(image_path) as img:
            # Convert to grayscale to improve recognition rate
            img_gray = img.convert('L')

            # Set barcode types to recognize, default to all supported types
            if supported_symbols:
                results = decode(img_gray, symbols=supported_symbols)
            else:
                results = decode(img_gray)

            # Extract decoded data
            decoded_data: List[str] = []
            for result in results:
                # Decode data
                data = result.data.decode('utf-8')
                # Symbol type
                symbol_type = result.type
                decoded_data.append(f"Type: {symbol_type}, Content: {data}")

            return decoded_data

    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None



def process_images(
    input_dir: str,
    output_dir: str,
    supported_symbols: Optional[List[ZBarSymbol]] = None
) -> float:
    """Process all images in a directory and save only successfully decoded results.
    
    Args:
        input_dir: Path to the input directory containing images
        output_dir: Path to the output directory for results
        supported_symbols: Optional list of barcode/QR code types to recognize
        
    Returns:
        float: Success rate percentage
    """
    # Ensure output directory exists
    create_output_directory(output_dir)

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
                print(f"Processing: {file_path}")

                # Decode image
                decoded_results = decode_image(file_path, supported_symbols)

                # Generate output file path
                output_file = os.path.splitext(file)[0] + ".txt"
                output_path = os.path.join(output_dir, output_file)

                # Only save successfully decoded results
                if decoded_results and len(decoded_results) > 0:
                    successfully_decoded += 1
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(f"Image: {file_path}\n")
                        f.write(f"Decoding successful, found {len(decoded_results)} codes\n\n")
                        for i, result in enumerate(decoded_results, 1):
                            f.write(f"Result {i}:\n{result}\n\n")
                    print(f"Decoding successful, results saved to: {output_path}")
                else:
                    print(f"No codes recognized from {file}, skipping result file generation")

    # Calculate and return success rate
    if total_files > 0:
        success_rate = (successfully_decoded / total_files) * 100
        print(f"\nProcessing completed! Total files processed: {total_files}, successfully decoded: {successfully_decoded}")
        print(f"Decoding success rate: {success_rate:.2f}%")
        return success_rate
    else:
        print("\nNo image files found")
        return 0



def main() -> None:
    """Main function to run the barcode/QR code decoder."""
    # Check command line arguments
    if len(sys.argv) != 3:
        print("Usage: python barcode_decoder.py <input_image_folder_path> <output_result_folder_path>")
        print("Example: python barcode_decoder.py ./images ./results")
        # Use default paths for demonstration
        input_dir = "./process/output_enhanced"
        output_dir = "./process/decode_results/output_enhanced"
        print(f"Using default paths: input={input_dir}, output={output_dir}")
    else:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2]

    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist")
        return

    # Specify which code types to recognize (optional)
    # For example, to recognize only QR codes and CODE128: [ZBarSymbol.QRCODE, ZBarSymbol.CODE128]
    # Set to None to recognize all supported types
    supported_symbols = None

    # Process images
    process_images(input_dir, output_dir, supported_symbols)



if __name__ == "__main__":
    main()