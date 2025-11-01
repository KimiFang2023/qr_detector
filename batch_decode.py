import os
import sys
import time
import csv
from pathlib import Path
from typing import Optional, List, Dict
from PIL import Image
from pyzbar.pyzbar import decode, ZBarSymbol
from pipeline import run_pipeline



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



def list_image_files(input_dir: str) -> List[str]:
    """List all image file paths in a directory (recursive)."""
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')
    files: List[str] = []
    for root, _dirs, filenames in os.walk(input_dir):
        for fname in filenames:
            if fname.lower().endswith(image_extensions):
                files.append(os.path.join(root, fname))
    return sorted(files)


def decode_directory(
    step_name: str,
    input_dir: str,
    output_base: str,
    supported_symbols: Optional[List[ZBarSymbol]] = None
) -> Dict[str, int]:
    """Decode all images in a directory, save per-image results, and return stats.

    Returns a dict with keys: total, success
    """
    step_output_dir = os.path.join(output_base, step_name)
    create_output_directory(step_output_dir)

    files = list_image_files(input_dir)
    total = len(files)
    success = 0

    for file_path in files:
        decoded_results = decode_image(file_path, supported_symbols)
        if decoded_results:
            success += 1
            output_file = os.path.splitext(os.path.basename(file_path))[0] + '.txt'
            output_path = os.path.join(step_output_dir, output_file)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Image: {file_path}\n")
                f.write(f"Decoding successful, found {len(decoded_results)} codes\n\n")
                for i, result in enumerate(decoded_results, 1):
                    f.write(f"Result {i}:\n{result}\n\n")
    print(f"Step [{step_name}] - total: {total}, success: {success}")
    return {"total": total, "success": success}


def run_pipeline_and_decode(
    original_dir: str,
    output_root: str,
    supported_symbols: Optional[List[ZBarSymbol]] = None
) -> None:
    """Run decode after each pipeline step and write a consolidated report.

    Success rate denominator is the number of images decodable in the original set.
    """
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_base = os.path.join(output_root, f"batch_decode_{timestamp}")
    create_output_directory(output_base)

    # 先跑完整流程，强制生成各步骤目录
    print("Running full pipeline...")
    run_pipeline(
        input_images_dir=original_dir,
        output_boxed_dir="./process/output_boxed",
        output_cropped_dir="./process/output_cropped",
        output_corrected_dir="./process/output_corrected",
        output_monochrome_dir="./process/output_monochrome",
        output_enhanced_dir="./process/output_enhanced",
        output_restored_dir="./process/output_restored",
        model_path="./models/best.pt",
        auto_repair=True,
        decode_threshold=0.5,
    )

    # 定义固定步骤顺序（不再按存在与否筛选）
    steps = [
        ("original", original_dir),
        ("output_boxed", "./process/output_boxed"),
        ("output_cropped", "./process/output_cropped"),
        ("output_corrected", "./process/output_corrected"),
        ("output_monochrome", "./process/output_monochrome"),
        ("output_enhanced", "./process/output_enhanced"),
        ("output_restored", "./process/output_restored"),
    ]

    # First, compute baseline decodable count from original images
    original_stats = decode_directory("original", steps[0][1], output_base, supported_symbols)
    baseline = original_stats["success"]
    if baseline == 0:
        print("No decodable images in original set. Using total original images as denominator.")
        baseline = original_stats["total"]
    if baseline == 0:
        print("Original directory contains no images. Abort.")
        return

    # Decode remaining steps
    all_stats: List[Dict[str, int]] = []
    summary_rows: List[List[str]] = []

    # Record original first
    all_stats.append({"step": "original", **original_stats})
    summary_rows.append([
        "original", str(original_stats["total"]), str(original_stats["success"]),
        f"{(original_stats['success'] / max(baseline, 1)) * 100:.2f}%"
    ])

    for step_name, step_dir in steps[1:]:
        stats = decode_directory(step_name, step_dir, output_base, supported_symbols)
        all_stats.append({"step": step_name, **stats})
        rate = (stats["success"] / baseline) * 100 if baseline > 0 else 0.0
        summary_rows.append([step_name, str(stats["total"]), str(stats["success"]), f"{rate:.2f}%"])

    # Write summary CSV and TXT
    summary_csv = os.path.join(output_base, 'summary.csv')
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["step", "total", "success", "success_rate_vs_original"])
        writer.writerows(summary_rows)

    summary_txt = os.path.join(output_base, 'summary.txt')
    with open(summary_txt, 'w', encoding='utf-8') as f:
        f.write("Decoding Summary (denominator = decodable count in original set)\n")
        f.write(f"Baseline (original decodable): {baseline}\n\n")
        for row in summary_rows:
            f.write(f"step={row[0]}, total={row[1]}, success={row[2]}, rate_vs_original={row[3]}\n")
    print(f"Summary written to: {summary_txt}\nCSV: {summary_csv}")



def main() -> None:
    """Run batch decode across pipeline steps.

    Usage:
      python batch_decode.py <original_image_folder_path> <output_root_folder_path>

    Defaults:
      original_image_folder_path = C:\\Users\\Kimi\\PycharmProjects\\baidu_images
      output_root_folder_path    = ./process/decode_results
    """
    if len(sys.argv) >= 2:
        original_dir = sys.argv[1]
    else:
        # Windows 路径，按用户要求
        original_dir = r"C:\\Users\\Kimi\\PycharmProjects\\baidu_images"

    if len(sys.argv) >= 3:
        output_root = sys.argv[2]
    else:
        output_root = "./process/decode_results"

    if not os.path.isdir(original_dir):
        print(f"Error: Original directory not found: {original_dir}")
        return

    supported_symbols = [ZBarSymbol.QRCODE]
    run_pipeline_and_decode(original_dir, output_root, supported_symbols)



if __name__ == "__main__":
    main()