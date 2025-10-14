import os
from pathlib import Path
from typing import Optional

# Local module imports
from yolo import process_images as detect_and_crop
from angle import process_cropped_qrcodes as correct_angle
from binarize import process_binarization
from decode import process_images as decode_dir
from enhance import process_enhancement
from repair_wrapper import repair_if_needed


def run_pipeline(
    input_images_dir: str = "./process/input_images",
    output_boxed_dir: str = "./process/output_boxed",
    output_cropped_dir: str = "./process/output_cropped",
    output_corrected_dir: str = "./process/output_corrected",
    output_monochrome_dir: str = "./process/output_monochrome",
    output_enhanced_dir: str = "./process/output_enhanced",
    output_restored_dir: str = "./process/output_restored",
    model_path: str = "./models/best.pt",
    auto_repair: bool = True,
    decode_threshold: float = 0.5,
) -> float:
    """Run the complete QR code processing pipeline.
    
    Args:
        input_images_dir: Path to input images directory
        output_boxed_dir: Path to save images with detected bounding boxes
        output_cropped_dir: Path to save cropped QR code regions
        output_corrected_dir: Path to save geometrically corrected QR codes
        output_monochrome_dir: Path to save binarized QR codes
        output_enhanced_dir: Path to save enhanced QR codes
        output_restored_dir: Path to save restored QR codes
        model_path: Path to YOLO detection model weights
        auto_repair: Whether to automatically repair undecodable QR codes
        decode_threshold: Threshold for automatic repair decision
        
    Returns:
        float: Final success rate after processing
    """
    # Create necessary directories
    for dir_path in [
        output_boxed_dir, output_cropped_dir, output_corrected_dir,
        output_monochrome_dir, output_enhanced_dir, output_restored_dir
    ]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    # 1) Detection and cropping
    detect_and_crop(input_images_dir, output_boxed_dir, output_cropped_dir, model_path)

    # 2) Angle/geometric correction
    correct_angle(output_cropped_dir, output_corrected_dir)

    # 3) Binarization (U-Net first, fallback to traditional thresholding)
    process_binarization(output_corrected_dir, output_monochrome_dir)

    # 4) Image enhancement (Residual U-Net/CLAHE fallback). 
    # 这里可以选择使用原始模型还是继续训练的模型
    # 使用原始模型:
    # enhance_weights = "./models/enhance_unet.pth"
    # 使用继续训练的模型:
    enhance_weights = "./models/enhance_unet_continued.pth"
    # 如果指定的模型不存在，则会使用传统方法(CLAHE+unsharp masking)作为后备
    weight_path = enhance_weights if os.path.exists(enhance_weights) else "./models/enhance_unet.pth" if os.path.exists("./models/enhance_unet.pth") else None
    process_enhancement(output_monochrome_dir, output_enhanced_dir, weight_path)

    # 5) Evaluate decoding rate and trigger repair if needed (using enhanced results as input)
    if auto_repair:
        success_rate = repair_if_needed(output_enhanced_dir, output_restored_dir, decode_threshold)
        print(f"Repair stage completed, final decoding success rate: {success_rate:.2f}%")
    else:
        # Only evaluate decoding rate after enhancement
        success_rate = decode_dir(output_enhanced_dir, "./process/decode_results/output_enhanced")
        print(f"Decoding success rate after enhancement: {success_rate:.2f}%")
        
    return success_rate


def main() -> None:
    """Main function to run the pipeline."""
    run_pipeline()


if __name__ == "__main__":
    main()