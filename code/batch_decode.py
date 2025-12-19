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


def check_pipeline_directories_exist(pipeline_dirs: List[str], original_count: int) -> bool:
    """检查pipeline处理后的目录是否存在且包含足够的文件
    
    Args:
        pipeline_dirs: 需要检查的目录列表
        original_count: 原始图像的数量，用于比较
    
    Returns:
        bool: 如果所有目录都存在且包含足够的文件，返回True
    """
    min_file_count = original_count * 0.8  # 至少需要原始图像数量的80%
    
    for dir_path in pipeline_dirs:
        if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
            print(f"目录不存在或不是目录: {dir_path}")
            return False
        
        files = list_image_files(dir_path)
        if len(files) < min_file_count:
            print(f"目录 {dir_path} 中的文件数量不足: {len(files)} < {min_file_count}")
            return False
    
    print("所有pipeline处理目录已存在且包含足够的文件，跳过pipeline运行")
    return True


def run_pipeline_and_decode(
    original_dir: str,
    output_root: str,
    supported_symbols: Optional[List[ZBarSymbol]] = None,
    force_run_pipeline: bool = False
) -> None:
    """Run decode after each pipeline step and write a consolidated report with improved statistics.
    
    Args:
        original_dir: 原始图像目录
        output_root: 输出根目录
        supported_symbols: 支持的条码符号类型
        force_run_pipeline: 是否强制运行pipeline，即使输出目录已存在
    """
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_base = os.path.join(output_root, f"batch_decode_{timestamp}")
    create_output_directory(output_base)
    
    # 定义固定步骤顺序和目录
    steps = [
        ("original", original_dir),
        ("output_boxed", "./process/output_boxed"),
        ("output_cropped", "./process/output_cropped"),
        ("output_corrected", "./process/output_corrected"),
        ("output_monochrome", "./process/output_monochrome"),
        ("output_enhanced", "./process/output_enhanced"),
        ("output_restored", "./process/output_restored"),
    ]
    
    # 获取原始图像数量
    original_files = list_image_files(original_dir)
    original_count = len(original_files)
    
    # 检查是否需要运行pipeline
    pipeline_dirs = [step[1] for step in steps[1:]]
    skip_pipeline = not force_run_pipeline and check_pipeline_directories_exist(pipeline_dirs, original_count)
    
    if not skip_pipeline:
        # 运行完整流程
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
    baseline_success = original_stats["success"]
    baseline_total = original_stats["total"]
    
    if baseline_total == 0:
        print("Original directory contains no images. Abort.")
        return
        
    original_success_rate = (baseline_success / baseline_total) * 100 if baseline_total > 0 else 0.0
    print(f"原始图像基线统计 - 总数: {baseline_total}, 成功解码: {baseline_success}, 成功率: {original_success_rate:.2f}%")

    # Decode remaining steps
    all_stats: List[Dict] = []
    summary_rows: List[List[str]] = []

    # Record original first
    all_stats.append({
        "step": "original", 
        "total": baseline_total,
        "success": baseline_success,
        "step_success_rate": original_success_rate,
        "improvement_vs_original": 0.0,
        "additional_success": 0
    })
    
    summary_rows.append([
        "original", 
        str(baseline_total), 
        str(baseline_success),
        f"{original_success_rate:.2f}%",  # 该步骤自身成功率
        "0.00%",                          # 相对于原始图像的成功率
        "0.00%",                          # 改进幅度
        "0"                               # 额外成功解码的数量
    ])

    # 存储成功解码的图像文件名（用于计算新增成功）
    original_success_files = set()
    step_dir = steps[0][1]
    for file_path in list_image_files(step_dir):
        decoded_results = decode_image(file_path, supported_symbols)
        if decoded_results:
            file_name = os.path.basename(file_path)
            original_success_files.add(file_name)

    # 处理其他步骤
    for step_name, step_dir in steps[1:]:
        stats = decode_directory(step_name, step_dir, output_base, supported_symbols)
        
        # 计算更合理的统计指标
        step_total = stats["total"]
        step_success = stats["success"]
        step_success_rate = (step_success / step_total) * 100 if step_total > 0 else 0.0
        
        # 相对于原始图像的成功率（基于原始图像总数）
        rate_vs_original = (step_success / baseline_total) * 100 if baseline_total > 0 else 0.0
        
        # 计算改进幅度
        improvement = rate_vs_original - original_success_rate
        
        # 计算额外成功解码的图像数量
        current_success_files = set()
        for file_path in list_image_files(step_dir):
            decoded_results = decode_image(file_path, supported_symbols)
            if decoded_results:
                file_name = os.path.basename(file_path)
                current_success_files.add(file_name)
        
        # 计算新增成功的图像数量（近似值，基于文件名匹配）
        # 这里我们只能做近似计算，因为经过pipeline处理后的文件名可能已更改
        # 实际应用中可能需要更精确的图像对应关系
        additional_success = step_success - baseline_success
        additional_success = max(0, additional_success)  # 确保不为负数
        
        all_stats.append({
            "step": step_name,
            "total": step_total,
            "success": step_success,
            "step_success_rate": step_success_rate,
            "improvement_vs_original": improvement,
            "additional_success": additional_success
        })
        
        summary_rows.append([
            step_name, 
            str(step_total), 
            str(step_success),
            f"{step_success_rate:.2f}%",           # 该步骤自身成功率
            f"{rate_vs_original:.2f}%",           # 相对于原始图像总数的成功率
            f"{improvement:+.2f}%",               # 改进幅度（带符号）
            str(additional_success)                # 额外成功解码的数量
        ])
        
        print(f"步骤 [{step_name}] - 统计:")
        print(f"  该步骤成功率: {step_success_rate:.2f}%")
        print(f"  相对于原始图像成功率: {rate_vs_original:.2f}%")
        print(f"  相对于原始图像的改进: {improvement:+.2f}%")
        print(f"  额外成功解码的图像数: {additional_success}")

    # Write summary CSV and TXT with improved statistics
    summary_csv = os.path.join(output_base, 'summary.csv')
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "step", 
            "total", 
            "success", 
            "step_success_rate",  # 该步骤自身的成功率
            "success_rate_vs_original_total",  # 相对于原始图像总数的成功率
            "improvement_vs_original",  # 相对于原始图像的改进幅度
            "additional_success_count"  # 额外成功解码的图像数量
        ])
        writer.writerows(summary_rows)

    summary_txt = os.path.join(output_base, 'summary.txt')
    with open(summary_txt, 'w', encoding='utf-8') as f:
        f.write("二维码批量解码统计报告\n")
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"原始图像基线 - 总数: {baseline_total}, 成功解码: {baseline_success}, 成功率: {original_success_rate:.2f}%\n")
        f.write(f"是否跳过Pipeline: {'是' if skip_pipeline else '否'}\n\n")
        
        f.write("详细统计数据:\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'步骤名称':<15} {'总数':<8} {'成功数':<8} {'自身成功率':<12} {'相对原始率':<15} {'改进幅度':<12} {'额外成功数':<12}\n")
        f.write("-" * 100 + "\n")
        
        for row in summary_rows:
            f.write(f"{row[0]:<15} {row[1]:<8} {row[2]:<8} {row[3]:<12} {row[4]:<15} {row[5]:<12} {row[6]:<12}\n")
        
        f.write("-" * 100 + "\n\n")
        
        # 添加分析总结
        best_step = max(all_stats[1:], key=lambda x: x['improvement_vs_original'])
        f.write(f"最佳改进步骤: {best_step['step']}\n")
        f.write(f"该步骤相对于原始图像的改进幅度: {best_step['improvement_vs_original']:+.2f}%\n")
        f.write(f"该步骤额外成功解码的图像数: {best_step['additional_success']}\n\n")
        
        # 计算总体改进
        total_improvement = 0
        for step_stats in all_stats[1:]:
            total_improvement += step_stats['improvement_vs_original']
        avg_improvement = total_improvement / len(all_stats[1:]) if all_stats[1:] else 0
        
        f.write(f"所有处理步骤的平均改进幅度: {avg_improvement:+.2f}%\n")
        f.write("注意: 改进幅度为正值表示相对于原始图像有提升，负值表示性能下降。\n")
    
    print(f"\n统计报告已生成:")
    print(f"文本报告: {summary_txt}")
    print(f"CSV报告: {summary_csv}")
    print(f"最佳改进步骤: {best_step['step']}")
    print(f"最大改进幅度: {best_step['improvement_vs_original']:+.2f}%")



def main() -> None:
    """Run batch decode across pipeline steps with improved statistics and directory checks.

    Usage:
      python batch_decode.py <original_image_folder_path> <output_root_folder_path> [--force]

    Arguments:
      original_image_folder_path: 原始图像文件夹路径
      output_root_folder_path: 输出结果的根文件夹路径
      --force: 强制运行pipeline，即使输出目录已存在（可选）

    Defaults:
      original_image_folder_path = C:/Users/Kimi/PycharmProjects/baidu_images
      output_root_folder_path    = ./process/decode_results
    """
    # 解析命令行参数
    force_run_pipeline = False
    if len(sys.argv) >= 4 and sys.argv[3] == "--force":
        force_run_pipeline = True
        print("强制运行模式: 将重新生成所有处理结果")
    
    # 设置原始目录
    if len(sys.argv) >= 2:
        original_dir = sys.argv[1]
    else:
        # 使用项目根目录下的baidu_images文件夹
        original_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baidu_images")

    # 设置输出根目录
    if len(sys.argv) >= 3:
        output_root = sys.argv[2]
    else:
        output_root = "./process/decode_results"

    # 验证原始目录
    if not os.path.isdir(original_dir):
        print(f"错误: 找不到原始目录: {original_dir}")
        return
    
    # 验证并创建输出目录
    create_output_directory(output_root)
    
    # 运行批量解码
    supported_symbols = [ZBarSymbol.QRCODE]
    run_pipeline_and_decode(
        original_dir=original_dir, 
        output_root=output_root, 
        supported_symbols=supported_symbols,
        force_run_pipeline=force_run_pipeline
    )



if __name__ == "__main__":
    main()