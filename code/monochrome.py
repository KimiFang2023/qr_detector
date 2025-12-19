import cv2
import os
from pathlib import Path


def convert_to_black_white(image):
    """
    将彩色图片转换为黑白（灰度）图片

    参数:
        image: 输入的BGR格式图片
    返回:
        gray_image: 转换后的黑白图片
    """
    # 转换为灰度图（黑白）
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 可以根据需要添加二值化处理，使黑白对比更强烈
    # _, gray_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    return gray_image


def process_to_black_white(input_dir, output_dir):
    """
    批量处理图片，将其转换为黑白效果

    参数:
        input_dir: 输入图片文件夹（校正后的图片）
        output_dir: 黑白图片输出文件夹
    """
    # 创建输出文件夹
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 获取所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(input_dir)
                   if os.path.isfile(os.path.join(input_dir, f))
                   and Path(f).suffix.lower() in image_extensions]

    if not image_files:
        print(f"在 {input_dir} 中未找到任何图片文件")
        return

    # 处理每张图片
    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        print(f"处理: {img_path}")

        # 读取图片
        image = cv2.imread(img_path)
        if image is None:
            print(f"无法读取图片 {img_file}，已跳过")
            continue

        # 转换为黑白图片
        bw_image = convert_to_black_white(image)

        # 保存黑白图片
        output_path = os.path.join(output_dir, img_file)
        cv2.imwrite(output_path, bw_image)
        print(f"已保存黑白图片: {output_path}")

    print("所有图片已转换为黑白效果!")


if __name__ == "__main__":
    # 配置文件夹路径（与之前的校正后图片文件夹对应）
    corrected_input = "./process/output_corrected"  # 输入：校正角度后的图片文件夹
    bw_output = "./process/output_monochrome"  # 输出：黑白图片文件夹

    # 执行处理
    process_to_black_white(corrected_input, bw_output)