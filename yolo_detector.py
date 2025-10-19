from ultralytics import YOLO
import os
import cv2
from pathlib import Path
from ultralytics import YOLO


def process_images(input_dir: str, output_boxed_dir: str, output_cropped_dir: str, model_path: str = 'yolov8n.pt') -> None:
    """Process all images in a directory using YOLOv8 model for object detection.

    Args:
        input_dir: Directory containing input images
        output_boxed_dir: Directory to save images with detected bounding boxes
        output_cropped_dir: Directory to save cropped detected objects
        model_path: Path to YOLOv8 model weights, default is pretrained yolov8n.pt
    """
    # 创建输出文件夹
    Path(output_boxed_dir).mkdir(parents=True, exist_ok=True)
    Path(output_cropped_dir).mkdir(parents=True, exist_ok=True)

    # 加载YOLOv8模型
    model = YOLO(model_path)

    # 获取输入文件夹中的所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_files = [f for f in os.listdir(input_dir)
                   if os.path.isfile(os.path.join(input_dir, f))
                   and Path(f).suffix.lower() in image_extensions]

    if not image_files:
        print(f"在输入文件夹 {input_dir} 中未找到任何图片文件")
        return

    # 处理每张图片
    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        print(f"处理图片: {img_path}")

        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}，跳过...")
            continue

        # 运行YOLOv8检测
        results = model(img)

        # 处理检测结果
        for result in results:
            # 绘制检测框
            annotated_img = result.plot()  # 生成带检测框的图片

            # 保存带检测框的图片
            boxed_img_path = os.path.join(output_boxed_dir, img_file)
            cv2.imwrite(boxed_img_path, annotated_img)

            # 裁剪并保存检测到的目标
            boxes = result.boxes.xyxy.cpu().numpy()  # 获取边界框坐标 (x1, y1, x2, y2)
            classes = result.boxes.cls.cpu().numpy()  # 获取类别索引

            for i, (box, cls) in enumerate(zip(boxes, classes)):
                x1, y1, x2, y2 = map(int, box)

                # 确保坐标在有效范围内
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img.shape[1], x2)
                y2 = min(img.shape[0], y2)

                # 裁剪目标区域
                cropped = img[y1:y2, x1:x2]

                # 获取类别名称
                cls_name = model.names[int(cls)]

                # 生成保存裁剪目标的文件名
                base_name = Path(img_file).stem
                ext = Path(img_file).suffix
                cropped_filename = f"{base_name}_obj_{i}_{cls_name}{ext}"
                cropped_path = os.path.join(output_cropped_dir, cropped_filename)

                # 保存裁剪的目标
                cv2.imwrite(cropped_path, cropped)

        print(f"已处理: {img_file}，结果已保存")

    print("所有图片处理完成!")


if __name__ == "__main__":
    # 配置文件夹路径
    input_directory = "./process/input_images"  # 输入图片所在文件夹
    boxed_output_directory = "./process/output_boxed"  # 带检测框的图片输出文件夹
    cropped_output_directory = "./process/output_cropped"  # 裁剪目标的输出文件夹

    # 可以替换为自己的模型路径，例如 'yolov8s.pt', 'yolov8m.pt' 或自定义训练的模型
    model_path = "./models/best.pt"

    # 执行处理
    process_images(input_directory, boxed_output_directory, cropped_output_directory, model_path)