import os
import cv2
import numpy as np
from repair import QRCodeRepair
from pathlib import Path
import time

def prepare_test_image():
    """
    准备测试图片 - 创建一个带有损坏区域的二维码图像
    如果没有测试图片，可以生成一个简单的示例
    """
    test_dir = "./input_images"
    Path(test_dir).mkdir(parents=True, exist_ok=True)
    
    test_image_path = os.path.join(test_dir, "test_qr_damaged.jpg")
    
    # 检查是否已有测试图片
    if os.path.exists(test_image_path):
        print(f"使用现有的测试图片: {test_image_path}")
        return test_image_path
    
    # 如果没有测试图片，创建一个简单的测试图像
    print("创建测试图片...")
    
    # 创建一个简单的二维码样式图像（实际应用中应该使用真实的二维码）
    qr_size = 256
    qr_img = np.ones((qr_size, qr_size), dtype=np.uint8) * 255  # 白色背景
    
    # 绘制简单的黑白方块模拟二维码
    block_size = 16
    for i in range(4, qr_size - 4, block_size):
        for j in range(4, qr_size - 4, block_size):
            if (i // block_size + j // block_size) % 2 == 0:
                qr_img[i:i+block_size-2, j:j+block_size-2] = 0  # 黑色方块
    
    # 添加三个定位图案
    # 左上角
    qr_img[8:40, 8:40] = 255
    qr_img[12:36, 12:36] = 0
    qr_img[16:32, 16:32] = 255
    # 右上角
    qr_img[8:40, qr_size-40:qr_size-8] = 255
    qr_img[12:36, qr_size-36:qr_size-12] = 0
    qr_img[16:32, qr_size-32:qr_size-16] = 255
    # 左下角
    qr_img[qr_size-40:qr_size-8, 8:40] = 255
    qr_img[qr_size-36:qr_size-12, 12:36] = 0
    qr_img[qr_size-32:qr_size-16, 16:32] = 255
    
    # 添加一些损坏区域
    # 横线损坏
    cv2.line(qr_img, (0, qr_size//3), (qr_size, qr_size//3), (128, 128, 128), 5)
    # 竖线损坏
    cv2.line(qr_img, (qr_size//2, 0), (qr_size//2, qr_size), (128, 128, 128), 5)
    # 斑点损坏
    for _ in range(10):
        x = np.random.randint(0, qr_size - 20)
        y = np.random.randint(0, qr_size - 20)
        cv2.circle(qr_img, (x, y), 10, (128, 128, 128), -1)
    
    # 添加模糊区域
    blur_region = qr_img[qr_size//4:qr_size//2, qr_size//4:qr_size//2]
    blur_region = cv2.GaussianBlur(blur_region, (15, 15), 0)
    qr_img[qr_size//4:qr_size//2, qr_size//4:qr_size//2] = blur_region
    
    # 保存测试图片
    cv2.imwrite(test_image_path, qr_img)
    print(f"测试图片已保存至: {test_image_path}")
    
    return test_image_path

def test_single_image_repair():
    """
    测试单张图片修复功能
    """
    print("\n===== 测试单张图片修复 =====")
    
    # 准备测试图片
    test_image_path = prepare_test_image()
    
    # 创建修复工具实例
    print("初始化二维码修复工具...")
    qr_repair = QRCodeRepair()
    
    # 记录开始时间
    start_time = time.time()
    
    # 修复图片
    print("开始修复图片...")
    restored_img = qr_repair.auto_detect_and_repair(test_image_path)
    
    # 记录结束时间
    end_time = time.time()
    
    if restored_img is not None:
        print(f"修复完成，耗时: {end_time - start_time:.2f} 秒")
        print("修复结果已保存至 ./process/output_restored/")
    else:
        print("修复失败")

def test_batch_repair():
    """
    测试批量修复功能
    """
    print("\n===== 测试批量图片修复 =====")
    
    # 准备测试图片目录
    test_dir = "./input_images"
    
    # 如果测试目录为空，创建一些测试图片
    if len([f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]) == 0:
        prepare_test_image()
        # 创建更多测试图片
        base_image = cv2.imread(os.path.join(test_dir, "test_qr_damaged.jpg"), cv2.IMREAD_GRAYSCALE)
        for i in range(1, 4):
            # 添加不同类型的损坏
            test_img = base_image.copy()
            
            # 添加随机噪声
            noise = np.random.normal(0, 20, test_img.shape).astype(np.uint8)
            noisy_img = cv2.add(test_img, noise)
            
            # 保存新的测试图片
            test_img_path = os.path.join(test_dir, f"test_qr_damaged_{i}.jpg")
            cv2.imwrite(test_img_path, noisy_img)
            print(f"已创建额外测试图片: {test_img_path}")
    
    # 创建修复工具实例
    print("初始化二维码修复工具...")
    qr_repair = QRCodeRepair()
    
    # 记录开始时间
    start_time = time.time()
    
    # 批量修复图片
    print("开始批量修复图片...")
    qr_repair.process_directory(test_dir)
    
    # 记录结束时间
    end_time = time.time()
    
    print(f"批量修复完成，总耗时: {end_time - start_time:.2f} 秒")

def create_requirements():
    """
    创建requirements.txt文件，列出项目依赖
    """
    requirements_path = "./requirements.txt"
    
    # 检查requirements.txt是否已存在
    if os.path.exists(requirements_path):
        print(f"requirements.txt 已存在")
        return
    
    print("创建 requirements.txt 文件...")
    
    requirements = [
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "opencv-python>=4.5.0",
        "numpy>=1.20.0",
        "pyzbar>=0.1.8",
        "matplotlib>=3.4.0",
        "tqdm>=4.60.0",
        "ultralytics>=8.0.0",  # YOLOv8 依赖
    ]
    
    with open(requirements_path, "w") as f:
        f.write("\n".join(requirements))
    
    print(f"requirements.txt 已保存至: {requirements_path}")
    print("安装依赖命令: pip install -r requirements.txt")

def main():
    """
    主函数，运行所有测试
    """
    print("===== 二维码修复功能测试 =====")
    
    # 创建requirements.txt文件
    create_requirements()
    
    # 测试单张图片修复
    test_single_image_repair()
    
    # 测试批量修复
    test_batch_repair()
    
    print("\n===== 测试完成 =====")
    print("修复功能已成功实现，您可以通过以下方式使用:")
    print("1. 修复单张图片: python repair.py")
    print("2. 在Python代码中使用:")
    print("   from repair import QRCodeRepair")
    print("   qr_repair = QRCodeRepair()")
    print("   qr_repair.auto_detect_and_repair('your_image_path.jpg')")

if __name__ == "__main__":
    main()