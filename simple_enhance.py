import os
import cv2
import numpy as np
from pathlib import Path

def enhance_image(image):
    # 确保图像为灰度图
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 应用CLAHE增强对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 应用高斯模糊和非锐化掩模
    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    sharp = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)
    
    return sharp

def batch_process(input_dir, output_dir):
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 支持的图像格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(input_dir) 
                  if Path(f).suffix.lower() in image_extensions]
    
    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f'无法读取图像: {img_file}')
            continue
        
        enhanced_img = enhance_image(img)
        out_path = os.path.join(output_dir, img_file)
        cv2.imwrite(out_path, enhanced_img)
        print(f'已保存增强图像: {out_path}')

if __name__ == '__main__':
    # 示例调用
    batch_process('./process/input_images', './process/output_enhanced')