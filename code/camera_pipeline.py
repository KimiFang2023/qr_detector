import numpy as np
import cv2
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from pyzbar.pyzbar import decode, ZBarSymbol

# 导入项目中的其他模块
try:
    # 直接导入当前目录下的模块
    from angle import correct_skew
    from binarize import binarize_image
    from enhance import enhance_image
    from repair_wrapper import repair_if_needed
except ImportError as e:
    print(f"警告：无法导入某些模块: {e}")
    print("尝试使用相对路径导入...")
    try:
        # 尝试使用相对路径导入
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from angle import correct_skew
        from binarize import binarize_image
        from enhance import enhance_image
        from repair_wrapper import repair_if_needed
    except ImportError:
        print("导入失败，使用内置实现的图像处理函数")
        # 定义自定义实现的图像处理函数
        def correct_skew(img):
            """检测图像倾斜角度并进行校正"""
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Binarization
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Morphological operation to remove noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

            # Find contours
            contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return img  # No contours found, return original image

            # Find the largest contour (assumed to be the QR code)
            largest_contour = max(contours, key=cv2.contourArea)

            # Fit minimum area rectangle
            rect = cv2.minAreaRect(largest_contour)
            angle = rect[2]

            # Adjust angle (minAreaRect returns angle in a special range)
            if angle < -45:
                angle += 90
            elif angle > 45:
                angle -= 90

            # Calculate rotation center and matrix
            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)

            # Calculate new image size after rotation to avoid cropping
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))

            # Adjust translation part of the rotation matrix
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]

            # Perform affine transformation (rotation)
            corrected = cv2.warpAffine(
                img, M, (new_w, new_h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255)  # White background fill
            )

            return corrected
        
        def binarize_image(img):
            """图像二值化处理"""
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            # 使用自适应阈值进行二值化，适合光照不均的情况
            binary = cv2.adaptiveThreshold(gray, 255, 
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 
                                          11, 2)
            return binary
        
        def enhance_image(img):
            """图像增强处理"""
            # 锐化滤波器增强边缘
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(img, -1, kernel)
            # 形态学操作：膨胀后腐蚀，增强二维码的完整性
            kernel = np.ones((2, 2), np.uint8)
            morph = cv2.morphologyEx(sharpened, cv2.MORPH_DILATE, kernel)
            morph = cv2.morphologyEx(morph, cv2.MORPH_ERODE, kernel)
            return morph
        
        def repair_if_needed(img, *args, **kwargs):
            """图像修复处理"""
            # 简单的修复：中值滤波去除噪声
            repaired = cv2.medianBlur(img, 3)
            return repaired, 0

# 设置支持中文的字体
def setup_font():
    """设置PIL字体以支持中文显示"""
    # 在Windows系统上，这些是常见的中文字体路径
    font_paths = [
        "C:/Windows/Fonts/simhei.ttf",  # 黑体
        "C:/Windows/Fonts/msyh.ttc",   # 微软雅黑
        "C:/Windows/Fonts/simsun.ttc", # 宋体
    ]
    
    # 尝试加载中文字体
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, 16)  # 16是字体大小
            except Exception:
                continue
    
    # 如果找不到中文字体，返回默认字体
    print("警告：无法找到支持中文的字体，可能会出现显示问题")
    return None

# 使用PIL绘制中文文本的函数
def put_text(img, text, org, font_face, font_scale, color, thickness=1):
    """
    使用PIL在图像上绘制文本，支持中英文显示
    
    参数:
        img: 目标图像（numpy数组格式）
        text: 要绘制的文本
        org: 文本的左下角坐标
        font_face: 字体类型（这里可以忽略，使用PIL的字体）
        font_scale: 字体大小
        color: 文本颜色
        thickness: 文本线条粗细
    
    返回:
        绘制后的图像
    """
    try:
        # 获取PIL字体
        pil_font = setup_font()
        
        # 如果有PIL字体，使用PIL绘制中文
        if pil_font:
            # 将OpenCV图像转换为PIL图像
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            # 计算字体大小
            font_size = int(16 * font_scale)  # 基础大小16乘以缩放因子
            if font_size < 8:  # 确保字体不会太小
                font_size = 8
                
            # 使用PIL绘制文本
            try:
                # 尝试调整字体大小
                pil_font = pil_font.font_variant(size=font_size)
            except Exception:
                # 如果调整大小失败，使用默认字体
                pil_font = ImageFont.load_default()
                
            # 绘制文本，注意PIL的坐标原点是左上角，而OpenCV的原点也是左上角，但我们需要调整位置
            x, y = org
            # 将BGR颜色转换为RGB
            rgb_color = (color[2], color[1], color[0])
            # 绘制文本
            draw.text((x, y - font_size), text, font=pil_font, fill=rgb_color)
            
            # 将PIL图像转换回OpenCV图像
            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            return img
        else:
            # 如果没有PIL字体，回退到OpenCV绘制
            cv2.putText(img, text, org, font_face, font_scale, color, thickness, cv2.LINE_AA)
            return img
    except Exception as e:
        # 如果失败，尝试使用英文替代
        print(f"绘制文本时出错: {e}")
        # 英文替代文本
        if text == "解码成功":
            cv2.putText(img, "Decoded", org, font_face, font_scale, color, thickness, cv2.LINE_AA)
        elif text == "解码失败":
            cv2.putText(img, "Not Decoded", org, font_face, font_scale, color, thickness, cv2.LINE_AA)
        else:
            cv2.putText(img, text, org, font_face, font_scale, color, thickness, cv2.LINE_AA)
        return img

# 解码二维码函数
def decode_qr_code(image, supported_symbols=None):
    """
    使用pyzbar库解码图像中的二维码
    
    参数:
        image: 输入图像（BGR格式）
        supported_symbols: 支持的条码类型列表
    
    返回:
        解码结果列表，如果没有解码成功则返回空列表
    """
    try:
        # 转换为PIL图像以便pyzbar处理
        if isinstance(image, np.ndarray):
            # 转换为RGB格式
            rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)
        else:
            pil_img = image
        
        # 转换为灰度图以提高识别率
        gray_img = pil_img.convert('L')
        
        # 设置要识别的条码类型
        if supported_symbols:
            results = decode(gray_img, symbols=supported_symbols)
        else:
            results = decode(gray_img)
        
        # 提取解码数据
        decoded_data = []
        for result in results:
            data = result.data.decode('utf-8')
            symbol_type = result.type
            decoded_data.append({
                'type': symbol_type,
                'content': data,
                'polygon': result.polygon,
                'rect': result.rect
            })
        
        return decoded_data
    except Exception as e:
        # 静默处理错误，不输出到终端
        # print(f"解码错误: {str(e)}")
        return []

# 使用完整pipeline处理图像，优化远距离二维码识别
def process_with_pipeline(image):
    """
    使用完整的pipeline处理图像：校正、二值化、增强、修复，优化远距离二维码识别
    
    参数:
        image: 输入图像
    
    返回:
        处理后的图像
    """
    try:
        # 1. 图像预处理：提高分辨率和降噪
        # 复制图像以避免修改原图
        processed_img = image.copy()
        
        # 尝试提高图像分辨率，特别是对于远距离二维码
        height, width = processed_img.shape[:2]
        # 如果图像太小，进行放大（特别是对于远距离捕获的小二维码）
        scale_factor = 2.0  # 默认放大倍数
        if height < 200 or width < 200:
            scale_factor = 3.0  # 更小的图像放大3倍
        elif height < 300 or width < 300:
            scale_factor = 2.0  # 中等大小的图像放大2倍
        
        if scale_factor > 1.0:
            processed_img = cv2.resize(processed_img, None, fx=scale_factor, fy=scale_factor,
                                      interpolation=cv2.INTER_CUBIC)  # 使用立方插值提高质量
        
        # 高斯模糊去除噪声，对远距离图像更保守的降噪
        processed_img = cv2.GaussianBlur(processed_img, (3, 3), 0)
        
        # 2. 角度校正 - 使用项目中的correct_skew函数
        corrected_img = correct_skew(processed_img)
        
        # 3. 二值化处理 - 使用项目中的binarize_image函数
        binary_img = binarize_image(corrected_img)
        
        # 4. 图像增强 - 使用项目中的enhance_image函数
        enhanced_img = enhance_image(binary_img)
        
        # 额外的增强步骤：针对远距离二维码的特殊处理
        # 再次锐化，增强二维码的细节
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        enhanced_img = cv2.filter2D(enhanced_img, -1, kernel)
        
        # 5. 修复（如果需要） - 使用项目中的repair_if_needed函数
        repaired_img, _ = repair_if_needed(enhanced_img)
        
        return repaired_img
    except Exception as e:
        print(f"Pipeline处理错误: {str(e)}")
        # 返回原始图像作为最后的回退
        return image

# 绘制解码结果
def draw_decoded_results(frame, detected_boxes, decoded_results_list):
    """
    在图像上绘制检测框和解码结果
    
    参数:
        frame: 原始帧
        detected_boxes: 检测框列表
        decoded_results_list: 每个检测框的解码结果列表
    
    返回:
        绘制后的帧
    """
    font = setup_font()
    for i, box in enumerate(detected_boxes):
        # 绘制检测框
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 显示解码结果
        if i < len(decoded_results_list):
            decoded_results = decoded_results_list[i]
            if decoded_results:
                # 解码成功，使用文本绘制函数
                frame = put_text(frame, "解码成功", (x1, y1 - 10), font, 0.9, (0, 255, 0), 2)
                
                # 显示解码内容（限制显示长度）
                content = decoded_results[0]['content']
                display_content = (content[:20] + '...') if len(content) > 20 else content
                frame = put_text(frame, display_content, (x1, y2 + 30), font, 0.6, (0, 255, 0), 2)
            else:
                # 解码失败，使用文本绘制函数
                frame = put_text(frame, "解码失败", (x1, y1 - 10), font, 0.9, (0, 0, 255), 2)
        
    return frame

def run_realtime_detection(camera_index=0):
    """
    实时检测摄像头画面中的二维码，先尝试传统解码，失败则使用pipeline处理
    
    参数:
        camera_index: 摄像头索引，默认0为系统默认摄像头
    """
    # 加载YOLOv8模型
    model = YOLO("./models/best.pt")  # 可替换为其他模型如yolov8s.pt、yolov8m.pt等
    
    # 打开摄像头
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print("无法打开摄像头，请检查设备连接或索引是否正确")
        return
    
    print("开始实时检测...")
    print("按 'q' 键退出程序")
    
    # 仅识别QR码
    supported_symbols = [ZBarSymbol.QRCODE]
    
    # 标志变量，用于跟踪是否遇到了OpenCV GUI错误
    opencv_gui_error = False
    
    while True:
        # 读取一帧画面
        ret, frame = cap.read()
        
        if not ret:
            print("无法获取摄像头画面，程序退出")
            break
        
        # 运行YOLOv8检测，禁止输出
        results = model(frame, verbose=False)
        
        # 获取检测框
        detected_boxes = []
        if results and len(results) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            for box in boxes:
                detected_boxes.append(box)
        
        # 为每个检测到的区域尝试解码
        decoded_results_list = []
        
        for box in detected_boxes:
            x1, y1, x2, y2 = map(int, box)
            
            # 裁剪二维码区域
            qr_roi = frame[y1:y2, x1:x2].copy()
            
            # 1. 先尝试传统方法解码
            decoded_results = decode_qr_code(qr_roi, supported_symbols)
            
            # 2. 如果传统解码失败，则使用优化后的pipeline处理并尝试多种方法解码
            if not decoded_results:
                # 优化1: 使用pipeline处理图像
                processed_roi = process_with_pipeline(qr_roi)
                
                # 优化2: 尝试解码处理后的图像
                decoded_results = decode_qr_code(processed_roi, supported_symbols)
                
                # 优化3: 如果仍然失败，尝试其他阈值方法
                if not decoded_results:
                    # 尝试不同的二值化方法
                    gray_roi = cv2.cvtColor(qr_roi, cv2.COLOR_BGR2GRAY)
                    _, thresh1 = cv2.threshold(gray_roi, 127, 255, cv2.THRESH_BINARY)
                    decoded_results = decode_qr_code(thresh1, supported_symbols)
                
                # 优化4: 如果还是失败，尝试旋转不同角度
                rotation_angles = [5, -5, 10, -10]  # 尝试的旋转角度
                for angle in rotation_angles:
                    if not decoded_results:
                        # 旋转图像
                        height, width = processed_roi.shape[:2]
                        center = (width // 2, height // 2)
                        M = cv2.getRotationMatrix2D(center, angle, 1.0)
                        rotated_img = cv2.warpAffine(processed_roi, M, (width, height))
                        
                        # 尝试解码旋转后的图像
                        decoded_results = decode_qr_code(rotated_img, supported_symbols)
                        if decoded_results:
                            break
            
            decoded_results_list.append(decoded_results)
        
        # 绘制结果
        annotated_frame = draw_decoded_results(frame, detected_boxes, decoded_results_list)
        
        # 显示处理后的画面
        try:
            cv2.imshow("二维码检测", annotated_frame)  # 简化窗口标题，减少中文
            
            # 按 'q' 键退出循环
            if cv2.waitKey(1) == ord('q'):
                break
        except cv2.error as e:
            # 捕获OpenCV显示错误
            print("\nOpenCV显示错误，这通常是由于OpenCV库没有正确配置显示支持。")
            print("请按照以下步骤解决：")
            print("1. 重新安装OpenCV库，确保它支持Windows显示功能")
            print("2. 或使用命令：pip install opencv-python-headless 来安装没有GUI支持的版本")
            print(f"详细错误信息：{e}")
            opencv_gui_error = True
            break
    
    # 释放资源
    cap.release()
    
    # 只有在没有GUI错误的情况下才尝试销毁窗口
    if not opencv_gui_error:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            # 如果销毁窗口也失败，静默处理
            pass
    
    print("检测结束")

if __name__ == "__main__":
    # 可修改摄像头索引，多个摄像头时尝试1、2等
    run_realtime_detection(camera_index=0)
    