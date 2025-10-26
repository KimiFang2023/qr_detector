import cv2
import numpy as np
from ultralytics import YOLO
from pyzbar import pyzbar
import time
from pathlib import Path
import os
from collections import deque

class CameraQRDetector:
    def __init__(self, model_path='./models/best.pt', camera_id=0, resolution=(640, 480), 
                 fps_limit=15, enable_preprocessing=True):
        """初始化摄像头二维码检测器
        
        Args:
            model_path: YOLO模型路径
            camera_id: 摄像头ID
            resolution: 摄像头分辨率 (宽度, 高度)
            fps_limit: FPS限制
            enable_preprocessing: 是否启用图像预处理
        """
        # 加载YOLO模型
        self.model = YOLO(model_path)
        
        # 设置摄像头
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
        # 性能优化参数
        self.fps_limit = fps_limit
        self.frame_interval = 1.0 / fps_limit
        self.last_frame_time = 0
        self.enable_preprocessing = enable_preprocessing
        
        # 创建结果保存目录
        self.results_dir = Path("./process/camera_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 二维码识别历史
        self.last_qr_data = None
        self.last_qr_time = 0
        self.detection_cooldown = 1.0  # 识别冷却时间(秒)
        
        # 性能统计
        self.frame_count = 0
        self.performance_stats = {
            'yolo_times': deque(maxlen=100),
            'qr_times': deque(maxlen=100),
            'total_times': deque(maxlen=100),
            'frame_skips': 0
        }
        
        # YOLO结果缓存
        self.last_yolo_result = None
        self.yolo_cache_time = 0
        self.yolo_cache_duration = 0.1  # 100ms缓存
        
        print("摄像头二维码检测器已初始化")
        print(f"分辨率: {resolution[0]}x{resolution[1]}")
        print(f"FPS限制: {fps_limit}")
        print(f"预处理: {'已启用' if enable_preprocessing else '已禁用'}")
    
    def preprocess_image(self, image):
        """传统图像预处理方法，优化二维码图像质量"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 应用CLAHE (对比度受限的自适应直方图均衡化)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_gray = clahe.apply(gray)
        
        # 应用高斯模糊降噪
        blurred = cv2.GaussianBlur(clahe_gray, (3, 3), 0)
        
        # 自适应阈值化
        thresh = cv2.adaptiveThreshold(blurred, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        # 中值滤波，进一步去除噪声
        processed = cv2.medianBlur(thresh, 3)
        
        return processed
    
    def detect_qr_codes(self, image):
        """使用pyzbar库识别二维码"""
        # 尝试识别原始图像中的二维码
        qr_codes = pyzbar.decode(image)
        
        # 如果原始图像识别失败，尝试处理后再识别
        if not qr_codes and self.enable_preprocessing:
            processed = self.preprocess_image(image)
            qr_codes = pyzbar.decode(processed)
            qr_codes = pyzbar.decode(cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR))
        
        results = []
        for qr_code in qr_codes:
            # 获取二维码数据和位置
            data = qr_code.data.decode("utf-8")
            rect = qr_code.rect
            
            # 提取二维码四个角的坐标（如果有）
            points = None
            if qr_code.polygon:
                points = np.array(qr_code.polygon, dtype=np.int32)
            
            results.append({
                'data': data,
                'rect': rect,
                'points': points
            })
        
        return results
    
    def get_performance_stats(self):
        """获取性能统计信息"""
        stats = {}
        for key, times in self.performance_stats.items():
            if key != 'frame_skips' and times:
                stats[key] = {
                    'avg': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'count': len(times)
                }
            elif key == 'frame_skips':
                stats[key] = times
        return stats

    def print_performance_stats(self):
        """打印性能统计信息"""
        stats = self.get_performance_stats()
        print("\n=== 性能统计 ===")
        for key, stat in stats.items():
            if isinstance(stat, dict):
                print(f"{key}: 平均={stat['avg']:.3f}s, 最小={stat['min']:.3f}s, 最大={stat['max']:.3f}s, 次数={stat['count']}")
            else:
                print(f"{key}: {stat}")
        print("===============\n")
    
    def draw_detections(self, frame, yolo_results, qr_results):
        """在图像上绘制检测结果"""
        # 绘制YOLO检测框
        for result in yolo_results:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()  # 获取置信度
            
            for box, cls, conf in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = map(int, box)
                
                # 绘制矩形框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 添加类别标签和置信度
                cls_name = self.model.names[int(cls)]
                label = f"{cls_name} {conf:.2f}"  # 格式化标签，显示类别和置信度
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # 绘制二维码识别结果
        current_time = time.time()
        for qr in qr_results:
            # 绘制二维码边界框
            rect = qr['rect']
            cv2.rectangle(frame, 
                         (rect.left, rect.top), 
                         (rect.left + rect.width, rect.top + rect.height), 
                         (255, 0, 0), 2)
            
            # 绘制多边形角点（如果有）
            if qr['points'] is not None:
                cv2.polylines(frame, [qr['points']], True, (0, 0, 255), 2)
            
            # 显示二维码数据，并进行冷却处理避免重复显示
            if qr['data'] != self.last_qr_data or current_time - self.last_qr_time > self.detection_cooldown:
                self.last_qr_data = qr['data']
                self.last_qr_time = current_time
                print(f"识别到二维码: {qr['data']}")
                
                # 保存识别到二维码的图像
                timestamp = int(time.time())
                img_filename = f"qr_detected_{timestamp}.jpg"
                img_path = self.results_dir / img_filename
                cv2.imwrite(str(img_path), frame)
                print(f"已保存图像: {img_path}")
            
            # 在图像上显示二维码内容（如果空间允许）
            if rect.top > 30:
                text_position = (rect.left, rect.top - 10)
            else:
                text_position = (rect.left, rect.top + rect.height + 30)
                
            # 简化显示，只显示部分内容
            display_text = qr['data'][:30] + ('...' if len(qr['data']) > 30 else '')
            cv2.putText(frame, display_text, text_position, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    def run(self):
        """启动摄像头实时检测 - 优化版本"""
        print("开始摄像头实时检测，按'q'键退出")
        
        # 优化的处理参数
        yolo_interval = 3  # 每3帧运行一次YOLO检测
        qr_interval = 2    # 每2帧运行一次二维码检测
        
        try:
            self.frame_count = 0
            start_time = time.time()
            
            while True:
                frame_start_time = time.time()
                
                # 控制帧率
                current_time = time.time()
                if current_time - self.last_frame_time < self.frame_interval:
                    self.performance_stats['frame_skips'] += 1
                    continue
                self.last_frame_time = current_time
                
                # 读取一帧图像
                ret, frame = self.cap.read()
                if not ret:
                    print("无法获取摄像头图像")
                    break
                
                # 创建图像副本用于处理
                frame_copy = frame.copy()
                
                # 使用缓存的YOLO结果
                yolo_results = []
                if (current_time - self.yolo_cache_time < self.yolo_cache_duration and 
                    self.last_yolo_result is not None):
                    yolo_results = self.last_yolo_result
                elif self.frame_count % yolo_interval == 0:
                    # 运行YOLO检测
                    yolo_start = time.time()
                    yolo_results = self.model(frame_copy, verbose=False, imgsz=640)
                    yolo_time = time.time() - yolo_start
                    self.performance_stats['yolo_times'].append(yolo_time)
                    
                    # 更新缓存
                    self.last_yolo_result = yolo_results
                    self.yolo_cache_time = current_time
                
                # 在检测到的区域内尝试识别二维码
                qr_results = []
                if yolo_results and self.frame_count % qr_interval == 0:
                    qr_start = time.time()
                    for result in yolo_results:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box)
                            
                            # 裁剪检测区域
                            roi = frame_copy[y1:y2, x1:x2]
                            
                            # 在ROI中识别二维码
                            roi_qr_results = self.detect_qr_codes(roi)
                            
                            # 调整二维码坐标到原始图像
                            for qr in roi_qr_results:
                                qr['rect'] = type('obj', (object,), {
                                    'left': qr['rect'].left + x1,
                                    'top': qr['rect'].top + y1,
                                    'width': qr['rect'].width,
                                    'height': qr['rect'].height
                                })
                                if qr['points'] is not None:
                                    qr['points'] += np.array([x1, y1])
                                
                                qr_results.append(qr)
                    qr_time = time.time() - qr_start
                    self.performance_stats['qr_times'].append(qr_time)
                
                # 即使YOLO没有检测到，也尝试在整幅图像中识别二维码（降低频率）
                if not qr_results and self.frame_count % (qr_interval * 3) == 0:
                    qr_start = time.time()
                    qr_results = self.detect_qr_codes(frame_copy)
                    qr_time = time.time() - qr_start
                    self.performance_stats['qr_times'].append(qr_time)
                
                # 绘制检测结果
                self.draw_detections(frame, yolo_results, qr_results)
                
                # 性能统计
                total_time = time.time() - frame_start_time
                self.performance_stats['total_times'].append(total_time)
                
                # 计算并显示FPS
                self.frame_count += 1
                elapsed_time = time.time() - start_time
                avg_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
                
                cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # 每100帧打印一次性能统计
                if self.frame_count % 100 == 0:
                    self.print_performance_stats()
                
                # 显示图像，修改窗口标题
                cv2.imshow('QR Detector - Press Q to Exit', frame)
                
                # 按'q'键退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("程序被用户中断")
        except Exception as e:
            print(f"发生错误: {e}")
        finally:
            # 打印最终性能统计
            self.print_performance_stats()
            # 释放资源
            self.cap.release()
            cv2.destroyAllWindows()
            print("摄像头已关闭，程序已退出")

if __name__ == "__main__":
    # 创建并运行检测器
    detector = CameraQRDetector(
        model_path='./models/best.pt',  # YOLO模型路径
        camera_id=0,                    # 默认摄像头
        resolution=(1280, 720),          # PC端更高分辨率
        fps_limit=30,                   # 提高FPS限制
        enable_preprocessing=True       # 启用图像预处理
    )
    detector.run()