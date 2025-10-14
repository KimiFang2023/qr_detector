import cv2
import numpy as np
from ultralytics import YOLO
from pyzbar import pyzbar
import time
from pathlib import Path
import os
import sys
import signal
from picamera2 import Picamera2, Preview
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue, Empty
import psutil

# 全局退出标志
global should_exit
should_exit = False

class RaspberryPiQRDetector:
    def __init__(self, model_path='./models/best.pt', resolution=(640, 480), 
                 fps_limit=30, enable_preprocessing=False, save_images=False,
                 yolo_confidence=0.4, image_save_interval=5.0, 
                 use_gpu=True, enable_multiprocessing=True):
        """初始化树莓派5摄像头二维码检测器
        
        Args:
            model_path: YOLO模型路径
            resolution: 摄像头分辨率 (宽度, 高度) - 树莓派5支持更高分辨率
            fps_limit: FPS限制 - 树莓派5可以支持更高FPS
            enable_preprocessing: 是否启用图像预处理
            save_images: 是否保存检测到的图像
            yolo_confidence: YOLO检测置信度阈值
            image_save_interval: 图像保存的最小间隔(秒)
            use_gpu: 是否使用GPU加速（如果可用）
            enable_multiprocessing: 是否启用多进程处理
        """
        # 检测树莓派5的硬件能力
        self.cpu_count = mp.cpu_count()
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        self.use_gpu = use_gpu and self._check_gpu_support()
        self.enable_multiprocessing = enable_multiprocessing and self.cpu_count >= 4
        
        print(f"检测到硬件配置: {self.cpu_count}核CPU, {self.memory_gb:.1f}GB内存")
        print(f"GPU支持: {'是' if self.use_gpu else '否'}")
        print(f"多进程支持: {'是' if self.enable_multiprocessing else '否'}")
        
        # 加载YOLO模型 - 针对树莓派5优化
        try:
            self.model = YOLO(model_path)
            # 针对树莓派5优化模型设置
            if self.use_gpu:
                self.model.to('cuda' if self.use_gpu else 'cpu')
            print(f"成功加载YOLO模型: {model_path}")
        except Exception as e:
            print(f"加载模型失败: {e}")
            sys.exit(1)
        
        # 设置摄像头 - 针对树莓派5优化配置
        self.picam2 = None
        try:
            # 创建摄像头对象
            self.picam2 = Picamera2()
            
            # 树莓派5优化的摄像头配置
            camera_config = self.picam2.create_still_configuration(
                main={"size": resolution},
                controls={
                    "FrameRate": fps_limit,
                    "NoiseReductionMode": 0,  # 关闭降噪以提高性能
                    "Sharpness": 1.0,         # 降低锐化以减少计算
                    "Brightness": 0.5,        # 优化亮度
                    "Contrast": 1.0,          # 优化对比度
                    "Saturation": 1.0,        # 优化饱和度
                    "ExposureTime": 1000,     # 固定曝光时间以提高一致性
                    "AnalogueGain": 1.0,      # 固定模拟增益
                    "DigitalGain": 1.0        # 固定数字增益
                }
            )
            self.picam2.configure(camera_config)
            
            # 启动摄像头
            self.picam2.start()
            print("摄像头启动成功")
        except Exception as e:
            print(f"初始化摄像头失败: {e}")
            sys.exit(1)
        
        # 性能优化参数
        self.fps_limit = fps_limit
        self.frame_interval = 1.0 / fps_limit
        self.last_frame_time = 0
        self.enable_preprocessing = enable_preprocessing
        self.save_images = save_images
        self.yolo_confidence = yolo_confidence
        self.last_save_time = 0
        self.image_save_interval = image_save_interval
        
        # 创建结果保存目录
        if save_images:
            self.results_dir = Path("/home/pi/QR_Code_Results")
            self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 二维码识别历史
        self.last_qr_data = None
        self.last_qr_time = 0
        self.detection_cooldown = 1.0  # 识别冷却时间(秒)
        
        # 性能统计
        self.frame_count = 0
        self.start_time = time.time()
        
        print("树莓派摄像头二维码检测器已初始化")
        print(f"分辨率: {resolution[0]}x{resolution[1]}")
        print(f"FPS限制: {fps_limit}")
        print(f"预处理: {'已启用' if enable_preprocessing else '已禁用'}")
        print(f"图像保存: {'已启用' if save_images else '已禁用'}")
        print(f"YOLO置信度阈值: {yolo_confidence}")
        if save_images:
            print(f"结果保存目录: {self.results_dir}")
        print("按 Ctrl+C 或 'q' 键退出程序")
        
        # 初始化多进程支持
        if self.enable_multiprocessing:
            self.executor = ThreadPoolExecutor(max_workers=min(4, self.cpu_count))
            self.frame_queue = Queue(maxsize=3)  # 限制队列大小防止内存溢出
            self.result_queue = Queue(maxsize=3)
        else:
            self.executor = None
            self.frame_queue = None
            self.result_queue = None
        
        # 注册信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def _check_gpu_support(self):
        """检查GPU支持"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def preprocess_image(self, image):
        """优化的图像预处理方法，针对树莓派5优化"""
        # 使用更高效的灰度转换
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 使用自适应阈值以提高二维码识别率
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        return thresh
    
    def yolo_inference_optimized(self, frame):
        """优化的YOLO推理方法"""
        try:
            # 使用更小的输入尺寸以提高速度
            results = self.model(frame, verbose=False, 
                               conf=self.yolo_confidence, 
                               imgsz=416,  # 适中的输入尺寸
                               device='cuda' if self.use_gpu else 'cpu',
                               half=self.use_gpu)  # 使用半精度浮点数（如果GPU可用）
            return results
        except Exception as e:
            print(f"YOLO推理错误: {e}")
            return []

    def detect_qr_codes(self, image):
        """高度优化的二维码识别方法，针对树莓派5优化"""
        results = []
        
        try:
            # 多级检测策略：从简单到复杂
            # 1. 首先尝试原始图像
            qr_codes = pyzbar.decode(image)
            
            # 2. 如果没检测到且启用预处理，尝试预处理图像
            if not qr_codes and self.enable_preprocessing:
                processed = self.preprocess_image(image)
                qr_codes = pyzbar.decode(processed)
            
            # 3. 如果还是没检测到，尝试缩放图像
            if not qr_codes:
                # 尝试放大图像（有时小图像检测效果更好）
                scale_factor = 2.0
                h, w = image.shape[:2]
                scaled = cv2.resize(image, (int(w * scale_factor), int(h * scale_factor)))
                qr_codes = pyzbar.decode(scaled)
                
                # 如果检测到，需要调整坐标
                if qr_codes:
                    for qr_code in qr_codes:
                        qr_code.rect = type('obj', (object,), {
                            'left': int(qr_code.rect.left / scale_factor),
                            'top': int(qr_code.rect.top / scale_factor),
                            'width': int(qr_code.rect.width / scale_factor),
                            'height': int(qr_code.rect.height / scale_factor)
                        })
            
            # 处理检测结果
            for qr_code in qr_codes:
                try:
                    data = qr_code.data.decode("utf-8")
                    rect = qr_code.rect
                    
                    results.append({
                        'data': data,
                        'rect': rect,
                        'points': np.array(qr_code.polygon, dtype=np.int32) if qr_code.polygon else None
                    })
                except UnicodeDecodeError:
                    # 跳过无法解码的二维码
                    continue
                    
        except Exception as e:
            print(f"二维码检测错误: {e}")
        
        return results

    def draw_detections(self, frame, yolo_results, qr_results):
        """高度优化的绘制函数，减少计算和渲染开销"""
        current_time = time.time()
        
        # 预分配绘制缓冲区以减少内存分配
        if not hasattr(self, '_draw_buffer'):
            self._draw_buffer = np.zeros_like(frame)
        
        # 绘制YOLO检测框 - 使用更高效的绘制
        for result in yolo_results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                # 限制绘制的检测框数量以提高性能
                max_boxes = min(3, len(boxes))
                for i in range(max_boxes):
                    box = boxes[i]
                    cls = classes[i]
                    
                    x1, y1, x2, y2 = map(int, box)
                    
                    # 使用更细的线条以减少绘制开销
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        # 绘制二维码识别结果 - 进一步优化
        for qr in qr_results:
            rect = qr['rect']
            
            # 绘制二维码边界框
            cv2.rectangle(frame, 
                         (rect.left, rect.top), 
                         (rect.left + rect.width, rect.top + rect.height), 
                         (255, 0, 0), 1)
            
            # 智能显示逻辑，避免重复输出
            if qr['data'] != self.last_qr_data or current_time - self.last_qr_time > self.detection_cooldown:
                self.last_qr_data = qr['data']
                self.last_qr_time = current_time
                print(f"识别到二维码: {qr['data']}")
                
                # 条件性保存图像，限制保存频率
                if self.save_images and (current_time - self.last_save_time > self.image_save_interval):
                    self.last_save_time = current_time
                    timestamp = int(time.time())
                    img_filename = f"qr_detected_{timestamp}.jpg"
                    img_path = self.results_dir / img_filename
                    # 使用更低的质量保存图像以节省空间
                    cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    print(f"已保存图像: {img_path}")
            
            # 优化的文本显示
            if rect.top > 20:
                text_position = (rect.left, rect.top - 5)
            else:
                text_position = (rect.left, rect.top + rect.height + 20)
                
            # 只显示部分内容，使用更小的字体
            display_text = qr['data'][:15] + ('...' if len(qr['data']) > 15 else '')
            cv2.putText(frame, display_text, text_position, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    def signal_handler(self, sig, frame):
        """捕获Ctrl+C信号，优雅退出程序"""
        print("\n用户请求退出，正在关闭摄像头...")
        # 直接设置一个退出标志，而不是在信号处理程序中调用close方法
        # 这样可以确保在主线程中关闭资源
        global should_exit
        should_exit = True
        
    def close(self):
        """关闭摄像头和释放资源"""
        if self.picam2 is not None:
            self.picam2.stop_preview()
            self.picam2.stop()
        
        # 关闭多进程支持
        if self.executor is not None:
            self.executor.shutdown(wait=True)
        
        cv2.destroyAllWindows()
        print("摄像头已关闭")
    
    def process_frame_async(self, frame):
        """异步处理帧数据"""
        if not self.enable_multiprocessing:
            return self.yolo_inference_optimized(frame)
        
        try:
            # 提交任务到线程池
            future = self.executor.submit(self.yolo_inference_optimized, frame)
            return future.result(timeout=0.1)  # 设置超时避免阻塞
        except Exception as e:
            print(f"异步处理错误: {e}")
            return []
    
    def get_performance_stats(self):
        """获取性能统计信息"""
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            avg_fps = self.frame_count / elapsed_time
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            return {
                'fps': avg_fps,
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'frame_count': self.frame_count
            }
        return None
        
    def run(self):
        """高度优化的实时检测主循环 - 针对树莓派5优化"""
        print("开始树莓派5摄像头实时检测")
        print("提示：已启用高性能优化模式")
        
        try:
            # 重置全局退出标志
            global should_exit
            should_exit = False
            
            # 初始化性能统计
            self.frame_count = 0
            self.start_time = time.time()
            self.last_stats_time = time.time()
            
            # 自适应处理参数
            yolo_interval = 3 if self.enable_multiprocessing else 2  # 多进程时减少YOLO频率
            qr_interval = 2  # 二维码检测间隔
            
            # 预分配缓冲区
            frame_buffer = None
            
            while not should_exit:
                # 精确的帧率控制
                current_time = time.time()
                if current_time - self.last_frame_time < self.frame_interval:
                    remaining_time = self.frame_interval - (current_time - self.last_frame_time)
                    if remaining_time > 0.001:
                        time.sleep(remaining_time * 0.8)  # 减少休眠时间以提高响应性
                    continue
                self.last_frame_time = current_time
                
                # 高效获取图像
                frame = self.picam2.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                yolo_results = []
                qr_results = []
                
                # 智能检测策略
                if self.frame_count % yolo_interval == 0:
                    # 异步YOLO检测
                    yolo_results = self.process_frame_async(frame)
                    
                    # 在YOLO检测区域中识别二维码
                    if yolo_results:
                        for result in yolo_results:
                            if result.boxes is not None:
                                boxes = result.boxes.xyxy.cpu().numpy()
                                
                                for box in boxes:
                                    x1, y1, x2, y2 = map(int, box)
                                    
                                    # 确保ROI有效
                                    if x2 > x1 and y2 > y1:
                                        roi = frame[y1:y2, x1:x2]
                                        roi_qr_results = self.detect_qr_codes(roi)
                                        
                                        # 调整坐标
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
                
                # 全图二维码检测（降低频率）
                if not qr_results and self.frame_count % qr_interval == 0:
                    # 使用更小的图像进行检测
                    h, w = frame.shape[:2]
                    small_frame = cv2.resize(frame, (w//2, h//2))
                    small_qr_results = self.detect_qr_codes(small_frame)
                    
                    # 调整坐标
                    for qr in small_qr_results:
                        qr['rect'] = type('obj', (object,), {
                            'left': qr['rect'].left * 2,
                            'top': qr['rect'].top * 2,
                            'width': qr['rect'].width * 2,
                            'height': qr['rect'].height * 2
                        })
                        if qr['points'] is not None:
                            qr['points'] = (qr['points'] * 2).astype(np.int32)
                        
                        qr_results.append(qr)
                
                # 绘制检测结果
                self.draw_detections(frame, yolo_results, qr_results)
                
                # 性能监控和显示
                self.frame_count += 1
                if current_time - self.last_stats_time > 2.0:  # 每2秒更新一次统计
                    stats = self.get_performance_stats()
                    if stats:
                        # 显示性能信息
                        cv2.putText(frame, f"FPS: {stats['fps']:.1f}", (5, 15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        cv2.putText(frame, f"CPU: {stats['cpu_percent']:.1f}%", (5, 35), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                        cv2.putText(frame, f"RAM: {stats['memory_percent']:.1f}%", (5, 55), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    self.last_stats_time = current_time
                
                # 高效显示
                cv2.imshow('QR Detector - Pi5 Optimized', frame)
                
                # 检查退出条件
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
                if should_exit:
                    break
        except KeyboardInterrupt:
            print("程序被用户中断")
        except Exception as e:
            print(f"发生错误: {e}")
        finally:
            # 释放资源
            self.close()
            print("程序已退出")

if __name__ == "__main__":
    # 树莓派5优化配置
    resolution = (640, 480)  # 树莓派5支持更高分辨率
    fps_limit = 30          # 树莓派5可以支持更高FPS
    
    print("=" * 50)
    print("树莓派5 QR码检测器 - 高性能优化版本")
    print("=" * 50)
    
    # 创建并运行检测器
    detector = RaspberryPiQRDetector(
        model_path='./models/best.pt',      # YOLO模型路径
        resolution=resolution,              # 优化的分辨率
        fps_limit=fps_limit,                # 优化的FPS限制
        enable_preprocessing=True,          # 启用预处理以提高识别率
        save_images=False,                  # 默认禁用图像保存以提高性能
        yolo_confidence=0.4,                # 平衡的置信度阈值
        use_gpu=True,                      # 启用GPU加速（如果可用）
        enable_multiprocessing=True         # 启用多进程支持
    )
    
    print("\n启动检测器...")
    detector.run()