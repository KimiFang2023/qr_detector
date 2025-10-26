import cv2
import numpy as np
from ultralytics import YOLO
from pyzbar import pyzbar
import time
from pathlib import Path
import os
import sys
import signal
import threading
from collections import deque
from picamera2 import Picamera2, Preview

# 全局退出标志
global should_exit
should_exit = False

class RaspberryPiQRDetector:
    def __init__(self, model_path='./models/best.pt', resolution=(480, 320), 
                 fps_limit=20, enable_preprocessing=False, save_images=False,
                 yolo_confidence=0.4, image_save_interval=5.0):
        """初始化树莓派摄像头二维码检测器
        
        Args:
            model_path: YOLO模型路径
            resolution: 摄像头分辨率 (宽度, 高度)
            fps_limit: FPS限制
            enable_preprocessing: 是否启用图像预处理
            save_images: 是否保存检测到的图像
            yolo_confidence: YOLO检测置信度阈值
            image_save_interval: 图像保存的最小间隔(秒)
        """
        # 加载YOLO模型 - 使用轻量级模式和优化设置
        try:
            self.model = YOLO(model_path)
            # 注意：不在初始化时转换为 half，让 YOLO 在推理时自动处理
            # 以避免类型不匹配问题
            print(f"成功加载YOLO模型: {model_path}")
        except Exception as e:
            print(f"加载模型失败: {e}")
            sys.exit(1)
        
        # 设置摄像头 - 使用Picamera2
        self.picam2 = None
        try:
            # 创建摄像头对象
            self.picam2 = Picamera2()
            
            # 配置摄像头参数，使用更轻量级的设置
            camera_config = self.picam2.create_still_configuration(
                main={"size": resolution},
                controls={
                    "FrameRate": fps_limit,
                    "NoiseReductionMode": 0  # 关闭降噪以提高性能
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
            self.results_dir = Path("/home/kimi/QR_Code_Results")
            self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 二维码识别历史
        self.last_qr_data = None
        self.last_qr_time = 0
        self.detection_cooldown = 1.0  # 识别冷却时间(秒)
        
        # 性能统计
        self.frame_count = 0
        self.start_time = time.time()
        
        # 多线程优化
        self.frame_queue = deque(maxlen=3)  # 限制队列大小
        self.processing_lock = threading.Lock()
        self.capture_thread = None
        self.processing_thread = None
        
        # 缓存优化
        self.last_yolo_result = None
        self.yolo_cache_time = 0
        self.yolo_cache_duration = 0.1  # 100ms缓存
        
        # 图像预处理缓存
        self.preprocessed_cache = None
        self.cache_timestamp = 0
        
        # 性能监控
        self.performance_stats = {
            'yolo_times': deque(maxlen=30),
            'qr_times': deque(maxlen=30),
            'total_times': deque(maxlen=30),
            'frame_skips': 0
        }
        
        print("树莓派摄像头二维码检测器已初始化")
        print(f"分辨率: {resolution[0]}x{resolution[1]}")
        print(f"FPS限制: {fps_limit}")
        print(f"预处理: {'已启用' if enable_preprocessing else '已禁用'}")
        print(f"图像保存: {'已启用' if save_images else '已禁用'}")
        print(f"YOLO置信度阈值: {yolo_confidence}")
        if save_images:
            print(f"结果保存目录: {self.results_dir}")
        print("按 Ctrl+C 或 'q' 键退出程序")
        
        # 注册信号处理
        signal.signal(signal.SIGINT, self.signal_handler)

    def preprocess_image(self, image):
        """优化的图像预处理方法，使用缓存和更高效的算法"""
        current_time = time.time()
        
        # 使用缓存避免重复处理
        if (self.preprocessed_cache is not None and 
            current_time - self.cache_timestamp < 0.05):  # 50ms缓存
            return self.preprocessed_cache
        
        # 直接使用灰度图，避免不必要的颜色转换
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 使用自适应阈值化，效果更好但计算量适中
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # 更新缓存
        self.preprocessed_cache = thresh
        self.cache_timestamp = current_time
        
        return thresh

    def detect_qr_codes(self, image):
        """高度优化的二维码识别方法，使用多级检测策略"""
        results = []
        
        # 第一级：直接检测原始图像
        try:
            qr_codes = pyzbar.decode(image)
            if qr_codes:
                for qr_code in qr_codes:
                    try:
                        data = qr_code.data.decode("utf-8")
                        results.append({
                            'data': data,
                            'rect': qr_code.rect,
                            'points': np.array(qr_code.polygon, dtype=np.int32) if qr_code.polygon else None
                        })
                    except UnicodeDecodeError:
                        continue
                return results
        except Exception:
            pass
        
        # 第二级：如果启用预处理且第一级失败，使用预处理图像
        if self.enable_preprocessing and not results:
            try:
                processed = self.preprocess_image(image)
                qr_codes = pyzbar.decode(processed)
                if qr_codes:
                    for qr_code in qr_codes:
                        try:
                            data = qr_code.data.decode("utf-8")
                            results.append({
                                'data': data,
                                'rect': qr_code.rect,
                                'points': np.array(qr_code.polygon, dtype=np.int32) if qr_code.polygon else None
                            })
                        except UnicodeDecodeError:
                            continue
            except Exception:
                pass
        
        return results

    def capture_frames(self):
        """独立的图像捕获线程"""
        while not should_exit:
            try:
                frame = self.picam2.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                with self.processing_lock:
                    if len(self.frame_queue) < self.frame_queue.maxlen:
                        self.frame_queue.append(frame)
            except Exception as e:
                print(f"捕获帧时出错: {e}")
                time.sleep(0.01)

    def process_frames(self):
        """独立的图像处理线程"""
        while not should_exit:
            frame = None
            with self.processing_lock:
                if self.frame_queue:
                    frame = self.frame_queue.popleft()
            
            if frame is not None:
                # 处理帧的逻辑将在这里实现
                pass
            else:
                time.sleep(0.001)

    def draw_detections(self, frame, yolo_results, qr_results):
        """高度优化的绘制函数，最小化计算和渲染开销"""
        current_time = time.time()
        
        # 批量绘制YOLO检测框 - 减少函数调用
        if yolo_results:
            for result in yolo_results:
                boxes = result.boxes.xyxy.cpu().numpy()
                # 只绘制前3个最高置信度的框以提高性能
                for i, box in enumerate(boxes[:3]):
                    x1, y1, x2, y2 = map(int, box)
                    # 使用更细的线条和更简单的颜色
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        # 优化二维码绘制 - 减少重复计算
        if qr_results:
            for qr in qr_results:
                rect = qr['rect']
                # 绘制简化的边界框
                cv2.rectangle(frame, 
                             (rect.left, rect.top), 
                             (rect.left + rect.width, rect.top + rect.height), 
                             (255, 0, 0), 1)
                
                # 优化文本显示逻辑
                if qr['data'] != self.last_qr_data or current_time - self.last_qr_time > self.detection_cooldown:
                    self.last_qr_data = qr['data']
                    self.last_qr_time = current_time
                    print(f"识别到二维码: {qr['data']}")
                    
                    # 异步保存图像以提高性能
                    if self.save_images and (current_time - self.last_save_time > self.image_save_interval):
                        self.last_save_time = current_time
                        # 在后台线程中保存图像
                        threading.Thread(target=self._save_image_async, 
                                       args=(frame.copy(),), daemon=True).start()
                
                # 简化文本显示 - 只在必要时绘制
                if rect.top > 20:
                    text_pos = (rect.left, rect.top - 5)
                else:
                    text_pos = (rect.left, rect.top + rect.height + 20)
                    
                # 限制文本长度并使用更小的字体
                display_text = qr['data'][:15] + ('...' if len(qr['data']) > 15 else '')
                cv2.putText(frame, display_text, text_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    def _save_image_async(self, frame):
        """异步保存图像，避免阻塞主线程"""
        try:
            timestamp = int(time.time())
            img_filename = f"qr_detected_{timestamp}.jpg"
            img_path = self.results_dir / img_filename
            cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            print(f"已保存图像: {img_path}")
        except Exception as e:
            print(f"保存图像失败: {e}")

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

    def signal_handler(self, sig, frame):
        """捕获Ctrl+C信号，优雅退出程序"""
        print("\n用户请求退出，正在关闭摄像头...")
        # 打印最终性能统计
        self.print_performance_stats()
        # 直接设置一个退出标志，而不是在信号处理程序中调用close方法
        # 这样可以确保在主线程中关闭资源
        global should_exit
        should_exit = True
        
    def close(self):
        """关闭摄像头和释放资源"""
        if self.picam2 is not None:
            self.picam2.stop_preview()
            self.picam2.stop()
        cv2.destroyAllWindows()
        print("摄像头已关闭")
        
    def run(self):
        """高度优化的实时检测主循环 - 树莓派5专用版本"""
        print("开始树莓派摄像头实时检测")
        print("提示：已启用树莓派5性能优化模式")
        
        try:
            # 重置全局退出标志
            global should_exit
            should_exit = False
            
            # 初始化性能统计
            self.frame_count = 0
            self.start_time = time.time()
            
            # 平衡性能和质量的参数
            yolo_interval = 5  # 每5帧运行一次YOLO检测
            qr_interval = 3     # 每3帧运行一次二维码检测
            display_interval = 1  # 每帧都显示
            
            # 预分配数组以减少内存分配
            frame_buffer = None
            
            while not should_exit:
                frame_start_time = time.time()
                
                # 控制帧率
                current_time = time.time()
                if current_time - self.last_frame_time < self.frame_interval:
                    remaining_time = self.frame_interval - (current_time - self.last_frame_time)
                    if remaining_time > 0.001:
                        time.sleep(remaining_time * 0.8)  # 减少休眠时间
                        self.performance_stats['frame_skips'] += 1
                    continue
                self.last_frame_time = current_time
                
                # 获取图像 - 使用更高效的方法
                frame = self.picam2.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                yolo_results = []
                qr_results = []
                
                # 使用缓存的YOLO结果
                if (current_time - self.yolo_cache_time < self.yolo_cache_duration and 
                    self.last_yolo_result is not None):
                    yolo_results = self.last_yolo_result
                elif self.frame_count % yolo_interval == 0:
                    # 使用合适的输入尺寸以保持检测质量
                    yolo_start = time.time()
                    # 先将图像缩小到合适的尺寸
                    small_frame_for_yolo = cv2.resize(frame, (416, 416))
                    yolo_results = self.model(small_frame_for_yolo, verbose=False, 
                                            conf=self.yolo_confidence, 
                                            imgsz=416,  # 适中的输入尺寸保持识别率
                                            device='cpu')  # 强制使用CPU
                    yolo_time = time.time() - yolo_start
                    self.performance_stats['yolo_times'].append(yolo_time)
                    
                    # 更新缓存
                    self.last_yolo_result = yolo_results
                    self.yolo_cache_time = current_time
                
                # 智能二维码检测 - 优先在YOLO检测区域，减少频率
                if yolo_results and self.frame_count % qr_interval == 0:
                    qr_start = time.time()
                    # 只处理第一个检测框
                    for result in yolo_results:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        if len(boxes) > 0:
                            # 只处理第一个框
                            box = boxes[0]
                            x1, y1, x2, y2 = map(int, box)
                            # 确保坐标在图像范围内
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                            
                            if x2 > x1 and y2 > y1:
                                roi = frame[y1:y2, x1:x2]
                                # 只检测一次，不进行预处理
                                try:
                                    roi_qr_codes = pyzbar.decode(roi)
                                    if roi_qr_codes:
                                        for qr_code in roi_qr_codes:
                                            try:
                                                data = qr_code.data.decode("utf-8")
                                                qr_results.append({
                                                    'data': data,
                                                    'rect': type('obj', (object,), {
                                                        'left': qr_code.rect.left + x1,
                                                        'top': qr_code.rect.top + y1,
                                                        'width': qr_code.rect.width,
                                                        'height': qr_code.rect.height
                                                    })
                                                })
                                            except UnicodeDecodeError:
                                                continue
                                except Exception:
                                    pass
                        break  # 只处理第一个结果
                    qr_time = time.time() - qr_start
                    self.performance_stats['qr_times'].append(qr_time)
                
                # 如果YOLO区域没有检测到，进行全图检测
                if not qr_results and self.frame_count % (qr_interval * 2) == 0:
                    qr_start = time.time()
                    # 使用适中的图像尺寸进行检测
                    scale_factor = 0.6  # 保持较好清晰度
                    new_width = int(frame.shape[1] * scale_factor)
                    new_height = int(frame.shape[0] * scale_factor)
                    small_frame = cv2.resize(frame, (new_width, new_height))
                    
                    try:
                        small_qr_codes = pyzbar.decode(small_frame)
                        if small_qr_codes:
                            for qr_code in small_qr_codes:
                                try:
                                    data = qr_code.data.decode("utf-8")
                                    # 调整坐标
                                    scale_x = frame.shape[1] / new_width
                                    scale_y = frame.shape[0] / new_height
                                    qr_results.append({
                                        'data': data,
                                        'rect': type('obj', (object,), {
                                            'left': int(qr_code.rect.left * scale_x),
                                            'top': int(qr_code.rect.top * scale_y),
                                            'width': int(qr_code.rect.width * scale_x),
                                            'height': int(qr_code.rect.height * scale_y)
                                        })
                                    })
                                except UnicodeDecodeError:
                                    continue
                    except Exception:
                        pass
                    qr_time = time.time() - qr_start
                    self.performance_stats['qr_times'].append(qr_time)
                
                # 简化绘制 - 减少性能开销
                if self.frame_count % display_interval == 0:
                    # 只绘制YOLO框，简化二维码绘制
                    if yolo_results:
                        for result in yolo_results:
                            boxes = result.boxes.xyxy.cpu().numpy()
                            for box in boxes[:2]:  # 只绘制前2个框
                                x1, y1, x2, y2 = map(int, box)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    
                    # 只绘制二维码文本，不画框
                    if qr_results:
                        for qr in qr_results[:1]:  # 只显示第一个二维码
                            rect = qr['rect']
                            print(f"QR: {qr['data'][:30]}")
                            # 只显示文本
                            cv2.putText(frame, "QR Detected", (5, frame.shape[0] - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                    
                    # 优化的FPS显示
                    self.frame_count += 1
                    elapsed_time = time.time() - self.start_time
                    if elapsed_time > 0:
                        avg_fps = self.frame_count / elapsed_time
                        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (5, 15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    
                    # 显示图像
                    cv2.imshow('QR Detector', frame)
                else:
                    self.frame_count += 1
                
                # 记录总处理时间
                total_time = time.time() - frame_start_time
                self.performance_stats['total_times'].append(total_time)
                
                # 每100帧打印一次性能统计
                if self.frame_count % 100 == 0:
                    self.print_performance_stats()
                
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
    # 树莓派5平衡质量和性能的配置
    resolution = (640, 480)  # 适中的分辨率保持清晰度
    fps_limit = 25          # 保持良好帧率
    
    # 创建并运行检测器
    detector = RaspberryPiQRDetector(
        model_path='./models/best.pt',      # YOLO模型路径
        resolution=resolution,              # 优化的分辨率
        fps_limit=fps_limit,                # 平衡的FPS限制
        enable_preprocessing=False,         # 禁用预处理以提升性能
        save_images=False,                  # 默认禁用图像保存以提高性能
        yolo_confidence=0.4,                # 适中的置信度保持识别率
        image_save_interval=3.0             # 图像保存间隔
    )
    detector.run()