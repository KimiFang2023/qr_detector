from picamera2 import Picamera2, Preview
import time
import signal
import sys

def signal_handler(sig, frame):
    """捕获Ctrl+C信号，优雅退出程序"""
    print("\n用户请求退出，正在关闭摄像头...")
    # 停止预览和相机
    picam2.stop_preview()
    picam2.stop()
    print("摄像头已关闭，程序退出")
    sys.exit(0)

if __name__ == "__main__":
    try:
        # 创建摄像头对象
        picam2 = Picamera2()
        
        # 配置预览参数（自动适配rev1.3摄像头，使用默认分辨率）
        # 若需调整窗口大小，可添加main={"size": (宽度, 高度)}，例如(1280, 720)
        preview_config = picam2.create_preview_configuration()
        picam2.configure(preview_config)
        
        # 启动预览（树莓派5推荐使用QTGL硬件加速预览）
        # 窗口位置(x=100, y=100)，可根据需要调整
        picam2.start_preview(Preview.QTGL, x=100, y=100)
        
        # 启动摄像头
        picam2.start()
        print("摄像头启动成功，实时画面已显示")
        print("按 Ctrl+C 退出程序")
        
        # 注册信号捕获（处理Ctrl+C退出）
        signal.signal(signal.SIGINT, signal_handler)
        
        # 保持程序运行（持续显示画面）
        while True:
            time.sleep(1)  # 休眠1秒，减少CPU占用

    except Exception as e:
        # 捕获其他异常并提示
        print(f"发生错误：{str(e)}")
        if 'picam2' in locals():
            picam2.stop_preview()
            picam2.stop()
        sys.exit(1)