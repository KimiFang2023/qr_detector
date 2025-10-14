# 树莓派5二维码检测器使用指南

本指南将帮助您在树莓派5上使用官方摄像头模块rev1.3（连接到1号接口）运行二维码检测程序。

## 设备准备

- 树莓派5
- 官方摄像头模块rev1.3（连接到1号CSI接口）
- 已安装Raspberry Pi OS的SD卡
- 键盘、鼠标和显示器（或通过SSH远程连接）

## 摄像头连接

1. 确保树莓派已断电
2. 将官方摄像头rev1.3连接到树莓派5的1号CSI接口（通常位于USB-C电源接口旁边）
3. 确保摄像头连接稳固
4. 接通树莓派电源

## 启用摄像头

1. 启动树莓派并进入Raspberry Pi OS
2. 打开终端
3. 运行以下命令打开配置工具：
   ```
   sudo raspi-config
   ```
4. 使用方向键导航到 "Interface Options" > "Camera"
5. 选择 "Yes" 启用摄像头
6. 退出配置工具并重启树莓派：
   ```
   sudo reboot
   ```

## 安装依赖

重启后，打开终端并运行以下命令安装必要的软件包：

```bash
# 更新系统软件包
sudo apt update && sudo apt upgrade -y

# 安装Python和pip
sudo apt install python3 python3-pip -y

# 安装OpenCV依赖
sudo apt install python3-opencv libopencv-dev -y

# 安装pyzbar依赖
sudo apt install libzbar0 -y

# 安装NumPy
pip3 install numpy

# 安装YOLO库
pip3 install ultralytics

# 安装pyzbar库
pip3 install pyzbar

# 安装pathlib（通常Python3已包含）
pip3 install pathlib
```

## 项目设置

1. 创建项目目录：
   ```bash
   mkdir -p ~/qr_detector
   cd ~/qr_detector
   ```

2. 复制项目文件到树莓派：
   - 使用SFTP、SCP或其他方法将以下文件从电脑复制到树莓派的`~/qr_detector`目录：
     - `raspberry_pi_camera_qr_detector.py`
     - `models/best.pt`（YOLO模型文件）

   如果通过SSH，可以使用以下命令：
   ```bash
   # 从电脑端执行，假设文件在本地的Downloads目录
   scp ~/Downloads/raspberry_pi_camera_qr_detector.py pi@<树莓派IP>:~/qr_detector/
   scp -r ~/Downloads/models pi@<树莓派IP>:~/qr_detector/
   ```

3. 创建结果保存目录：
   ```bash
   mkdir -p ~/QR_Code_Results
   ```

## 运行程序

1. 在树莓派上，打开终端并导航到项目目录：
   ```bash
   cd ~/qr_detector
   ```

2. 运行二维码检测程序：
   ```bash
   python3 raspberry_pi_camera_qr_detector.py
   ```

3. 程序启动后，您将看到摄像头捕获的画面，并能够实时检测二维码

4. 要退出程序，按键盘上的 `q` 键

## 可能的问题及解决方案

### 问题1：无法打开摄像头

- 检查摄像头连接是否正确
- 确认摄像头已在`raspi-config`中启用
- 检查摄像头设备是否存在：
  ```bash
  ls -l /dev/video*
  ```
  如果看到`/dev/video0`或`/dev/video1`，说明摄像头已被识别
- 如果设备路径不是`/dev/video0`，需要修改代码中的`camera_id`参数

### 问题2：程序运行缓慢或卡顿

- 降低分辨率：在代码中修改`resolution`参数，例如改为`(320, 240)`
- 降低FPS限制：在代码中修改`fps_limit`参数，例如改为`5`
- 关闭图像预处理：将`enable_preprocessing`设置为`False`

### 问题3：二维码识别率不高

- 确保二维码清晰可见，避免模糊或严重倾斜
- 确保光线充足
- 可以尝试调整代码中的预处理参数
- 确保使用的是适合二维码检测的YOLO模型

### 问题4：图像显示乱码

- 本程序已使用英文窗口标题以避免乱码问题

## 程序参数说明

您可以根据需要修改`raspberry_pi_camera_qr_detector.py`文件末尾的以下参数：

- `model_path`: YOLO模型文件路径，默认为`./models/best.pt`
- `camera_id`: 摄像头设备路径，默认为`/dev/video0`（树莓派官方摄像头rev1.3 1号接口）
- `resolution`: 摄像头分辨率，默认为`(640, 480)`
- `fps_limit`: FPS限制，默认为`10`
- `enable_preprocessing`: 是否启用图像预处理，默认为`True`

## 高级优化建议

对于树莓派5，您还可以尝试以下优化措施以获得更好的性能：

1. **超频**：适当超频树莓派CPU可以提高处理速度，但请注意散热

2. **使用TensorFlow Lite或ONNX Runtime**：如果YOLO推理速度较慢，可以考虑使用这些轻量级推理引擎

3. **关闭不必要的服务**：关闭树莓派上不需要的后台服务以释放系统资源

4. **使用swap空间**：如果内存不足，可以增加swap空间
   ```bash
   sudo dphys-swapfile swapoff
   sudo nano /etc/dphys-swapfile
   # 将CONF_SWAPSIZE从默认值改为更大的值，例如512或1024
   sudo dphys-swapfile setup
   sudo dphys-swapfile swapon
   ```

## 结果查看

程序会自动将识别到二维码的图像保存在`~/QR_Code_Results/`目录中。您可以使用图像查看器打开这些图像，或通过SFTP将它们传输到电脑上查看。

## 教育与学习建议

如果您想进一步了解计算机视觉和二维码识别，可以尝试：

1. 学习如何训练自己的YOLO模型以提高特定场景下的二维码检测率
2. 尝试添加更多的图像预处理方法以适应不同的环境条件
3. 探索如何将识别结果通过网络发送到其他设备
4. 研究如何在低功耗模式下运行此程序以延长树莓派的电池续航（如果使用电池供电）

祝您在树莓派上愉快地探索计算机视觉世界！