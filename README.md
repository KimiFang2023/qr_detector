# 二维码检测器使用指南

这个工具可以帮助你使用电脑或树莓派的摄像头实时检测和识别二维码！

## 功能特点

- 📷 使用YOLO模型快速定位图像中的二维码
- 🔍 结合传统图像处理方法（CLAHE、阈值化等）提高二维码识别率
- 💻 支持实时显示检测结果和FPS
- 💾 自动保存识别到二维码的图像
- ⚡ 针对树莓派进行了性能优化
- 🎯 支持光照自适应处理，适应不同环境条件

## 安装要求

### 通用依赖（电脑和树莓派都需要）

```bash
pip install opencv-python numpy ultralytics pyzbar
```

### 树莓派额外依赖

在树莓派上，需要安装以下系统依赖：

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
```

## 设备准备

### 电脑端
- 带有摄像头的电脑（内置或外接USB摄像头）
- 已安装Python 3.x

### 树莓派端
- 树莓派5（或其他型号）
- 官方摄像头模块rev1.3（连接到1号CSI接口）
- 已安装Raspberry Pi OS的SD卡
- 键盘、鼠标和显示器（或通过SSH远程连接）

## 摄像头连接（树莓派）

1. 确保树莓派已断电
2. 将官方摄像头rev1.3连接到树莓派5的1号CSI接口（通常位于USB-C电源接口旁边）
3. 确保摄像头连接稳固
4. 接通树莓派电源

## 启用摄像头（树莓派）

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

## 项目设置

### 电脑端

1. 确保项目目录结构如下：
   ```
   qr_detector/
   ├── camera_qr_detector.py
   ├── models/
   │   └── best.pt
   └── process/
       └── camera_results/
   ```

2. 创建结果保存目录：
   ```bash
   mkdir -p ./process/camera_results
   ```

### 树莓派端

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

## 如何使用

### 电脑端使用

1. 确保你的电脑已经连接了摄像头
2. 确保项目中已有训练好的YOLO模型（默认路径：`./models/best.pt`）
3. 直接运行以下命令：

```bash
python camera_qr_detector.py
```

### 树莓派端使用

1. 在树莓派上，打开终端并导航到项目目录：
   ```bash
   cd ~/qr_detector
   ```

2. 运行二维码检测程序：
   ```bash
   python3 raspberry_pi_camera_qr_detector.py
   ```

3. 程序启动后，您将看到摄像头捕获的画面，并能够实时检测二维码

### 退出程序

在程序运行时，按下键盘上的 `q` 键即可退出。

## 结果保存

- **电脑端**：程序会自动将识别到二维码的图像保存在 `./process/camera_results/` 文件夹中
- **树莓派端**：程序会自动将识别到二维码的图像保存在 `~/QR_Code_Results/` 目录中

## 性能优化说明

为了在树莓派上流畅运行，这个工具做了以下优化：

1. 限制了FPS（默认10-15帧/秒）
2. 降低了摄像头分辨率（默认640x480）
3. 采用了轻量级的图像处理方法
4. 避免重复识别相同的二维码
5. 光照自适应处理，根据环境自动调整参数

## 程序参数说明

您可以根据需要修改代码中的以下参数：

- `model_path`: YOLO模型文件路径，默认为`./models/best.pt`
- `camera_id`: 摄像头设备ID或路径
  - 电脑端：默认为`0`（第一个摄像头）
  - 树莓派：默认为`/dev/video0`
- `resolution`: 摄像头分辨率，默认为`(640, 480)`
- `fps_limit`: FPS限制，默认为`10`（树莓派）或`15`（电脑）
- `enable_preprocessing`: 是否启用图像预处理，默认为`True`

## 常见问题解决

### 问题1：无法启动摄像头

- 确保摄像头已正确连接
- 检查是否有其他程序正在使用摄像头
- 尝试修改代码中的 `camera_id` 参数
- **树莓派**：检查摄像头设备是否存在：
  ```bash
  ls -l /dev/video*
  ```
  如果看到`/dev/video0`或`/dev/video1`，说明摄像头已被识别

### 问题2：识别率不高

- 确保二维码清晰可见，避免模糊或严重倾斜
- 确保光线充足
- 可以尝试调整代码中的预处理参数
- 确保使用的是适合二维码检测的YOLO模型

### 问题3：运行卡顿

- 可以尝试进一步降低分辨率，例如 `resolution=(320, 240)`
- 可以降低FPS限制，例如 `fps_limit=5`
- 可以将 `enable_preprocessing` 设置为 `False` 来关闭预处理

### 问题4：图像显示乱码（树莓派）

- 程序已使用英文窗口标题以避免乱码问题

## 高级优化建议（树莓派）

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

## 代码结构说明

这个工具的代码主要包含以下几个部分：

1. **初始化**：设置摄像头参数、加载YOLO模型
2. **光照评估**：分析环境光照条件（低光、高光、不均匀光照等）
3. **图像预处理**：根据光照条件自适应地优化图像质量
4. **二维码检测**：使用YOLO模型定位二维码
5. **二维码识别**：使用pyzbar库识别二维码内容
6. **结果绘制**：在图像上显示检测框和识别结果
7. **主循环**：持续获取摄像头图像并进行处理

## 教育与学习建议

如果您想进一步了解计算机视觉和二维码识别，可以尝试：

1. 学习Python编程语言基础
2. 了解OpenCV库的基本使用方法
3. 学习简单的图像处理概念（如图像滤波、阈值化、CLAHE等）
4. 了解二维码的基本原理
5. 尝试修改代码中的参数，观察不同参数对识别效果的影响
6. 学习如何训练自己的YOLO模型以提高特定场景下的二维码检测率
7. 探索如何将识别结果通过网络发送到其他设备
8. 研究如何在低功耗模式下运行此程序以延长树莓派的电池续航（如果使用电池供电）

希望这个工具能帮助你探索计算机视觉的奇妙世界！
