import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from pyzbar.pyzbar import decode
import matplotlib.pyplot as plt

# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

class QRCodeDiffusionModel(nn.Module):
    """
    基于扩散模型的二维码修复模型
    实现了一个简化版的条件扩散模型，专门用于二维码修复任务
    """
    def __init__(self, img_size=256, channels=1, hidden_dim=64):
        super(QRCodeDiffusionModel, self).__init__()
        self.img_size = img_size
        self.channels = channels
        
        # 扩散模型参数
        self.num_timesteps = 1000
        self.beta_start = 0.0001
        self.beta_end = 0.02
        
        # 预计算扩散模型的参数
        self.beta = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        self.alpha = 1. - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        
        # 下采样网络（U-Net架构）
        self.down1 = self._create_block(channels, hidden_dim)
        self.down2 = self._create_block(hidden_dim, hidden_dim * 2)
        self.down3 = self._create_block(hidden_dim * 2, hidden_dim * 4)
        self.down4 = self._create_block(hidden_dim * 4, hidden_dim * 8)
        
        # 上采样网络 - 注意输入通道数需要考虑跳跃连接
        self.up1 = self._create_block(hidden_dim * 12, hidden_dim * 4)  # 512 + 256 = 768，调整为更合理的12*64=768
        self.up2 = self._create_block(hidden_dim * 6, hidden_dim * 2)   # 256 + 128 = 384，调整为6*64=384
        self.up3 = self._create_block(hidden_dim * 3, hidden_dim)       # 128 + 64 = 192，调整为3*64=192
        self.up4 = self._create_block(hidden_dim, channels)
        
        # 输出层
        self.output = nn.Sigmoid()
    
    def _create_block(self, in_channels, out_channels):
        """创建网络的基本卷积块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, t):
        """前向传播"""
        # 简化实现：移除了复杂的时间嵌入，专注于基本的U-Net架构
        # 下采样
        d1 = self.down1(x)
        d1_pool = F.max_pool2d(d1, 2)
        
        d2 = self.down2(d1_pool)
        d2_pool = F.max_pool2d(d2, 2)
        
        d3 = self.down3(d2_pool)
        d3_pool = F.max_pool2d(d3, 2)
        
        d4 = self.down4(d3_pool)
        
        # 上采样
        u1 = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True)
        u1 = torch.cat([u1, d3], dim=1)
        u1 = self.up1(u1)
        
        u2 = F.interpolate(u1, scale_factor=2, mode='bilinear', align_corners=True)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.up2(u2)
        
        u3 = F.interpolate(u2, scale_factor=2, mode='bilinear', align_corners=True)
        u3 = torch.cat([u3, d1], dim=1)
        u3 = self.up3(u3)
        
        # 确保输出尺寸与输入尺寸匹配
        if u3.shape[2:] != x.shape[2:]:
            u3 = F.interpolate(u3, size=x.shape[2:], mode='bilinear', align_corners=True)
            
        u4 = self.up4(u3)
        
        # 再次确保最终输出尺寸与输入尺寸匹配
        if u4.shape[2:] != x.shape[2:]:
            u4 = F.interpolate(u4, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        # 输出
        return self.output(u4)
        
    def add_noise(self, x_start, t):
        """向图像添加噪声"""
        # 确保alpha_cumprod在正确的设备上
        alpha_cumprod = self.alpha_cumprod.to(x_start.device)
        
        noise = torch.randn_like(x_start)
        sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod[t])
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1. - alpha_cumprod[t])
        
        x_noisy = sqrt_alpha_cumprod_t.view(-1, 1, 1, 1) * x_start + sqrt_one_minus_alpha_cumprod_t.view(-1, 1, 1, 1) * noise
        return x_noisy, noise

class QRCodeRepair:
    """
    二维码修复工具类
    提供基于扩散模型的二维码修复功能
    """
    def __init__(self, model_path=None):
        # 检查是否有可用的GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 初始化模型
        self.model = QRCodeDiffusionModel().to(self.device)
        
        # 如果提供了预训练模型路径，加载模型
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"成功加载预训练模型: {model_path}")
            except Exception as e:
                print(f"加载模型失败: {e}")
        
        # 扩散模型参数
        self.num_timesteps = 1000
        self.beta_start = 0.0001
        self.beta_end = 0.02
        
        # 预计算扩散模型的参数
        self.beta = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps).to(self.device)
        self.alpha = 1. - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        
        # 确保输出目录存在
        Path("./process/output_restored").mkdir(parents=True, exist_ok=True)
    
    def add_noise(self, x_start, t):
        """向图像添加噪声"""
        # 确保alpha_cumprod在正确的设备上
        alpha_cumprod = self.alpha_cumprod.to(x_start.device)
        
        noise = torch.randn_like(x_start)
        sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod[t])
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1. - alpha_cumprod[t])
        
        x_noisy = sqrt_alpha_cumprod_t.view(-1, 1, 1, 1) * x_start + sqrt_one_minus_alpha_cumprod_t.view(-1, 1, 1, 1) * noise
        return x_noisy, noise
    
    def inpaint(self, img, mask, num_inference_steps=50):
        """
        修复二维码图像
        img: 输入图像 (numpy数组，灰度图)
        mask: 掩码图像，0表示需要修复的区域，1表示保留的区域
        num_inference_steps: 推理步数
        """
        # 预处理图像
        img = cv2.resize(img, (256, 256))
        img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float() / 255.0
        
        mask = cv2.resize(mask, (256, 256))
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
        
        # 移动到设备
        img_tensor = img_tensor.to(self.device)
        mask_tensor = mask_tensor.to(self.device)
        
        # 初始化噪声图像
        x = torch.randn_like(img_tensor).to(self.device)
        
        # 采样过程
        self.model.eval()
        with torch.no_grad():
            step_size = self.num_timesteps // num_inference_steps
            for i in tqdm(range(0, self.num_timesteps, step_size)):
                t = self.num_timesteps - i - 1
                t_tensor = torch.tensor([t], dtype=torch.long).to(self.device)
                
                # 预测噪声
                predicted_noise = self.model(x, t_tensor.float() / self.num_timesteps)
                
                # 确保predicted_noise与x尺寸相匹配
                if predicted_noise.shape != x.shape:
                    predicted_noise = F.interpolate(predicted_noise, size=x.shape[2:], mode='bilinear', align_corners=True)
                
                # 应用掩码，只修复需要的区域
                x = x * mask_tensor + predicted_noise * (1 - mask_tensor)
                
                # 应用扩散步骤更新
                if t > 0:
                    beta_t = self.beta[t]
                    alpha_t = self.alpha[t]
                    alpha_cumprod_t = self.alpha_cumprod[t]
                    
                    noise = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
                    x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise) + torch.sqrt(beta_t) * noise
        
        # 后处理
        restored_img = x.squeeze().cpu().numpy()
        
        # 检查输出是否异常
        if restored_img.min() == restored_img.max() or np.isnan(restored_img).any():
            print("扩散模型输出异常，返回原图")
            return img
            
        # 归一化到0-255
        restored_img = np.clip(restored_img * 255, 0, 255).astype(np.uint8)
        restored_img = cv2.resize(restored_img, (img.shape[1], img.shape[0]))
        
        return restored_img
    
    def auto_detect_and_repair(self, img_path, output_path=None):
        """
        自动检测二维码并进行修复
        """
        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像: {img_path}")
            return None
        
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 尝试解码原始二维码
        original_results = decode(gray)
        original_decoded = len(original_results) > 0
        
        # 创建掩码 - 自动检测损坏区域
        # 这里使用简化的方法，实际应用中可以根据具体情况改进
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 计算图像梯度，检测模糊或低对比度区域
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_mean = np.mean(gradient_magnitude)
        
        # 创建掩码：低梯度区域认为是损坏区域
        mask = np.ones_like(gray, dtype=np.uint8)
        mask[gradient_magnitude < gradient_mean * 0.5] = 0
        
        # 修复图像
        restored_img = self.inpaint(gray, mask)
        
        # 尝试解码修复后的二维码
        restored_results = decode(restored_img)
        restored_decoded = len(restored_results) > 0
        
        # 如果没有提供输出路径，自动生成
        if output_path is None:
            base_name = Path(img_path).stem
            ext = Path(img_path).suffix
            output_path = f"./process/output_restored/{base_name}_restored{ext}"
        
        # 保存修复后的图像
        cv2.imwrite(output_path, restored_img)
        
        # 输出修复结果信息
        print(f"原始二维码解码: {'成功' if original_decoded else '失败'}")
        print(f"修复后二维码解码: {'成功' if restored_decoded else '失败'}")
        
        if original_decoded:
            print(f"原始二维码内容: {original_results[0].data.decode('utf-8')}")
        
        if restored_decoded:
            print(f"修复后二维码内容: {restored_results[0].data.decode('utf-8')}")
        
        print(f"修复后的图像已保存至: {output_path}")
        
        return restored_img
    
    def process_directory(self, input_dir, output_dir=None):
        """
        批量处理文件夹中的所有二维码图像
        """
        if output_dir is None:
            output_dir = "./process/output_restored"
        
        # 确保输出目录存在
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 获取所有图片文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [f for f in os.listdir(input_dir)
                      if os.path.isfile(os.path.join(input_dir, f))
                      and Path(f).suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"在 {input_dir} 中未找到任何图片文件")
            return
        
        # 处理每张图片
        success_count = 0
        for img_file in tqdm(image_files):
            img_path = os.path.join(input_dir, img_file)
            output_path = os.path.join(output_dir, img_file)
            
            print(f"处理: {img_path}")
            restored_img = self.auto_detect_and_repair(img_path, output_path)
            
            if restored_img is not None:
                success_count += 1
            
            print("="*50)
        
        print(f"所有图片处理完成！共处理 {len(image_files)} 张图片，成功修复 {success_count} 张")
        print(f"修复成功率: {success_count / len(image_files) * 100:.2f}%")

import torch.utils.data as data
import albumentations as A
from albumentations.pytorch import ToTensorV2

class QRCodeDataset(data.Dataset):
    """
    二维码修复数据集
    期望的数据结构：
    - data_dir/
        - damaged/      # 损坏的二维码图像
        - original/     # 对应的原始完好二维码图像
    """
    def __init__(self, data_dir, img_size=256, transform=None):
        self.data_dir = data_dir
        self.img_size = img_size
        
        # 默认数据增强和转换
        self.transform = transform if transform else A.Compose([
            A.Resize(img_size, img_size),
            A.ToGray(num_output_channels=1),
            A.Normalize(mean=0.5, std=0.5),
            ToTensorV2(),
        ])
        
        # 加载文件列表
        self.damaged_dir = os.path.join(data_dir, 'damaged')
        self.original_dir = os.path.join(data_dir, 'original')
        
        # 获取图片文件列表并排序以确保对应关系
        self.image_files = [f for f in os.listdir(self.damaged_dir)
                          if os.path.isfile(os.path.join(self.damaged_dir, f))
                          and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        # 验证对应的原始图像是否存在
        valid_files = []
        for f in self.image_files:
            if os.path.exists(os.path.join(self.original_dir, f)):
                valid_files.append(f)
        
        self.image_files = valid_files
        print(f"加载了 {len(self.image_files)} 对训练数据")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 加载损坏和原始图像
        img_name = self.image_files[idx]
        
        # 读取损坏图像
        damaged_path = os.path.join(self.damaged_dir, img_name)
        damaged_img = cv2.imread(damaged_path)
        if damaged_img is None:
            # 如果图像读取失败，返回一个默认的灰度图像
            damaged_img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            damaged_img = cv2.cvtColor(damaged_img, cv2.COLOR_BGR2RGB)
        
        # 读取原始图像
        original_path = os.path.join(self.original_dir, img_name)
        original_img = cv2.imread(original_path)
        if original_img is None:
            # 如果图像读取失败，返回一个默认的灰度图像
            original_img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # 应用变换
        augmented = self.transform(image=damaged_img)
        damaged_tensor = augmented['image']
        
        # 对原始图像单独应用相同的变换
        augmented_original = self.transform(image=original_img)
        original_tensor = augmented_original['image']
        
        # 确保两个张量都是1通道的灰度图
        if damaged_tensor.shape[0] == 3:
            damaged_tensor = damaged_tensor.mean(dim=0, keepdim=True)
        if original_tensor.shape[0] == 3:
            original_tensor = original_tensor.mean(dim=0, keepdim=True)
        
        return damaged_tensor, original_tensor

# 训练扩散模型
def train_model(data_dir, epochs=50, batch_size=16, save_path="./models/qr_diffusion_model.pth"):
    """
    训练二维码修复扩散模型
    
    参数:
    - data_dir: 数据集目录，包含damaged和original子目录
    - epochs: 训练轮数
    - batch_size: 批量大小
    - save_path: 模型保存路径
    """
    # 检查数据目录结构
    if not os.path.exists(os.path.join(data_dir, 'damaged')) or not os.path.exists(os.path.join(data_dir, 'original')):
        print("错误: 数据目录结构不正确")
        print("请确保数据目录包含damaged和original两个子目录")
        print("建议的数据集准备方法:")
        print("1. 创建一个训练目录，如qr_training_data")
        print("2. 在其中创建damaged和original两个子目录")
        print("3. 将损坏的二维码图片放入damaged目录")
        print("4. 将对应的原始完好二维码图片放入original目录，保持文件名一致")
        return
    
    # 检查CUDA可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建数据集和数据加载器
    dataset = QRCodeDataset(data_dir)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    
    # 创建模型
    model = QRCodeDiffusionModel().to(device)
    
    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    
    # 创建模型保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 训练循环
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for damaged_imgs, original_imgs in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            # 移动数据到设备
            damaged_imgs = damaged_imgs.to(device)
            original_imgs = original_imgs.to(device)
            
            # 随机采样时间步
            t = torch.randint(0, model.num_timesteps, (damaged_imgs.shape[0],), device=device).long()
            
            # 添加噪声到原始图像
            noisy_imgs, noise = model.add_noise(original_imgs, t)
            
            # 混合损坏图像和噪声图像
            # 创建掩码（简单版本，实际应用中可以更复杂）
            _, mask = cv2.threshold(damaged_imgs.cpu().numpy().squeeze(), 127, 255, cv2.THRESH_BINARY)
            mask_tensor = torch.from_numpy(mask).unsqueeze(1).float().to(device) / 255.0
            
            # 混合图像：损坏区域使用噪声图像，完好区域保持原样
            input_imgs = damaged_imgs * mask_tensor + noisy_imgs * (1 - mask_tensor)
            
            # 预测噪声
            optimizer.zero_grad()
            predicted_noise = model(input_imgs, t.float() / model.num_timesteps)
            
            # 计算损失
            loss = criterion(predicted_noise, noise)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * damaged_imgs.size(0)
        
        # 计算平均损失
        avg_loss = epoch_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        # 保存最好的模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"模型已保存到 {save_path}，当前最佳损失: {best_loss:.6f}")
    
    print("训练完成！")
    print(f"最终模型已保存到 {save_path}")
    print("\n使用训练好的模型:")
    print("1. 可以直接在QRCodeRepair初始化时指定模型路径")
    print("   qr_repair = QRCodeRepair(model_path='./models/qr_diffusion_model.pth')")
    print("2. 或者将模型放在默认路径，程序会自动加载")

if __name__ == "__main__":
    # 创建修复工具实例
    qr_repair = QRCodeRepair()
    
    # 示例用法
    # 1. 修复单张图片
    # qr_repair.auto_detect_and_repair("./process/output_cropped/sample_qr.jpg")
    
    # 2. 批量处理文件夹中的图片
    # qr_repair.process_directory("./process/output_cropped")
    
    # 3. 训练模型（需要准备数据集）
    # train_model("./qr_dataset")
    
    print("二维码修复工具已初始化完成！\n使用方法：\n1. 修复单张图片: qr_repair.auto_detect_and_repair('图片路径')\n2. 批量处理: qr_repair.process_directory('输入文件夹路径')")