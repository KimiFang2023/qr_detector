from pathlib import Path
import os
import random
import torch
from train_enhance import train_enhancer, set_seed


def continue_training(
    existing_model_path: str = "./models/enhance_unet.pth",
    new_model_save_path: str = "./models/enhance_unet_continued.pth",
    additional_epochs: int = 30,
    clean_dir: str = "./process/output_monochrome",
    img_size: int = 256,
    batch_size: int = 16,
    lr: float = 5e-5,  # 降低学习率，适合继续训练
    black_weight: float = 3.0,
    structure_weight: float = 0.5
) -> None:
    """
    在已训练的模型基础上继续训练。
    
    参数:
        existing_model_path: 已训练模型的路径
        new_model_save_path: 新模型保存路径
        additional_epochs: 继续训练的轮数
        clean_dir: 包含干净二维码图像的目录
        img_size: 输入图像大小
        batch_size: 训练批次大小
        lr: 学习率（通常低于初始训练的学习率）
        black_weight: 损失计算中黑色区域的权重因子
        structure_weight: 结构一致性损失的权重因子
    """
    # 设置随机种子以确保可重复性
    set_seed(42)
    
    # 检查现有模型是否存在
    if not os.path.exists(existing_model_path):
        print(f"错误: 找不到现有的模型文件: {existing_model_path}")
        print("请检查模型文件路径是否正确。")
        return
    
    # 确保保存目录存在
    Path(os.path.dirname(new_model_save_path)).mkdir(parents=True, exist_ok=True)
    
    print(f"正在加载现有模型: {existing_model_path}")
    print(f"将继续训练 {additional_epochs} 轮，并保存到: {new_model_save_path}")
    print(f"使用学习率: {lr}")
    
    # 调用训练函数，使用pretrained_path参数加载已有模型
    train_enhancer(
        clean_dir=clean_dir,
        save_path=new_model_save_path,
        img_size=img_size,
        epochs=additional_epochs,
        batch_size=batch_size,
        lr=lr,
        black_weight=black_weight,
        structure_weight=structure_weight,
        pretrained_path=existing_model_path  # 这是关键参数，指定要加载的已有模型
    )


if __name__ == "__main__":
    # 可以根据需要调整以下参数
    continue_training(
        existing_model_path="./models/enhance_unet.pth",  # 已有的模型路径
        new_model_save_path="./models/enhance_unet_continued.pth",  # 新模型保存路径
        additional_epochs=30,  # 继续训练的轮数
        lr=5e-5,  # 降低学习率，适合微调
        batch_size=16,  # 可根据GPU内存调整
        black_weight=3.0,  # 可以尝试增加这个值以更好地学习黑色模块
        structure_weight=0.5  # 结构一致性损失权重
    )