import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dataset import data_loader
from utils import Metric
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from hydra.core.hydra_config import HydraConfig
import os
import shutil
from pathlib import Path
import torchvision
import matplotlib.pyplot as plt


logging = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["HYDRA_FULL_ERROR"] = "1"

def vae_loss(x_recon, x, mu, logvar, loss_dtype="max"):
    if loss_dtype == "max":
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction=loss_dtype)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return (recon_loss + kl_div) / x.size(0), recon_loss / x.size(0), kl_div / x.size(0)  # batch 平均
    elif loss_dtype == "mean":
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction=loss_dtype)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_div = kl_div / x.numel()  # x.numel() 是总像素数
        return recon_loss + kl_div, recon_loss , kl_div 
    else:
        raise NotImplementedError

# 展示图像
def show_images(images, nrow=4):
    grid_img = torchvision.utils.make_grid(images, nrow=nrow)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis("off")
    plt.show()
    plt.pause(0.01)
    plt.close()

def save_generated_images(samples, epoch, ckpt_dir="save_img", nrow=4, prefix="vae_sample"):
    """
    保存生成图像到指定路径

    Args:
        samples (Tensor): 形状为 [B, C, H, W] 的图像张量
        epoch (int): 当前 epoch，用于命名文件
        ckpt_dir (str): 保存目录
        nrow (int): 每行显示的图像数量
        prefix (str): 文件名前缀
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    file_path = os.path.join(ckpt_dir, f"{prefix}_{epoch}_mean.png")
    torchvision.utils.save_image(samples, file_path, nrow=nrow, normalize=True)
    print(f"[INFO] 已保存图像到: {file_path}")


def save_ckpt(model, optimizer, epoch, path):
    """保存模型检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, path)
    logging.info(f"模型已保存到: {path}")


def load_ckpt(model, optimizer, path):
    """加载模型检查点"""
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        logging.info(f"已从 {path} 加载模型，继续从第 {start_epoch} 轮训练")
        return start_epoch
    else:
        logging.info("未找到检查点，从头开始训练")
        return 0


@hydra.main(version_base=None ,config_name="default", config_path="configs")
def train(cfg: DictConfig):
    # 获取当前Hydra输出目录
    hydra_output_dir = HydraConfig.get().runtime.output_dir
    # 在Hydra输出目录下创建tensorboard子目录
    log_dir = os.path.join(hydra_output_dir, "runs")
    # 创建模型保存目录
    logging.info(OmegaConf.to_yaml(cfg))
    
    ckpt_dir = os.path.join(hydra_output_dir, "checkpoints")
    if not os.path.exists(ckpt_dir):
        print(f"模型检查点将保存到: {ckpt_dir}")
        os.makedirs(ckpt_dir, exist_ok=True)

    # logging.info(f"配置已保存到: {config_path}")
    logging.info(f"TensorBoard日志将保存到: {log_dir}")
    logging.info(f"模型检查点将保存到: {ckpt_dir}")
    # logging.info("\n" + cfg.pretty())
     # 打印完整配置结构，帮助调试
    print(OmegaConf.to_yaml(cfg))
    
    # 检查data配置组是否包含root字段
    print(f"Data config keys: {list(cfg.data.keys())}")

    # 使用Hydra配置加载数据  (并不是写在main函数外面，写在里面就可以利用装饰器去调用)
    train_loader, val_loader, test_loader = data_loader(
        root=cfg.data.data.root, 
        batch_size=cfg.data.data.batch_size,
        num_workers=cfg.data.data.num_workers,
        pin_memory=cfg.data.data.pin_memory,
        split=cfg.data.data.split
    )

    model = hydra.utils.instantiate(cfg.model.model)
    print(model)
    print(type(model))
    model = model.to(device)

    logging.info(f"模型结构:\n{model}")
    print("\n")
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.trainer.optim.lr)
    # loss_fn = torch.nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir=log_dir)
    
    # 加载检查点（如果存在）
    start_epoch = 0
    if cfg.trainer.trainer.resume and os.path.exists(os.path.join(ckpt_dir, "last.ckpt")):
        start_epoch = load_ckpt(model, optimizer, os.path.join(ckpt_dir, "last.ckpt"))

    best_val_accuracy = 0.0
    
    # 训练循环
    for epoch in range(start_epoch, cfg.trainer.trainer.epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0

        # 遍历训练数据
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # 将数据迁移到GPU

            # 前向传播
            outputs, mu, logvar = model(images)
            loss, recon_loss, kl_div = vae_loss(outputs, images, mu, logvar, loss_dtype="mean")

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 计算训练准确度和损失
        train_loss = running_loss / len(train_loader)

        # 记录到TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Recon', recon_loss, epoch)
        writer.add_scalar('Loss/KL', kl_div, epoch)

        logging.info(f'Epoch [{epoch + 1}/{cfg.trainer.trainer.epochs}], '
                    f'Train Loss: {train_loss:.4f}, '
                    f'Recon Loss: {recon_loss:.4f}, '
                    f'KL Loss: {kl_div:.4f}'
                    )
        
        with torch.no_grad():
            z = torch.randn(16, model.encoder.fc_mu.out_features).to(device)
            samples = model.decoder(z).cpu()
            writer.add_images("Generated/VAE", samples, epoch)
            if epoch % 10 == 0:
                #show_images(samples, nrow=4)
                save_generated_images(samples, epoch)

        
        # 保存最新模型
        if epoch % cfg.trainer.trainer.save_interval == 0:
            save_ckpt(model, optimizer, epoch, os.path.join(ckpt_dir, "last.pt"))
            logging.info(f"模型已保存到: {os.path.join(ckpt_dir, 'last.pt')}")


    # 关闭TensorBoard writer
    writer.close()
    
    # 加载最佳模型进行最终测试
    if os.path.exists(os.path.join(ckpt_dir, "best.pt")):
        load_ckpt(model, optimizer, os.path.join(ckpt_dir, "best.pt"))
    else:
        logging.warning("未找到最佳模型，跳过最终测试")


if __name__ == "__main__":
    train()