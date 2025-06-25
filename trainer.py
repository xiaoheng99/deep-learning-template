import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
# from simple_model import ResNet, LeNet
from dataset import data_loader
from utils import Metric
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from hydra.core.hydra_config import HydraConfig
import os
import shutil
from pathlib import Path

logging = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["HYDRA_FULL_ERROR"] = "1"

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

    # model = LeNet(
    #     num_classes=cfg.model.num_classes, 
    #     hidden_dim_1=cfg.model.hidden_dim_1,
    #     hidden_dim_2=cfg.model.hidden_dim_2, 
    #     conv_hidden_dim_1=cfg.model.conv_hidden_dim_1,
    #     conv_hidden_dim_2=cfg.model.conv_hidden_dim_2
    # ).to(device)
    # model = ResNet18().to(device)

    model = hydra.utils.instantiate(cfg.model).to(device)
    
    logging.info(f"模型结构:\n{model}")
    print("\n")
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.trainer.optim.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

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
        correct_train = 0
        total_train = 0

        # 遍历训练数据
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # 将数据迁移到GPU

            # 前向传播
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 记录训练集的预测和真实标签
            _, predicted = torch.max(outputs, 1)  # 返回索引
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        # 计算训练准确度和损失
        train_accuracy = Metric(correct_train, total_train).accuracy()
        train_loss = running_loss / len(train_loader)

        # 记录到TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_accuracy, epoch)

        logging.info(f'Epoch [{epoch + 1}/{cfg.trainer.trainer.epochs}], '
                    f'Train Loss: {train_loss:.4f}, '
                    f'Train Accuracy: {train_accuracy:.4f}')

        # 验证模型
        val_accuracy, val_loss = evaluate(model, val_loader, loss_fn, device, "Validation")
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)

        # 保存最佳模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_ckpt(model, optimizer, epoch, os.path.join(ckpt_dir, "best.pt"))
            logging.info(f"新的最佳验证准确率: {best_val_accuracy:.4f}")
        
        # 保存最新模型
        if epoch % cfg.trainer.trainer.save_interval == 0:
            save_ckpt(model, optimizer, epoch, os.path.join(ckpt_dir, "last.pt"))
            logging.info(f"模型已保存到: {os.path.join(ckpt_dir, 'last.pt')}")

        # 记录权重和梯度的分布
        for name, param in model.named_parameters():
            writer.add_histogram(f'weights/{name}', param, epoch)
            if param.grad is not None:
                writer.add_histogram(f'gradients/{name}', param.grad, epoch)

    # 关闭TensorBoard writer
    writer.close()
    
    # 加载最佳模型进行最终测试
    if os.path.exists(os.path.join(ckpt_dir, "best.pt")):
        load_ckpt(model, optimizer, os.path.join(ckpt_dir, "best.pt"))
        test_accuracy, test_loss = evaluate(model, test_loader, loss_fn, device, "Test")
        logging.info(f'最终测试结果: Loss={test_loss:.4f}, Accuracy={test_accuracy:.4f}')
    else:
        logging.warning("未找到最佳模型，跳过最终测试")


def evaluate(model, data_loader, loss_fn, device, phase="Validation"):
    """评估模型性能"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = Metric(correct, total).accuracy()
    loss = running_loss / len(data_loader)
    
    logging.info(f'{phase} Loss: {loss:.4f}, {phase} Accuracy: {accuracy:.4f}')
    return accuracy, loss


if __name__ == "__main__":
    train()