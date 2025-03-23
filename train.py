import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from model import LeNet
from dataset import data_loader
from utils import Metric
import argparse
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf, DictConfig
import logging
from hydra.core.hydra_config import HydraConfig
import os


logging = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(config_name="default", config_path="configs")
def train(cfg: DictConfig):
    # 获取当前Hydra输出目录
    hydra_output_dir = HydraConfig.get().runtime.output_dir
    # 在Hydra输出目录下创建tensorboard子目录
    log_dir = os.path.join(hydra_output_dir, "runs")

    logging.info(OmegaConf.to_yaml(cfg))
    model = LeNet(num_classes=cfg.model.num_classes, hidden_dim_1=cfg.model.hidden_dim_1,
                  hidden_dim_2=cfg.model.hidden_dim_2, conv_hidden_dim_1=cfg.model.conv_hidden_dim_1,
                  conv_hidden_dim_2=cfg.model.conv_hidden_dim_2).to(device)
    # model = LeNet(num_classes=args.num_classes)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=cfg.optim.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir=log_dir)
    # 训练循环
    for epoch in range(cfg.trainer.epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # 遍历训练数据
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{cfg.trainer.epochs} [Train]'):
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
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        # 使用Metric计算训练准确度
        train_accuracy = Metric(correct_train, total_train).accuracy()
        train_loss = running_loss / len(train_loader)

        # 分别记录损失和准确度
        writer.add_scalar('Loss/Train Loss', train_loss, epoch)
        writer.add_scalar('Accuracy/Train Accuracy', train_accuracy, epoch)

        print(f'Epoch [{epoch + 1}/{cfg.trainer.epochs}], Loss: {train_loss:.4f}, '
              f'Train Accuracy: {train_accuracy:.4f}')

        # 验证模型
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():   # 在验证时不计算梯度
            for images, labels in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{cfg.trainer.epochs} [Val]'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                val_running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()

        # 使用Metric计算验证准确度
        val_accuracy = Metric(correct_val, total_val).accuracy()
        val_loss = val_running_loss / len(val_loader)

        # 分别记录验证损失和准确度
        writer.add_scalar('Loss/Validation Loss', val_loss, epoch)
        writer.add_scalar('Accuracy/Validation Accuracy', val_accuracy, epoch)

        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        # 测试模型
        model.eval()
        correct_test = 0
        total_test = 0
        test_running_loss = 0.0

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f'Epoch {epoch + 1}/{cfg.trainer.epochs} [Test]'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                test_running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += predicted.eq(labels).sum().item()

        # 使用Metric计算测试准确度
        test_accuracy = Metric(correct_test, total_test).accuracy()
        test_loss = test_running_loss / len(test_loader)

        # 分别记录测试损失和准确度
        writer.add_scalar('Loss/Test Loss', test_loss, epoch)
        writer.add_scalar('Accuracy/Test Accuracy', test_accuracy, epoch)

        # 记录权重的分布
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)

        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    # 关闭TensorBoard writer
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--root", type=str, default="./data")
    args = parser.parse_args()
    print(args)
    train_loader, val_loader, test_loader = data_loader(root=args.root, batch_size=args.batch_size)
    train()
