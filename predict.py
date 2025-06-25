import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import os 
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from torchvision import datasets, transforms


# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None ,config_name="default", config_path="configs")
def predict(cfg: DictConfig):
    # 1.加载模型
    logging.info(f"加载模型: {cfg.model.__target__}")
    model = hydra.utils.instantiate(cfg.model)

    # 2.加载ckpt
    checkpoint_path = cfg.checkpoint_path
    if not os.path.exist(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")

    checkpoint = torch.load(cfg.ckpt_path, map_location=torch.device('cuda:0'))
    model.load_state_dict(checkpoint["model_state_dict"])
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    logger.info(f"已加载检查点: {checkpoint_path}")

    # 3.图像预处理
    transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 4.加载图像
    image_path = cfg.image_path
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    
    image = Image.open(image_path).convert('RGB')
    logger.info(f"已加载图像: {image_path}")

    # 5.预处理和预测
    image_tensor = transform(image).unsqueeze(0)  # 添加批次维度

    with torch.no_grad():
        output = model(image_tensor)
        # 计算概率分布
        prob = F.softmax(output, dim=1)
        # 获取预测类别和置信度
        confidence, predicted = torch.max(prob, 1)

    # 打印结果
    print(f"预测类别: {predicted.item()}")
    print(f"置信度: {confidence.item()}")

if __name__ == "__main__":
    predict()