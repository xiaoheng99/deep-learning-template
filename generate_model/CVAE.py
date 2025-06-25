import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int, image_channels: int, hidden_channels: int):
        super().__init__()

        self.embed = nn.Embedding(num_classes, num_classes)

        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels + num_classes, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

    
    def forward(self, x, cond):
        cond_embed = self.embed(cond).unsqueeze(2).unsqueeze(3).expand(-1, -1, x.size(2), x.size(3))
        pass
