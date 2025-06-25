import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, latent_dim: int, image_channels: int, hidden_channel: int):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, hidden_channel, kernel_size=4, stride=2, padding=1),  # ->16*16
            nn.ReLU(),
            nn.Conv2d(hidden_channel, hidden_channel * 2, kernel_size=4, stride=2, padding=1),  # ->8*8
            nn.ReLU(),
            nn.Conv2d(hidden_channel * 2, hidden_channel * 4, kernel_size=4, stride=2, padding=1),  # ->4*4
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_channel * 4 * 4 * 4, latent_dim)  # 这里需要得到image的图像尺寸，这里可能不够自动化
        self.fc_logvar = nn.Linear(hidden_channel * 4 * 4 * 4, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # [batch_size, dim]
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim: int , image_channels: int, hidden_channel: int):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, hidden_channel * 4 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_channel * 4, hidden_channel * 2, kernel_size=4, stride=2, padding=1),  # ->8*8
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channel * 2, hidden_channel, kernel_size=4, stride=2, padding=1),  # ->16*16
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channel, image_channels, kernel_size=4, stride=2, padding=1),  # ->32*32
            nn.Sigmoid(),  # 得到一个0-1的值
        )
    
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), -1, 4, 4)
        x = self.decoder(x)
        return x


class VAE(nn.Module):
    def __init__(self, latent_dim: int, image_channels: int, hidden_channel: int):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim, image_channels, hidden_channel)
        self.decoder = Decoder(latent_dim, image_channels, hidden_channel)
    
    def reparameter(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameter(mu, logvar)
        return self.decoder(z), mu, logvar