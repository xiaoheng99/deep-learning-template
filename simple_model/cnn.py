import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class LeNet(nn.Module):
    def __init__(self, num_classes: int, hidden_dim_1: int, hidden_dim_2: int, conv_hidden_dim_1: int,
                 conv_hidden_dim_2: int):
        super(LeNet, self).__init__()
        self.conv_1 = nn.Conv2d(3, conv_hidden_dim_1, kernel_size=(5, 5))  # [3 ,32, 32]
        self.conv_2 = nn.Conv2d(conv_hidden_dim_1, conv_hidden_dim_2, kernel_size=(5, 5))
        self.fc1 = nn.Linear(16 * 5 * 5, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv_1(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv_2(x)), kernel_size=2)
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = LeNet(10, 120, 84, 6, 16).to(device=device)
    x = torch.randn(1, 3, 32, 32).to(device)
    print(net)
    print(net(x).shape)
    print("-----------------------------------------")
    summary(net, (3, 32, 32))

