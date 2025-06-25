import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os


def data_loader(root, batch_size, split, pin_memory, num_workers):
    transform = transforms.Compose([
        transforms.ToTensor(),  # 归一化到[0, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root=root, download=True, train=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=root, download=True, train=False, transform=transform)

    datasets_size = len(train_dataset)
    train_size, val_size = int(datasets_size*split[0]), int(datasets_size*split[1])
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = data_loader()
    print(f"test_loader:{len(test_loader)}, train_loader:{len(train_loader)}, val_loader:{len(val_loader)}")
    for i, data in enumerate(train_loader):
        image, label = data
        print(image.shape)
        print(label.shape)
        break