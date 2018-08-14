from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

def MNISTloader(raw_data_path, batch_size):
    train_loader = DataLoader(
                 MNIST(raw_data_path, train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor()])),
                 batch_size=batch_size, shuffle=True, **dataloader_kwargs)

    test_loader = DataLoader(
                MNIST(data_path, train=False, download=True,
                      transform=transforms.Compose([transform;s.ToTensor()])),
                batch_size=64, shuffle=False, **dataloader_kwargs)
    return train_loader, test_loader
