import torchvision.transforms as transforms
import torchvision
import torch
from torch.utils.data import DataLoader


class MNISTDataset:
    def __init__(self, train=True):

        transform = transforms.Compose([
            transforms.ToTensor(),            
        ])

        self.data = torchvision.datasets.MNIST(root="./MNIST", train=train, download=True, transform=transform)

        self.loader = DataLoader(self.data, batch_size=64, shuffle=False, num_workers=2)


    def __len__(self):
        return len(self.data)

    def get_data(self):

        return self.loader