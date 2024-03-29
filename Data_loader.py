from torchvision.datasets import MNIST
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms




def loadMNIST():
    
    train_transform = transforms.Compose([
        transforms.Pad(padding=2, fill=0, padding_mode="constant"),
        transforms.ToTensor()
    ])
    
    # download dataset
    train_set = MNIST(root="../data/", train=True, transform=train_transform, download=True)
    test_set = MNIST(root="../data/", train=False, transform=train_transform)
    
    # load dataset
    batch_size = 4
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader