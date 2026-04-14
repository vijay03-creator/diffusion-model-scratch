import torch
import torchvision
from torch.utils.data import DataLoader

def get_dataloader(batch_size=128):
    """Download and return MNIST dataloader"""
    
    dataset = torchvision.datasets.MNIST(
        root="data/",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    print(f"Dataset loaded: {len(dataset)} images")
    print(f"Batch size: {batch_size}")
    print(f"Total batches: {len(dataloader)}")
    
    return dataloader