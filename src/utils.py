import torch

def corrupt(x, amount):
    """
    Corrupt image by mixing with noise.
    amount=0 -> clean image
    amount=1 -> pure noise
    """
    noise  = torch.rand_like(x)
    amount = amount.view(-1, 1, 1, 1)
    return x * (1 - amount) + noise * amount


def get_device():
    """Return GPU if available, else CPU"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device