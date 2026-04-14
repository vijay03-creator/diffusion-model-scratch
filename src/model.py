import torch
from torch import nn

class BasicUNet(nn.Module):
    """
    Minimal UNet for diffusion model.
    Input  : noisy image (1 channel, 28x28)
    Output : predicted clean image (1 channel, 28x28)
    """

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # Down path - compress the image
        self.down_layers = nn.ModuleList([
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
        ])

        # Up path - expand back to original size
        self.up_layers = nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, out_channels, kernel_size=5, padding=2),
        ])

        self.act       = nn.SiLU()
        self.downscale = nn.MaxPool2d(2)
        self.upscale   = nn.Upsample(scale_factor=2)

    def forward(self, x):
        h = []

        # Down path
        for i, layer in enumerate(self.down_layers):
            x = self.act(layer(x))
            if i < 2:
                h.append(x)          # save for skip connection
                x = self.downscale(x)

        # Up path
        for i, layer in enumerate(self.up_layers):
            if i > 0:
                x = self.upscale(x)
                x += h.pop()         # skip connection
            x = self.act(layer(x))

        return x


def get_model(device):
    """Create model and move to device"""
    model = BasicUNet()
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")
    return model