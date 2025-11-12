import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import Downsample, ResBlock


class Encoder(nn.Module):
    """
    The CNN Encoder (E).
    Takes an image x and maps it to a latent representation z_hat.
    """
    def __init__(self, in_channels=3, n_z=256, n_blocks=4):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(in_channels, 64, kernel_size=3, padding=1))
        
        # Downsampling blocks
        for i in range(n_blocks):
            self.layers.append(Downsample(64 * (2**i), 64 * (2**(i+1))))
        
        # Residual blocks
        self.layers.append(ResBlock(64 * (2**n_blocks), 64 * (2**n_blocks)))
        
        # Final projection to n_z dimensions
        self.layers.append(nn.Conv2d(64 * (2**n_blocks), n_z, kernel_size=1))
        
    def forward(self, x):
        # Apply ReLU to all layers *except* the last one
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        
        # Apply the last layer without ReLU
        x = self.layers[-1](x)
        return x