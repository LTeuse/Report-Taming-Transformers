import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import ResBlock, Upsample


class Decoder(nn.Module):
    """
    The CNN Decoder (G).
    Takes a quantized latent z_q and maps it back to an image x_recon.
    """
    def __init__(self, out_channels=3, n_z=256, n_blocks=4):
        super().__init__()
        
        # Initial projection from n_z
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(n_z, 64 * (2**n_blocks), kernel_size=1))
        
        # Residual block
        self.layers.append(ResBlock(64 * (2**n_blocks), 64 * (2**n_blocks)))
        
        # Upsampling blocks
        for i in range(n_blocks):
            self.layers.append(Upsample(64 * (2**(n_blocks-i)), 64 * (2**(n_blocks-i-1))))

        # Final projection to out_channels
        self.layers.append(nn.Conv2d(64, out_channels, kernel_size=3, padding=1))

    def forward(self, z_q):
        # Apply ReLU to all layers *except* the last one
        for layer in self.layers[:-1]:
            z_q = F.relu(layer(z_q))
        
        # Apply the last layer without ReLU
        z_q = self.layers[-1](z_q)
        return z_q
