"""
This is a implementation of the VQGAN, as described in the
CVPR 2021 paper "Taming Transformers for High-Resolution Image Synthesis"
(https://arxiv.org/abs/2012.09841).

The VQGAN is a VQVAE trained with an adversarial (GAN) loss
and a perceptual loss. This file use the module
- encoder
- codebook
- decoder
to build and train the VQGAN (Stage 1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import Encoder
from models.codebook import Codebook
from models.decoder import Decoder
from models.discriminator import PatchDiscriminator

class VQGAN(nn.Module):
    """
    The full VQGAN model, which combines the Encoder, Decoder,
    and Codebook.
    """
    def __init__(self, in_channels=3, out_channels=3, n_z=256, 
                 num_codes=1024, n_blocks=4, commitment_cost=0.25):
        super().__init__()
        self.encoder = Encoder(in_channels, n_z, n_blocks)
        self.codebook = Codebook(num_codes, n_z, commitment_cost)
        self.decoder = Decoder(out_channels, n_z, n_blocks)

    def forward(self, x):
        """
        Forward pass: x -> E(x) -> Z(E(x)) -> G(Z(E(x)))
        """
        z_hat = self.encoder(x)
        z_q, codebook_loss, commitment_loss, indices = self.codebook(z_hat)
        x_recon = self.decoder(z_q)
        
        return x_recon, codebook_loss, commitment_loss, indices

    def get_indices(self, x):
        """
        helper method to only get the indices.
        """
        z_hat = self.encoder(x)
        _, _, _, indices = self.codebook(z_hat)
        return indices