import torch
import torch.nn as nn
import torch.nn.functional as F

class Codebook(nn.Module):
    """
    The discrete Codebook (Z).
    This module performs the vector quantization.
    It stores the K codebook vectors e_k and finds the closest one
    for each encoder output vector z_e.
    """
    def __init__(self, num_codes=1024, embedding_dim=256, commitment_cost=0.25):
        super().__init__()
        self.num_codes = num_codes
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost  # Beta from Eq. (4) [cite: 131]
        
        # Codebook embedding vectors
        self.embedding = nn.Embedding(self.num_codes, self.embedding_dim)
        # Initialize embeddings for stability
        self.embedding.weight.data.uniform_(-1.0 / self.num_codes, 1.0 / self.num_codes)

    def forward(self, z_hat):
        """
        Takes the continuous output of the encoder z_hat [B, C, H, W]
        and maps it to the closest discrete codebook vectors.
        """
        # 1. Reshape z_hat to [B*H*W, C]
        B, C, H, W = z_hat.shape
        # Permute to [B, H, W, C] and then flatten
        z_hat = z_hat.permute(0, 2, 3, 1).contiguous()
        z_hat_flat = z_hat.view(-1, self.embedding_dim)
        
        # 2. Calculate L2 distances to all codebook vectors
        distances = torch.sum(z_hat_flat**2, dim=1, keepdim=True) + \
                    torch.sum(self.embedding.weight**2, dim=1) - \
                    2 * torch.matmul(z_hat_flat, self.embedding.weight.t())
        
        # 3. Find the index of the closest codebook vector (Eq. 2)
        embedding_indices = torch.argmin(distances, dim=1) # [B*H*W]
        
        # 4. Get the quantized vectors z_q
        z_q_flat = self.embedding(embedding_indices) # [B*H*W, C]
        z_q = z_q_flat.view(B, H, W, C) # Reshape back to [B, H, W, C]
        
        # --- 5. Compute losses from Equation (4) ---
        # Codebook loss (moves embeddings towards encoder output)
        codebook_loss = F.mse_loss(z_q, z_hat.detach())
        
        # Commitment loss (moves encoder output towards embeddings)
        commitment_loss = self.commitment_cost * F.mse_loss(z_hat, z_q.detach())
        
        # --- 6. Straight-Through Estimator (STE) ---
        z_q_ste = z_hat + (z_q - z_hat).detach()
        
        # Permute back to [B, C, H, W] for the decoder
        z_q_ste = z_q_ste.permute(0, 3, 1, 2).contiguous()
        
        # --- Also return the indices ---
        # Reshape indices to [B, H, W] for saving
        embedding_indices = embedding_indices.view(B, H, W)
        
        return z_q_ste, codebook_loss, commitment_loss, embedding_indices