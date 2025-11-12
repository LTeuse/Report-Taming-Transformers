import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerBlock(nn.Module):
    """
    A standard Transformer block with Multi-Head Attention and a Feed-Forward Network.
    """
    def __init__(self, embed_dim, n_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # x shape is [B, seq_len, embed_dim]
        # mask shape is [seq_len, seq_len]
        
        # Self-Attention
        attn_output, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = x + self.dropout(attn_output) # Residual connection
        x = self.norm1(x)
        
        # Feed-Forward Network
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output) # Residual connection
        x = self.norm2(x)
        
        return x

class AutoregressiveTransformer(nn.Module):
    """
    The main autoregressive Transformer model.
    """
    def __init__(self, vocab_size, embed_dim, n_layers, n_heads, seq_len, dropout=0.1):
        """
        Args:
            vocab_size (int): The number of codes (e.g., n_embed=1024).
            embed_dim (int): The embedding dimension (e.g., 256).
            n_layers (int): Number of Transformer blocks.
            n_heads (int): Number of attention heads.
            seq_len (int): The length of the code sequence (e.g., 8*8 = 64).
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        
        # Token embedding: Maps code indices to vectors
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional embedding: Maps positions (0 to seq_len-1) to vectors
        self.position_embedding = nn.Embedding(seq_len, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Stack of Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, n_heads, dropout) for _ in range(n_layers)]
        )
        
        # Final layer norm and linear head to predict logits
        self.norm_final = nn.LayerNorm(embed_dim)
        self.to_logits = nn.Linear(embed_dim, vocab_size)
        
        # Create a permanent causal mask
        # This is a square matrix that prevents "looking ahead"
        self.register_buffer("causal_mask", self._generate_causal_mask(seq_len))

    def _generate_causal_mask(self, size):
        """
        Generates a causal mask (triangular) to prevent attention to future tokens.
        Shape: [size, size]
        """
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, idx):
        # idx shape: [B, seq_len]
        B, T = idx.shape
        
        # 1. Get token embeddings
        token_embed = self.token_embedding(idx) # [B, seq_len, embed_dim]
        
        # 2. Get positional embeddings
        # Create a tensor of positions: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(0, T, device=idx.device).unsqueeze(0) # [1, seq_len]
        pos_embed = self.position_embedding(positions) # [1, seq_len, embed_dim]
        
        # 3. Add them together
        x = token_embed + pos_embed
        x = self.dropout(x)
        
        # 4. Pass through Transformer blocks
        # --- THIS IS THE FIX ---
        # Slice the pre-computed mask to match the input sequence length T
        mask = self.causal_mask[:T, :T]
        for block in self.blocks:
            x = block(x, mask) # Pass the (potentially sliced) causal mask here
            
        # 5. Final norm and projection to logits
        x = self.norm_final(x)
        logits = self.to_logits(x) # [B, seq_len, vocab_size]
        
        return logits