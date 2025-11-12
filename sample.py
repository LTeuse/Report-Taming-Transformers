import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import os
import argparse
from tqdm import tqdm

# Import your custom models
from models.vqgan import VQGAN
from models.transformer import AutoregressiveTransformer

@torch.no_grad()
def sample_model(transformer_model, vqgan_model, device, num_samples, seq_len, top_k):
    """
    Autoregressively sample new codes from the Transformer and decode them
    into images with the VQGAN.
    """
    
    # Start with a random code as the "prime"
    codes = torch.randint(0, vqgan_model.codebook.num_codes, (num_samples, 1), device=device)

    print(f"Sampling {num_samples} images, one token at a time...")
    
    # 2. Autoregressive loop
    # We generate seq_len - 1 new tokens
    for _ in tqdm(range(seq_len - 1)):
        # Pass the current sequence of codes through the transformer
        logits = transformer_model(codes) # Shape: [B, current_seq_len, vocab_size]
        
        # We only care about the logits for the *next* token
        logits = logits[:, -1, :] # Shape: [B, vocab_size]
        
        # --- Top-k Sampling ---
        v, _ = torch.topk(logits, top_k, dim=-1)
        logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        next_code = torch.multinomial(probs, num_samples=1) # Shape: [B, 1]
        
        # Append the new token to our sequence
        codes = torch.cat((codes, next_code), dim=1)

    print("Code sequence generation complete.")
    
    # 3. Decode the generated codes into an image
    print("Decoding codes into images...")
    
    # Get the embedding vectors for each code
    z_q_flat = vqgan_model.codebook.embedding(codes) # Shape: [B, 64, 256]
    
    # Reshape to the grid shape the decoder expects [B, 256, 8, 8]
    B, _, embed_dim = z_q_flat.shape
    grid_size = int((seq_len)**0.5) # e.g., sqrt(64) = 8
    z_q = z_q_flat.permute(0, 2, 1).reshape(B, embed_dim, grid_size, grid_size)
    
    # Pass through the decoder
    generated_images = vqgan_model.decoder(z_q)
    
    print("Image generation complete.")
    return generated_images

def main():
    
    # --- EDIT YOUR CONSTANTS HERE ---
    VQGAN_CHECKPOINT_PATH = "checkpoints/vqgan_afhq_best.pt"
    TRANSFORMER_CHECKPOINT_PATH = "checkpoints/transformer_afhq_best.pt"
    NUM_SAMPLES = 16
    TOP_K = 100
    OUT_FILE = "generated_samples_tin.png"
    # --- END OF CONSTANTS ---

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. Load VQGAN (Stage 1 Model) ---
    print(f"Loading VQGAN from {VQGAN_CHECKPOINT_PATH}...")
    vqgan_ckpt = torch.load(VQGAN_CHECKPOINT_PATH, map_location=device)
    
    # --- Manually define VQGAN hparams ---
    # !! IMPORTANT: Make sure these match your Stage 1 training script !!
    VQGAN_HPARAMS = {
        'in_channels': 3, 
        'out_channels': 3, 
        'n_z': 256,          # embed_dim
        'num_codes': 1024,   # n_embed
        'n_blocks': 3,
        'commitment_cost': 50.0
    }
    
    vqgan = VQGAN(
        in_channels=VQGAN_HPARAMS['in_channels'],
        out_channels=VQGAN_HPARAMS['out_channels'],
        n_z=VQGAN_HPARAMS['n_z'],
        num_codes=VQGAN_HPARAMS['num_codes'],
        n_blocks=VQGAN_HPARAMS['n_blocks'],
        commitment_cost=VQGAN_HPARAMS['commitment_cost']
    ).to(device)
    vqgan.load_state_dict(vqgan_ckpt['vqgan_state_dict'])
    vqgan.eval()

    # --- 2. Load Transformer (Stage 2 Model) ---
    print(f"Loading Transformer from {TRANSFORMER_CHECKPOINT_PATH}...")
    trans_ckpt = torch.load(TRANSFORMER_CHECKPOINT_PATH, map_location=device)
    
    # Load hparams from checkpoint (this is the best way)
    if 'hparams' not in trans_ckpt:
        print("ERROR: Transformer checkpoint does not contain 'hparams'.")
        print("Please re-run train_stage2.py or manually define hparams here.")
        return
        
    trans_hparams = trans_ckpt['hparams']
    
    transformer = AutoregressiveTransformer(
        vocab_size=trans_hparams['vocab_size'],
        embed_dim=trans_hparams['embed_dim'],
        n_layers=trans_hparams['n_layers'],
        n_heads=trans_hparams['n_heads'],
        seq_len=trans_hparams['seq_len']
    ).to(device)
    transformer.load_state_dict(trans_ckpt['model_state_dict'])
    transformer.eval()

    # --- 3. Generate Samples ---
    generated_images = sample_model(
        transformer_model=transformer,
        vqgan_model=vqgan,
        device=device,
        num_samples=NUM_SAMPLES,
        seq_len=trans_hparams['seq_len'],
        top_k=TOP_K
    )

    # --- 4. Save Image ---
    save_image(generated_images, OUT_FILE, nrow=int(NUM_SAMPLES**0.5), normalize=True)
    print(f"\nSuccessfully saved {NUM_SAMPLES} generated images to {OUT_FILE}")

if __name__ == "__main__":
    main()