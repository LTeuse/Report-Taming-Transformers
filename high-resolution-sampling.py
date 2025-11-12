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
def sample_high_res(transformer_model, vqgan_model, device, num_samples, 
                    target_img_size, code_grid_size, seq_len, top_k):
    """
    Autoregressively sample new codes in a sliding-window fashion.
    """
    
    # 1. Create an empty grid to hold our generated codes
    # We use a start token (e.g., 0) to prime the model
    codes = torch.zeros(num_samples, code_grid_size, code_grid_size, 
                        dtype=torch.long, device=device)

    print(f"Sampling {num_samples} images ({target_img_size}x{target_img_size}), one token at a time...")
    
    # 2. Autoregressive loop (sliding window)
    # We iterate over every single position in our new, large code grid
    for i in tqdm(range(code_grid_size)):
        for j in range(code_grid_size):
            
            # --- This is the core sliding-window logic ---
            
            # 1. Flatten the 2D grid of codes into a 1D sequence
            codes_flat = codes.reshape(num_samples, -1) # Shape: [B, 32*32 = 1024]
            
            # 2. Calculate our current position in the 1D sequence
            current_pos = i * code_grid_size + j
            
            # 3. Get the "context" - the (seq_len - 1) tokens before this position
            if current_pos == 0:
                # This is the very first token, we have no context
                # So we create a "start token" (e.g., 0)
                context = torch.zeros(num_samples, 1, dtype=torch.long, device=device)
            else:
                # Get the slice of tokens that the Transformer can see
                start = max(0, current_pos - (seq_len - 1))
                context = codes_flat[:, start:current_pos] # Shape: [B, (up to 63)]
            
            # 4. Pass the context through the transformer
            # The transformer will predict the *next* token
            logits = transformer_model(context) # Shape: [B, context_len, vocab_size]
            
            # 5. We only care about the logits for the *last* token
            logits = logits[:, -1, :] # Shape: [B, vocab_size]
            
            # 6. --- Top-k Sampling ---
            v, _ = torch.topk(logits, top_k, dim=-1)
            logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_code = torch.multinomial(probs, num_samples=1) # Shape: [B, 1]
            
            # 7. Place the new token into our 2D grid
            codes[:, i, j] = next_code.squeeze()

    print("Code sequence generation complete.")
    
    # 3. Decode the full 32x32 grid into an image
    print("Decoding codes into images...")
    
    # Get the embedding vectors for each code
    # codes is [B, 32, 32] -> flatten to [B, 1024]
    codes_flat = codes.reshape(num_samples, -1)
    z_q_flat = vqgan_model.codebook.embedding(codes_flat) # Shape: [B, 1024, 256]
    
    # Reshape to the grid shape the decoder expects [B, 256, 32, 32]
    B, _, embed_dim = z_q_flat.shape
    z_q = z_q_flat.permute(0, 2, 1).reshape(B, embed_dim, code_grid_size, code_grid_size)
    
    # Pass through the decoder
    generated_images = vqgan_model.decoder(z_q)
    
    print("Image generation complete.")
    return generated_images

def main():
    
    # --- EDIT YOUR CONSTANTS HERE ---
    VQGAN_CHECKPOINT_PATH = "checkpoints/vqgan_afhq_best.pt"
    TRANSFORMER_CHECKPOINT_PATH = "checkpoints/transformer_afhq_best.pt"
    
    # --- New High-Resolution Constants ---
    TARGET_IMAGE_SIZE = 64 # The 256x256 image you want to generate
    
    NUM_SAMPLES = 4  # (Keep this low, high-res sampling is slow!)
    TOP_K = 100
    OUT_FILE = "generated_samples_afhq_high_res.png"
    # --- END OF CONSTANTS ---

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. Load VQGAN (Stage 1 Model) ---
    print(f"Loading VQGAN from {VQGAN_CHECKPOINT_PATH}...")
    vqgan_ckpt = torch.load(VQGAN_CHECKPOINT_PATH, map_location=device)
    
    # !! IMPORTANT: Make sure these match your Stage 1 training script !!
    VQGAN_HPARAMS = {
        'in_channels': 3, 'out_channels': 3, 'n_z': 256, 'num_codes': 1024, 
        'n_blocks': 4, 'commitment_cost': 50.0
    }
    
    vqgan = VQGAN(**VQGAN_HPARAMS).to(device)
    vqgan.load_state_dict(vqgan_ckpt['vqgan_state_dict'])
    vqgan.eval()

    # --- 2. Load Transformer (Stage 2 Model) ---
    print(f"Loading Transformer from {TRANSFORMER_CHECKPOINT_PATH}...")
    trans_ckpt = torch.load(TRANSFORMER_CHECKPOINT_PATH, map_location=device)
    trans_hparams = trans_ckpt['hparams']
    
    transformer = AutoregressiveTransformer(**trans_hparams).to(device)
    transformer.load_state_dict(trans_ckpt['model_state_dict'])
    transformer.eval()

    # --- 3. Calculate Model Parameters ---
    # We trained with n_blocks=3, so 2^3 = 8
    # The VQGAN downsamples the image by a factor of 8.
    DOWNSAMPLE_FACTOR = 2 ** VQGAN_HPARAMS['n_blocks'] 
    
    # To get a 256x256 image, we need a 32x32 code grid (256 / 8 = 32)
    CODE_GRID_SIZE = TARGET_IMAGE_SIZE // DOWNSAMPLE_FACTOR
    
    # The Transformer's context window (from training)
    SEQ_LEN = trans_hparams['seq_len']
    
    print(f"Target image size: {TARGET_IMAGE_SIZE}x{TARGET_IMAGE_SIZE}")
    print(f"VQGAN downsample factor: {DOWNSAMPLE_FACTOR}")
    print(f"Target code grid size: {CODE_GRID_SIZE}x{CODE_GRID_SIZE}")
    print(f"Transformer context window: {SEQ_LEN} tokens")

    # --- 4. Generate Samples ---
    generated_images = sample_high_res(
        transformer_model=transformer,
        vqgan_model=vqgan,
        device=device,
        num_samples=NUM_SAMPLES,
        target_img_size=TARGET_IMAGE_SIZE,
        code_grid_size=CODE_GRID_SIZE,
        seq_len=SEQ_LEN,
        top_k=TOP_K
    )

    # --- 5. Save Image ---
    save_image(generated_images, OUT_FILE, nrow=int(NUM_SAMPLES**0.5), normalize=True)
    print(f"\nSuccessfully saved {NUM_SAMPLES} high-res images to {OUT_FILE}")

if __name__ == "__main__":
    main()