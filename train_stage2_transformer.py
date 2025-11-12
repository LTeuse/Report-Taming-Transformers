import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# Removed SummaryWriter import
from models.transformer import AutoregressiveTransformer
from data.code_dataset import CodeDataset # The dataloader for our new codes
import os
# from tqdm import tqdm # For a nice progress bar

def main():
    """
    Main function to run the FULL training for Stage 2 (Transformer).
    """
    # --- 1. Setup and Configuration ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Hyperparameters ---
    num_epochs = 50      # Set the total number of epochs
    lr = 3e-4            # Learning rate (from our check script)
    batch_size = 64      # Adjust based on your GPU memory
    log_interval = 100   # How often to print loss to console
    
    # --- Model Hyperparameters (Must match check script) ---
    VOCAB_SIZE = 16384  # Your n_embed from VQGAN
    EMBED_DIM = 512    # Transformer embedding dimension
    N_LAYERS = 8       # Number of Transformer blocks
    N_HEADS = 8        # Number of attention heads
    SEQ_LEN = 256      # 16x16 grid = 256 tokens ; 64 for 8x8
    
    # --- Paths ---
    train_code_dir = '/local/data/afhq_codes/train'
    val_code_dir = '/local/data/afhq_codes/val'
    checkpoint_dir = 'checkpoints'
    
    # Removed log_dir, as it was only for TensorBoard
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # --- 2. Data Loading ---
    print("Loading datasets...")
    if not os.path.exists(train_code_dir) or not os.path.exists(val_code_dir):
        print(f"Error: Code directories not found. Did you run 'scripts/precompute_codes.py'?")
        return
        
    train_dataset = CodeDataset(codes_dir=train_code_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    val_dataset = CodeDataset(codes_dir=val_code_dir)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Loaded {len(train_dataset)} training codes and {len(val_dataset)} validation codes.")

    # --- 3. Model, Optimizer, and Loss ---
    print("Initializing Transformer model...")
    model = AutoregressiveTransformer(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        seq_len=SEQ_LEN
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # CrossEntropyLoss *is* the Negative Log-Likelihood (NLL)
    criterion = nn.CrossEntropyLoss()
    
    # --- 4. Logging ---
    # Removed SummaryWriter initialization
    print(f"Checkpoints will be saved to: {checkpoint_dir}")

    # --- 5. Training Loop ---
    print(f"Starting full training for {num_epochs} epochs...")
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        # --- Training Phase ---
        model.train()
        
        # train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False)
        
        # for i, batch in enumerate(train_pbar):
        for i, batch in enumerate(train_loader):
            codes = batch.to(device)
            
            # --- Autoregressive setup ---
            inputs = codes[:, :-1]  # Shape [B, 63]
            targets = codes[:, 1:]  # Shape [B, 63]
            
            # Forward pass
            logits = model(inputs) # Shape [B, 63, vocab_size]
            
            # Calculate loss
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # --- This is the Stage-1 style logging ---
            if (i + 1) % log_interval == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss (NLL): {loss.item():.4f}')
        
        # train_pbar.close() # Close the progress bar
            
        # --- Validation Phase ---
        print("Running validation...")
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                codes = batch.to(device)
                
                inputs = codes[:, :-1]
                targets = codes[:, 1:]
                
                logits = model(inputs)
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        # --- End of Epoch Summary ---
        print(f"\n--- End of Epoch {epoch+1} ---")
        print(f"Average Validation Loss (NLL): {avg_val_loss:.4f}")
        
        # --- Checkpointing ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(checkpoint_dir, "transformer_afhq_best.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                # Save hparams so we can load it for sampling
                'hparams': {
                    'vocab_size': VOCAB_SIZE,
                    'embed_dim': EMBED_DIM,
                    'n_layers': N_LAYERS,
                    'n_heads': N_HEADS,
                    'seq_len': SEQ_LEN
                }
            }, checkpoint_path)
            print(f"New best model saved with Val NLL: {best_val_loss:.4f} to {checkpoint_path}\n")
        else:
            print(f"Validation loss did not improve from best: {best_val_loss:.4f}\n")

    # Removed writer.close()
    print("\n--- Stage 2 Training Complete ---")

if __name__ == "__main__":
    main()