import torch
import torch.nn.functional as F
from models.vqgan import VQGAN                 
from models.discriminator import PatchDiscriminator 
from torch.utils.data import DataLoader
from data.afhq_dataset import AFHQDataset
import os
from torchvision.utils import save_image
import lpips  # <-- ADDED: Import the LPIPS library

def main():
    """
    Main function to run the FULL VQGAN Stage 1 training.
    This version uses the correct L1 + LPIPS loss combination.
    """
    # --- 1. Setup and Configuration ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Hyperparameters ---
    lr_g = 1e-4  # Generator LR
    lr_d = 1e-6  # Discriminator LR
    gan_weight = 0.01
    commitment_cost = 50.0
    l1_weight = 1.0
    perceptual_weight = 1.0
    
    # --- Full Training Parameters ---
    num_epochs = 100    # Set the total number of epochs
    batch_size = 64     # Adjust this based on your GPU memory
    image_size = 256    # Tiny ImageNet default size
    log_interval = 100   # How often to print loss
    
    # --- Model Structure Parameters ---
    n_embed =   8192
    embed_dim = 256
    n_blocks = 4
    
    # --- Paths ---
    dataset_root = '/local/'
    log_dir = "logs/"
    checkpoint_dir = "checkpoints"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    CONTINUE_FROM_CHECKPOINT = "checkpoints/vqgan_afhq_best.pt" # Or None to start from scratch

    # --- 2. Data Loading ---
    # if not os.path.exists(os.path.join(dataset_root, 'tiny-imagenet-200')):
    #     print(f"Error: Tiny ImageNet dataset not found in '{os.path.join(dataset_root, 'tiny-imagenet-200')}'")
    #     return
    if not os.path.exists(os.path.join(dataset_root, 'afhq')):
        print(f"Error: AFHQ dataset not found in '{os.path.join(dataset_root, 'afhq')}'")
        return
    
    train_dataset = AFHQDataset(root_dir=dataset_root, split='train', size=image_size, random_flip=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    val_dataset = AFHQDataset(root_dir=dataset_root, split='val', size=image_size, random_flip=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Loaded {len(train_dataset)} training images and {len(val_dataset)} validation images.")


    # --- 3. Model and Optimizer Initialization ---
    vqgan = VQGAN(in_channels=3, 
                  out_channels=3, 
                  n_z=embed_dim, 
                  num_codes=n_embed, 
                  n_blocks=n_blocks,
                  commitment_cost=commitment_cost).to(device)
    
    discriminator = PatchDiscriminator(in_channels=3, n_layers=3).to(device)

    optimizer_g = torch.optim.Adam(vqgan.parameters(), lr=lr_g)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d)

    # --- Initialize the actual Perceptual Loss (LPIPS) ---
    perceptual_loss = lpips.LPIPS(net='vgg').to(device)

    start_epoch = 0
    best_val_l1 = float('inf')

    if CONTINUE_FROM_CHECKPOINT is not None and os.path.exists(CONTINUE_FROM_CHECKPOINT):
        print(f"--- Loading checkpoint from {CONTINUE_FROM_CHECKPOINT} ---")
        checkpoint = torch.load(CONTINUE_FROM_CHECKPOINT, map_location=device)
        
        # Load model weights
        vqgan.load_state_dict(checkpoint['vqgan_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        # Load optimizer states
        optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        
        # Load epoch and loss info
        start_epoch = checkpoint['epoch'] # We'll start at the *next* epoch
        best_val_l1 = checkpoint['avg_recon_loss']
        
        print(f"Resuming training from Epoch {start_epoch + 1}")
        print(f"Loaded best validation L1: {best_val_l1:.4f}")
    else:
        print("--- Starting training from scratch ---")

    # --- 4. Training Loop ---
    print(f"Starting full training from epoch {start_epoch + 1} to {num_epochs}...")

    for epoch in range(start_epoch, num_epochs):
        vqgan.train() 
        discriminator.train()
        
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_percep_loss = 0.0 # To track perceptual loss

        for i, batch in enumerate(train_loader):

            x = batch.to(device) 

            # --- Generator Step ---
            optimizer_g.zero_grad()

            # Assuming your VQGAN correctly returns 4 items
            x_recon, loss_codebook, loss_commit, _ = vqgan(x) 

            # --- EDITED: Calculate L1 and Perceptual Loss ---
            loss_recon = F.l1_loss(x_recon, x) * l1_weight # L1 loss
            loss_p = perceptual_loss(x_recon, x).mean() * perceptual_weight # Perceptual loss

            # Generator GAN Loss
            logits_fake = discriminator(x_recon)
            loss_g_gan = -torch.mean(logits_fake)

            # --- EDITED: Total Generator Loss (L1 + LPIPS + VQ + GAN) ---
            loss_g = loss_recon + loss_p + \
                     loss_codebook + loss_commit + \
                     gan_weight * loss_g_gan

            loss_g.backward()
            optimizer_g.step()

            # --- Discriminator Step ---
            optimizer_d.zero_grad()

            logits_real = discriminator(x.detach())
            logits_fake = discriminator(x_recon.detach())

            loss_d_real = torch.mean(F.relu(1. - logits_real))
            loss_d_fake = torch.mean(F.relu(1. + logits_fake))
            loss_d = 0.5 * (loss_d_real + loss_d_fake)

            loss_d.backward()
            optimizer_d.step()
            
            # --- Logging ---
            epoch_g_loss += loss_g.item()
            epoch_d_loss += loss_d.item()
            epoch_recon_loss += loss_recon.item()
            epoch_percep_loss += loss_p.item()

            if (i + 1) % log_interval == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], '
                      f'G_Loss: {loss_g.item():.4f}, D_Loss: {loss_d.item():.4f}, '
                      f'Recon_L1: {loss_recon.item():.4f}, '
                      f'Percep_L: {loss_p.item():.4f}, '
                      f'Codebook: {loss_codebook.item():.4f}, '
                      f'Commit: {loss_commit.item():.4f}')

        # --- End of Epoch ---
        avg_g_loss = epoch_g_loss / len(train_loader)
        avg_d_loss = epoch_d_loss / len(train_loader)
        avg_recon_loss = epoch_recon_loss / len(train_loader)
        avg_percep_loss = epoch_percep_loss / len(train_loader)
        
        print(f"\n--- End of Epoch {epoch+1} ---")
        print(f"Average G_Loss: {avg_g_loss:.4f}, Average D_Loss: {avg_d_loss:.4f}")
        print(f"Average Recon_L1: {avg_recon_loss:.4f}, Average Percep_L: {avg_percep_loss:.4f}")

        # --- Validation and Image Saving ---
        vqgan.eval()
        with torch.no_grad():
            val_batch = next(iter(val_loader)).to(device)
            x_val_recon, _, _, _= vqgan(val_batch)
            
            comparison = torch.cat([val_batch[:8], x_val_recon[:8]])
            save_image(comparison, 
                       os.path.join(log_dir, f"recon_epoch_{epoch+1}.png"), 
                       normalize=True, nrow=8)
            print(f"Saved validation reconstruction image to {log_dir}")

        # --- Model Checkpointing ---
        if avg_recon_loss < best_val_l1: # Save based on L1 loss
            best_val_l1 = avg_recon_loss
            checkpoint_path = os.path.join(checkpoint_dir, f"vqgan_afhq_best.pt")
            torch.save({
                'epoch': epoch + 1,
                'vqgan_state_dict': vqgan.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'avg_recon_loss': avg_recon_loss,
                # Save hparams for Stage 2 sampling
                'hparams': {
                    'in_channels': 3, 'out_channels': 3, 'n_z': embed_dim, 
                    'num_codes': n_embed, 'n_blocks': n_blocks, 
                    'commitment_cost': commitment_cost
                }
            }, checkpoint_path)
            print(f"New best model saved with Val L1: {best_val_l1:.4f} to {checkpoint_path}\n")
        else:
            print(f"Validation loss did not improve from best: {best_val_l1:.4f}\n")
            
    print("--- Full Training Complete ---")

if __name__ == "__main__":
    main()