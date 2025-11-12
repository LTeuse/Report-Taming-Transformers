import torch
from torch.utils.data import DataLoader
from models.vqgan import VQGAN
from data.afhq_dataset import AFHQDataset
# from data.tiny_imagenet_dataset import TinyImageNetDataset
import os
from tqdm import tqdm

def precompute_and_save(vqgan, data_loader, save_dir):
    """
    Runs all images in the data_loader through the vqgan's encoder
    and saves the resulting code indices to save_dir.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # We will save indices sequentially, 0.pt, 1.pt, etc.
    file_counter = 0 
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Pre-computing codes in {save_dir}"):
            images = batch.to(device)
            
            # Use the new helper method to get indices
            indices = vqgan.get_indices(images) # Shape [B, H, W]
            
            # Save each index map in the batch as a separate file
            for i in range(indices.size(0)):
                # Move to CPU, convert to int16 to save space
                index_map = indices[i].cpu().to(torch.int16) 
                save_path = os.path.join(save_dir, f"{file_counter}.pt")
                torch.save(index_map, save_path)
                file_counter += 1

if __name__ == "__main__":
    # --- Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- IMPORTANT: Match these to your trained model ---
    checkpoint_path = "checkpoints/vqgan_afhq_best.pt" # Assuming 50 epochs
    n_embed = 16384
    embed_dim = 256
    image_size = 256
    n_blocks = 4
    commitment_cost = 50.0
    
    # --- Dataset paths ---
    dataset_root = '/local/'
    output_code_dir = '/local/data/afhq_codes'
    
    # --- 1. Load Trained VQGAN ---
    print(f"Loading VQGAN from {checkpoint_path}...")
    vqgan = VQGAN(in_channels=3, 
                  out_channels=3, 
                  n_z=embed_dim, 
                  num_codes=n_embed, 
                  n_blocks=n_blocks,
                  commitment_cost=commitment_cost).to(device)
                  
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vqgan.load_state_dict(checkpoint['vqgan_state_dict'])
    vqgan.eval() # Set model to evaluation mode
    print("VQGAN loaded successfully.")

    # --- 2. Load Datasets ---
    # We use a larger batch size for pre-computation as no gradients are needed
    batch_size = 128 
    
    train_dataset = AFHQDataset(root_dir=dataset_root, split='train', size=image_size, random_flip=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    val_dataset = AFHQDataset(root_dir=dataset_root, split='val', size=image_size, random_flip=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # --- 3. Run Pre-computation ---
    # Process and save the training set codes
    precompute_and_save(vqgan, train_loader, save_dir=os.path.join(output_code_dir, 'train'))
    
    # Process and save the validation set codes
    precompute_and_save(vqgan, val_loader, save_dir=os.path.join(output_code_dir, 'val'))

    print("\nAll codes pre-computed and saved.")
    print(f"Your new dataset is ready in: {output_code_dir}")