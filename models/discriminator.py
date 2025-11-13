"""
This is an implementation of the Patch-Based Discriminator (D)
used in the VQGAN paper ("Taming Transformers").

This module is designed to be imported into the main training script.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchDiscriminator(nn.Module):
    """
    The Patch-Based CNN Discriminator (D).

    It's a "patch-based" discriminator, meaning its output is a 2D grid
    of logits, not a single scalar. This is also known as a PatchGAN.
    """
    def __init__(self, in_channels=3, n_layers=3, base_channels=64):
        """
        Initializes the discriminator.
        Args:
            in_channels (int): Number of input channels (e.g., 3 for RGB).
            n_layers (int): Number of strided convolutional layers.
            base_channels (int): Number of channels in the first layer.
        """
        super().__init__()
        
        layers = [
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        ch_multiplier = 1
        for i in range(1, n_layers):
            in_ch = base_channels * ch_multiplier
            ch_multiplier *= 2
            out_ch = base_channels * ch_multiplier
            
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),  # Using BatchNorm as is common
                nn.LeakyReLU(0.2, inplace=True)
            ])
            
        # Final layer to produce logits
        # After the loop, in_ch is the output of the last layer
        in_ch = base_channels * ch_multiplier
        layers.append(
            nn.Conv2d(in_ch, 1, kernel_size=4, stride=1, padding=1)
        )
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Takes an image batch [B, C, H, W] and returns a
        logit grid [B, 1, H_patch, W_patch].
        """
        return self.model(x)

    def compute_loss(self, real_logits, fake_logits):
        """
        Calculates the hinge loss for the discriminator.
        This is a common and stable choice for GANs.
        The discriminator tries to make real_logits > 1 and fake_logits < -1.
        """
        # Loss for real images
        real_loss = torch.mean(F.relu(1.0 - real_logits))
        
        # Loss for fake images
        fake_loss = torch.mean(F.relu(1.0 + fake_logits))
        
        # Total discriminator loss
        total_d_loss = 0.5 * (real_loss + fake_loss)
        return total_d_loss





# --- Main execution block to test the module ---

if __name__ == "__main__":
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing PatchDiscriminator on device: {device}")

    # Model parameters
    BATCH_SIZE = 4
    IMG_SIZE = 256
    
    # 1. Instantiate the discriminator
    discriminator = PatchDiscriminator(in_channels=3, n_layers=3).to(device)
    
    # 2. Create dummy inputs (a batch of "real" and "fake" images)
    real_images = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE).to(device)
    fake_images = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE).to(device)
    
    # 3. Test the forward pass
    real_logits = discriminator(real_images)
    fake_logits = discriminator(fake_images.detach()) # Detach fake images
    
    print(f"Input image shape:  {real_images.shape}")
    print(f"Output logit shape: {real_logits.shape}")
    
    # 4. Test the loss calculation
    loss = discriminator.compute_loss(real_logits, fake_logits)
    print(f"Calculated loss:    {loss.item():.4f}")

    # 5. Check output shape
    # For a 256x256 input with n_layers=3:
    # 256 -> 128 (s=2)
    # 128 -> 64  (s=2)
    # 64  -> 32  (s=2)
    # 32  -> 31  (s=1, k=4, p=1) -> (32 - 4 + 2*1)/1 + 1 = 31
    expected_shape = (BATCH_SIZE, 1, 31, 31)
    assert real_logits.shape == expected_shape
    
    print("\nPatchDiscriminator stand-alone file test successful!")