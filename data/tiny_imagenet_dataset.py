import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class TinyImageNetDataset(Dataset):
    """
    A PyTorch Dataset class for the Tiny ImageNet dataset.
    
    EDITED: This version is simplified for Stage 1 VQGAN training.
    It does *not* load or return class labels, only the images.
    """
    def __init__(self, root_dir, split='train', size=64, random_flip=True):
        """
        Args:
            root_dir (string): The root directory of the Tiny ImageNet dataset (e.g., 'data/').
            split (string): The dataset split, 'train' or 'val'.
            size (int): The desired size for the images after resizing.
                        Tiny ImageNet's native size is 64x64.
            random_flip (bool): Whether to apply random horizontal flipping.
        """
        self.root_dir = os.path.join(root_dir, 'tiny-imagenet-200')
        self.split = split
        self.image_paths = []

        # --- Label loading is not needed for unconditional VQGAN training ---
        # We just need to gather all image paths.

        if self.split == 'train':
            train_dir = os.path.join(self.root_dir, 'train')
            # Training images are in class-specific subfolders
            for class_folder in os.listdir(train_dir):
                class_dir = os.path.join(train_dir, class_folder, 'images')
                if not os.path.isdir(class_dir):
                    continue
                for img_name in os.listdir(class_dir):
                    self.image_paths.append(os.path.join(class_dir, img_name))
        
        elif self.split == 'val':
            val_dir = os.path.join(self.root_dir, 'val', 'images')
            annotations_file = os.path.join(self.root_dir, 'val', 'val_annotations.txt')
            
            # Read annotations to get image filenames
            val_annotations = pd.read_csv(annotations_file, sep='\t', header=None, names=['File', 'WNID', 'X1', 'Y1', 'X2', 'Y2'])
            
            for img_name in val_annotations['File']:
                self.image_paths.append(os.path.join(val_dir, img_name))

        # Define image transformations
        transform_list = [
            transforms.Resize((size, size)),
        ]
        if random_flip:
            transform_list.append(transforms.RandomHorizontalFlip())

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize to [-1, 1]
        ])
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Error loading image {img_path}. Skipping. Error: {e}")
            # Return a dummy image or try the next one
            return self.__getitem__((idx + 1) % len(self)) 
            
        image = self.transform(image)
        
        # For Stage 1 (VQGAN), we only need the image.
        return image