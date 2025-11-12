import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class AFHQDataset(Dataset):
    """
    A PyTorch Dataset class for the AFHQ (v2) dataset.
    This is simplified for unconditional VQGAN training and only loads images.
    """
    def __init__(self, root_dir, split='train', size=256, random_flip=True):
        """
        Args:
            root_dir (string): The root directory of the AFHQ dataset (e.g., 'data/').
            split (string): The dataset split, 'train' or 'val'.
            size (int): The desired size for the images (e.g., 256).
            random_flip (bool): Whether to apply random horizontal flipping.
        """
        self.split_dir = os.path.join(root_dir, 'afhq', split)
        self.image_paths = []

        # Gather all image paths
        for category in ['cat', 'dog', 'wild']:
            category_dir = os.path.join(self.split_dir, category)
            if not os.path.isdir(category_dir):
                print(f"Warning: Directory not found {category_dir}")
                continue
                
            for img_name in os.listdir(category_dir):
                if img_name.endswith('.jpg') or img_name.endswith('.png'):
                    self.image_paths.append(os.path.join(category_dir, img_name))

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
            # Return the next image as a fallback
            return self.__getitem__((idx + 1) % len(self)) 
            
        image = self.transform(image)
        
        # For Stage 1 (VQGAN), we only need the image.
        return image