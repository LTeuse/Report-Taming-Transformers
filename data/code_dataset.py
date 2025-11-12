import torch
from torch.utils.data import Dataset
import os

class CodeDataset(Dataset):
    """
    A simple Dataset class that loads the pre-computed code
    files (.pt) saved by the precompute_codes.py script.
    """
    def __init__(self, codes_dir):
        """
        Args:
            codes_dir (string): Directory containing the .pt code files.
        """
        self.codes_dir = codes_dir
        
        # Get all .pt file paths and sort them numerically
        self.code_files = sorted(
            [os.path.join(codes_dir, f) for f in os.listdir(codes_dir) if f.endswith('.pt')],
            key=lambda f: int(os.path.splitext(os.path.basename(f))[0])
        )

    def __len__(self):
        return len(self.code_files)

    def __getitem__(self, idx):
        """
        Loads a code file, converts it to a flat sequence of
        long integers (which is what the Transformer expects).
        """
        # Load the code grid, e.g., shape [8, 8]
        code_grid = torch.load(self.code_files[idx])
        
        # Flatten the grid into a 1D sequence, e.g., shape [64]
        code_sequence = code_grid.view(-1)
        
        # Convert to long, as required by nn.Embedding and CrossEntropyLoss
        return code_sequence.long()