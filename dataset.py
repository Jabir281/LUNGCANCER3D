import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class LungCancerDataset(Dataset):
    """
    Custom Dataset for 3D Lung Cancer patches.
    
    Expected filename format:
        - {uid}_pos_{i}.npy for cancer samples (label = 1)
        - {uid}_neg_{i}.npy for non-cancer samples (label = 0)
    
    Args:
        data_dir (str): Directory containing .npy files
        transform (callable, optional): Optional transform to apply
    """
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Find all .npy files
        self.file_list = list(self.data_dir.rglob("*.npy"))
        
        if len(self.file_list) == 0:
            print(f"Warning: No .npy files found in {data_dir}")
        
        # Extract labels from filenames
        self.labels = []
        for file_path in self.file_list:
            filename = file_path.stem  # Get filename without extension
            if "_pos_" in filename:
                self.labels.append(1)
            elif "_neg_" in filename:
                self.labels.append(0)
            else:
                # Default to negative or raise error? Raising error is safer for data integrity
                raise ValueError(f"Invalid filename format: {file_path.name}. "
                               "Expected '_pos_' or '_neg_' in filename.")
        
        if self.file_list:
            print(f"Loaded {len(self.file_list)} samples from {data_dir}")
            print(f"Positive samples: {sum(self.labels)}, Negative samples: {len(self.labels) - sum(self.labels)}")
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # Load the 3D patch
        file_path = self.file_list[idx]
        try:
            patch = np.load(file_path).astype(np.float32)
        except Exception as e:
            raise IOError(f"Error loading file {file_path}: {e}")
        
        # Validate shape
        if patch.shape != (64, 64, 64):
            raise ValueError(f"Expected shape (64, 64, 64), got {patch.shape} for {file_path.name}")
        
        # Add channel dimension: (64, 64, 64) -> (1, 64, 64, 64)
        patch = np.expand_dims(patch, axis=0)
        
        # Convert to torch tensor
        patch = torch.from_numpy(patch)
        
        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        # Apply transforms if any
        if self.transform:
            patch = self.transform(patch)
        
        return patch, label
    
    def get_class_distribution(self):
        """Returns the distribution of classes in the dataset."""
        num_pos = sum(self.labels)
        num_neg = len(self.labels) - num_pos
        return {"positive": num_pos, "negative": num_neg}
