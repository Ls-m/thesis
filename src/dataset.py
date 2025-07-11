import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple


class PPGRespiratoryDataset(Dataset):
    """PyTorch Dataset for PPG to respiratory waveform estimation."""
    
    def __init__(self, ppg_data: np.ndarray, resp_data: np.ndarray):
        """
        Initialize the dataset.
        
        Args:
            ppg_data: PPG signal segments of shape (n_samples, sequence_length)
            resp_data: Respiratory signal segments of shape (n_samples, sequence_length)
        """
        # Check for NaN values in input data
        ppg_nan_count = np.isnan(ppg_data).sum()
        resp_nan_count = np.isnan(resp_data).sum()
        
        if ppg_nan_count > 0 or resp_nan_count > 0:
            print(f"Warning: Found {ppg_nan_count} NaN values in PPG data and {resp_nan_count} NaN values in respiratory data")
            # Replace NaN values with 0
            ppg_data = np.nan_to_num(ppg_data, nan=0.0, posinf=0.0, neginf=0.0)
            resp_data = np.nan_to_num(resp_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Check for infinite values
        ppg_inf_count = np.isinf(ppg_data).sum()
        resp_inf_count = np.isinf(resp_data).sum()
        
        if ppg_inf_count > 0 or resp_inf_count > 0:
            print(f"Warning: Found {ppg_inf_count} Inf values in PPG data and {resp_inf_count} Inf values in respiratory data")
            # Replace Inf values with 0
            ppg_data = np.nan_to_num(ppg_data, nan=0.0, posinf=0.0, neginf=0.0)
            resp_data = np.nan_to_num(resp_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Clip extreme values
        ppg_data = np.clip(ppg_data, -10.0, 10.0)
        resp_data = np.clip(resp_data, -10.0, 10.0)
        
        self.ppg_data = torch.FloatTensor(ppg_data)
        self.resp_data = torch.FloatTensor(resp_data)
        
        # Add channel dimension for 1D CNN
        if len(self.ppg_data.shape) == 2:
            self.ppg_data = self.ppg_data.unsqueeze(1)  # (batch, 1, sequence)
        if len(self.resp_data.shape) == 2:
            self.resp_data = self.resp_data.unsqueeze(1)  # (batch, 1, sequence)
        
        # Final check for NaN/Inf in tensors
        if torch.isnan(self.ppg_data).any() or torch.isnan(self.resp_data).any():
            print("Error: NaN values still present in tensors after cleaning")
        if torch.isinf(self.ppg_data).any() or torch.isinf(self.resp_data).any():
            print("Error: Inf values still present in tensors after cleaning")
    
    def __len__(self) -> int:
        return len(self.ppg_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.ppg_data[idx], self.resp_data[idx]


def create_data_loaders(fold_data: dict, batch_size: int, num_workers: int = 4) -> dict:
    """Create PyTorch DataLoaders for train, validation, and test sets."""
    
    # Create datasets
    train_dataset = PPGRespiratoryDataset(
        fold_data['train_ppg'], 
        fold_data['train_resp']
    )
    
    val_dataset = PPGRespiratoryDataset(
        fold_data['val_ppg'], 
        fold_data['val_resp']
    )
    
    test_dataset = PPGRespiratoryDataset(
        fold_data['test_ppg'], 
        fold_data['test_resp']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
