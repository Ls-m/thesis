import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import butter, filtfilt, resample
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import math

# --- Data Analysis ---
def analyze_data(data_dir='bidmc_data/bidmc_csv'):
    excluded_subjects = {13, 15, 19}
    signal_files = [f for f in os.listdir(data_dir) if f.endswith('_Signals.csv')]
    numerics_files = [f for f in os.listdir(data_dir) if f.endswith('_Numerics.csv')]
    
    print(f"Total signal files: {len(signal_files)}")
    print(f"Total numerics files: {len(numerics_files)}")
    
    # Record counts
    for signal_file, numerics_file in zip(signal_files[:5], numerics_files[:5]):  # Check first 5 for brevity
        sig_df = pd.read_csv(os.path.join(data_dir, signal_file))
        num_df = pd.read_csv(os.path.join(data_dir, numerics_file))
        print(f"{signal_file}: {len(sig_df)} records, {numerics_file}: {len(num_df)} records")
    
    # Statistical analysis across subjects
    ppg_means, ppg_stds, rr_means, rr_stds = [], [], [], []
    for subject_id in range(1, 54):
        if subject_id in excluded_subjects:
            continue 
        subject_id_str = f"{subject_id:02d}"
        sig_df = pd.read_csv(os.path.join(data_dir, f'bidmc_{subject_id_str}_Signals.csv'))
        num_df = pd.read_csv(os.path.join(data_dir, f'bidmc_{subject_id_str}_Numerics.csv'))
        sig_df.columns = sig_df.columns.str.strip()
        num_df.columns = num_df.columns.str.strip()
        sig_df = sig_df.apply(pd.to_numeric, errors='coerce')
        num_df = num_df.apply(pd.to_numeric, errors='coerce')
        ppg = sig_df['PLETH'].values
        rr = num_df['RESP'].values
        num_nans = num_df['RESP'].isna().sum().sum()
        nan_indices = num_df[num_df['RESP'].isna()].index
        if num_nans>0:
            print("in file: ",subject_id_str," total Nan values in num: ",num_nans)
            print(nan_indices)
        
        ppg_means.append(np.mean(ppg))
        ppg_stds.append(np.std(ppg))
        rr_means.append(np.mean(rr))
        rr_stds.append(np.std(rr))
    
    print(f"\nPPG Stats - Mean: {np.mean(ppg_means):.4f}, Std: {np.mean(ppg_stds):.4f}")
    print(f"RR Stats - Mean: {np.mean(rr_means):.4f}, Std: {np.mean(rr_stds):.4f}")
    

# --- Preprocessing Functions ---
def normalize_signal(signal):
    return (signal - np.mean(signal)) / np.std(signal)

def bandpass_filter(signal, lowcut=0.5, highcut=5.0, fs=125, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def preprocess_subject(subject_id, data_dir='bidmc_data/bidmc_csv', window_size=30, downsample_fs=25, original_fs=125):
    subject_id_str = f"{subject_id:02d}"
    signal_file = os.path.join(data_dir, f'bidmc_{subject_id_str}_Signals.csv')
    numerics_file = os.path.join(data_dir, f'bidmc_{subject_id_str}_Numerics.csv')
    
    # Load data
    sig_df = pd.read_csv(signal_file)
    num_df = pd.read_csv(numerics_file)
    ppg = sig_df['PLETH'].values
    rr = num_df['RESP'].values
    time_num = num_df['Time [s]'].values
    
    # Preprocess entire signal
    ppg = normalize_signal(ppg)
    ppg = bandpass_filter(ppg, lowcut=0.1, highcut=0.6, fs=original_fs)
    num_samples = int(len(ppg) * downsample_fs / original_fs)
    ppg_ds = resample(ppg, num_samples)
    
    # Segment into windows
    T = int(time_num[-1])
    window_samples = window_size * downsample_fs
    windows, labels = [], []
    
    for t in range(window_size, T + 1):
        start_idx = int((t - window_size) * downsample_fs)
        end_idx = int(t * downsample_fs)
        window = ppg_ds[start_idx:end_idx]
        if len(window) == window_samples:
            label_idx = np.where(time_num == t)[0]
            if len(label_idx) > 0:
                windows.append(window)
                labels.append(rr[label_idx[0]])
    
    return np.array(windows), np.array(labels)

# --- Load and Split Data ---
def load_data(data_dir='bidmc_data/bidmc_csv'):
    subject_data = {}
    excluded_subjects = {13, 15, 19}
    for sid in range(1, 54):
        if sid in excluded_subjects:
            continue 
        windows, labels = preprocess_subject(sid, data_dir)
        subject_data[sid] = (windows, labels)
    
    # Split: 1-40 train, 41-47 val, 48-53 test

    
    
    valid_ids = [sid for sid in range(1, 54) if sid not in excluded_subjects]

    # Define splits (adjust lengths as needed)
    train_ids = valid_ids[:35]    # first 35 valid subjects
    val_ids = valid_ids[35:43]    # next 8
    test_ids = valid_ids[43:]     # remaining
    
    train_data = [subject_data[sid] for sid in train_ids]
    val_data = [subject_data[sid] for sid in val_ids]
    test_data = [subject_data[sid] for sid in test_ids]

    train_windows = np.concatenate([d[0] for d in train_data], axis=0)
    train_labels = np.concatenate([d[1] for d in train_data], axis=0)
    val_windows = np.concatenate([d[0] for d in val_data], axis=0)
    val_labels = np.concatenate([d[1] for d in val_data], axis=0)
    test_windows = np.concatenate([d[0] for d in test_data], axis=0)
    test_labels = np.concatenate([d[1] for d in test_data], axis=0)
    
    return (train_windows, train_labels), (val_windows, val_labels), (test_windows, test_labels)

# --- Adapted Transformer Model ---
class ImprovedTransformer(nn.Module):
    def __init__(self, input_size: int = 750, hidden_size: int = 256, 
                 num_layers: int = 6, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Multi-scale input processing
        self.input_convs = nn.ModuleList([
            nn.Conv1d(1, hidden_size // 4, kernel_size=k, padding=k//2)
            for k in [3, 7, 15, 31]
        ])
        
        self.input_norm = nn.BatchNorm1d(hidden_size)
        self.input_dropout = nn.Dropout(dropout)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(input_size, hidden_size)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Modified output layer for single RR prediction
        self.output_layer = nn.Linear(hidden_size, 1)
        
    def _create_positional_encoding(self, max_len: int, d_model: int):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def forward(self, x):
        # x shape: (B, 1, T), e.g., (B, 1, 750)
        B, _, T = x.shape
        
        # Multi-scale feature extraction
        conv_features = [conv(x) for conv in self.input_convs]
        x = torch.cat(conv_features, dim=1)  # (B, hidden_size, T)
        x = self.input_norm(x)
        x = self.input_dropout(x)
        
        # Transpose for transformer: (B, T, hidden_size)
        x = x.transpose(1, 2)
        
        # Add positional encoding
        if T <= self.pos_encoding.size(1):
            x = x + self.pos_encoding[:, :T, :]
        else:
            pe_extended = self.pos_encoding.repeat(1, (T // self.pos_encoding.size(1)) + 1, 1)
            x = x + pe_extended[:, :T, :]
        
        # Transformer processing
        x = self.transformer(x)  # (B, T, hidden_size)
        
        # Global average pooling and output
        x = x.mean(dim=1)  # (B, hidden_size)
        output = self.output_layer(x)  # (B, 1)
        
        return output

# --- Lightning Module ---
class RRTransformerLightning(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.model = ImprovedTransformer(input_size=750)  # 30s * 25 Hz
        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate
        self.save_hyperparameters()
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

# --- Main Execution ---
if __name__ == "__main__":
    # Step 1: Analyze Data
    print("Analyzing Data...")
    analyze_data()
    
    # Step 2: Load and Preprocess Data
    print("\nLoading and Preprocessing Data...")
    (train_windows, train_labels), (val_windows, val_labels), (test_windows, test_labels) = load_data()
    
    # Convert to tensors and create dataloaders
    train_dataset = TensorDataset(torch.tensor(train_windows, dtype=torch.float32).unsqueeze(1),
                                 torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1))
    val_dataset = TensorDataset(torch.tensor(val_windows, dtype=torch.float32).unsqueeze(1),
                               torch.tensor(val_labels, dtype=torch.float32).unsqueeze(1))
    test_dataset = TensorDataset(torch.tensor(test_windows, dtype=torch.float32).unsqueeze(1),
                                torch.tensor(test_labels, dtype=torch.float32).unsqueeze(1))
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)
    
    # Step 3: Setup Model and Training
    print("\nSetting Up Model and Training...")
    model = RRTransformerLightning()
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=True)
    checkpoint = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, verbose=True,
                                filename='best-checkpoint-{epoch}-{val_loss:.2f}')
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[early_stopping, checkpoint],
        logger=pl.loggers.TensorBoardLogger(save_dir='logs/', name='rr_estimation'),
        log_every_n_steps=10,
        precision=16 if torch.cuda.is_available() else 32  # Mixed precision on GPU
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    # Test
    print("\nTesting Model...")
    trainer.test(model, test_loader, ckpt_path='best')