import os
import glob
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt, resample
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, message='Failed to initialize NumPy: No module named \'numpy\'')
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

# --- 1. Configuration ---
class Config:
    """Configuration class for all hyperparameters and settings."""
    # Data parameters
    DATA_DIR = './bidmc_data/bidmc_csv/'
    SUBJECTS_TO_EXCLUDE = [13, 15, 19]  # Subjects with known NaN issues in RESP

    # Signal processing parameters
    ORIG_FS = 125  # Hz
    NEW_FS = 25    # Hz
    WINDOW_SEC = 32  # seconds
    STEP_SEC = 16    # seconds (32-8 = 24s overlap, i.e., 75%)
    FILTER_LOWCUT = 0.1  # Hz
    FILTER_HIGHCUT = 0.8  # Hz
    FILTER_ORDER = 2

    # Calculated parameters
    WINDOW_SAMPLES = WINDOW_SEC * NEW_FS
    STEP_SAMPLES = STEP_SEC * NEW_FS

    # Model hyperparameters
    MODEL_HIDDEN_SIZE = 256
    MODEL_NUM_LAYERS = 4
    MODEL_NUM_HEADS = 8
    MODEL_DROPOUT = 0.2

    # Training parameters
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    EPOCHS = 100
    PATIENCE = 10  # For early stopping
    NUM_WORKERS = 4  # For DataLoader

# --- 2. Exploratory Data Analysis (EDA) ---
def analyze_data(config: Config):
    """
    Performs EDA on the BIDMC dataset.
    - Prints file counts and shapes.
    - Shows distribution of Respiratory Rates.
    """
    print("--- 🩺 Starting Data Analysis 🩺 ---")
    signals_files = sorted(glob.glob(os.path.join(config.DATA_DIR, '*_Signals.csv')))
    numerics_files = sorted(glob.glob(os.path.join(config.DATA_DIR, '*_Numerics.csv')))

    print(f"Found {len(signals_files)} Signals files and {len(numerics_files)} Numerics files.")

    all_rr_labels = []
    subjects_with_nan = []

    for numerics_file in tqdm(numerics_files, desc="Analyzing subjects"):
        subject_id = int(os.path.basename(numerics_file).split('_')[1])
        if subject_id in config.SUBJECTS_TO_EXCLUDE:
            print(f"Skipping excluded subject: {subject_id}")
            continue

        try:
            df = pd.read_csv(numerics_file)
            df.columns = df.columns.str.strip()
            rr_values = df['RESP'].dropna()

            if df['RESP'].isnull().any():
                subjects_with_nan.append(subject_id)

            all_rr_labels.extend(rr_values.tolist())
        except Exception as e:
            print(f"Could not process {numerics_file}: {e}")

    print(f"\nSubjects with any NaN in RESP column: {subjects_with_nan}")
    print("\n--- Respiratory Rate (RR) Distribution ---")
    rr_series = pd.Series(all_rr_labels)
    print(rr_series.describe())

    plt.figure(figsize=(12, 6))
    sns.histplot(rr_series, bins=50, kde=True)
    plt.title('Distribution of Respiratory Rates (breaths/minute) Across All Subjects')
    plt.xlabel('Respiratory Rate (breaths/minute)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    print("--- ✅ Data Analysis Complete ✅ ---\n")
    return rr_series.mean(), rr_series.std()

# --- 3. Preprocessing & PyTorch Dataset ---
def bandpass_filter(data, lowcut, highcut, fs, order=2):
    """Applies a Butterworth bandpass filter."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

class PPGDataset(Dataset):
    """PyTorch Dataset for PPG windows and RR labels."""
    def __init__(self, windows, labels):
        self.windows = windows
        self.labels = labels

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        label = self.labels[idx]
        window_tensor = torch.from_numpy(window).float().unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.float)
        return window_tensor, label_tensor

class PPGDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for handling the PPG dataset."""
    def __init__(self, config: Config, label_mean: float, label_std: float):
        super().__init__()
        self.config = config
        self.label_mean = label_mean
        self.label_std = label_std
        self.all_windows = []
        self.all_labels = []
        self._data_prepared = False

    def prepare_data(self):
        if self._data_prepared:
            return

        print("--- 🔄 Preparing Data 🔄 ---")
        signals_files = sorted(glob.glob(os.path.join(self.config.DATA_DIR, '*_Signals.csv')))

        for signal_file in tqdm(signals_files, desc="Processing Subjects"):
            subject_id = int(os.path.basename(signal_file).split('_')[1])
            if subject_id in self.config.SUBJECTS_TO_EXCLUDE:
                continue

            numerics_file = signal_file.replace('_Signals.csv', '_Numerics.csv')

            try:
                signals_df = pd.read_csv(signal_file)
                numerics_df = pd.read_csv(numerics_file)
                signals_df.columns = signals_df.columns.str.strip()
                numerics_df.columns = numerics_df.columns.str.strip()

                ppg_signal = signals_df['PLETH'].values
                rr_labels = numerics_df['RESP'].values
                rr_labels = pd.Series(rr_labels).ffill().bfill().values

                ppg_filtered = bandpass_filter(ppg_signal, self.config.FILTER_LOWCUT,
                                               self.config.FILTER_HIGHCUT, self.config.ORIG_FS,
                                               order=self.config.FILTER_ORDER)
                num_samples_new = int(len(ppg_filtered) * self.config.NEW_FS / self.config.ORIG_FS)
                ppg_downsampled = resample(ppg_filtered, num_samples_new)
                num_windows = (len(ppg_downsampled) - self.config.WINDOW_SAMPLES) // self.config.STEP_SAMPLES + 1

                for i in range(num_windows):
                    start_idx = i * self.config.STEP_SAMPLES
                    end_idx = start_idx + self.config.WINDOW_SAMPLES
                    window = ppg_downsampled[start_idx:end_idx]
                    window = (window - np.mean(window)) / np.std(window)
                    start_time_sec = start_idx / self.config.NEW_FS
                    end_time_sec = end_idx / self.config.NEW_FS
                    relevant_labels = rr_labels[math.floor(start_time_sec):math.ceil(end_time_sec)]
                    if len(relevant_labels) > 0:
                        label = np.mean(relevant_labels)
                        normalized_label = (label - self.label_mean) / self.label_std
                        self.all_windows.append(window)
                        self.all_labels.append(normalized_label)

            except Exception as e:
                print(f"Error processing subject {subject_id}: {e}")

        self.all_windows = np.array(self.all_windows)
        self.all_labels = np.array(self.all_labels)
        self._data_prepared = True
        print(f"--- ✅ Data Preparation Complete. Generated {len(self.all_windows)} windows. ✅ ---")

    def setup(self, stage: str = None):
        indices = np.arange(len(self.all_windows))
        train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
        train_indices, val_indices = train_test_split(train_indices, test_size=0.15, random_state=42)

        if stage == 'fit' or stage is None:
            self.train_dataset = PPGDataset(self.all_windows[train_indices], self.all_labels[train_indices])
            self.val_dataset = PPGDataset(self.all_windows[val_indices], self.all_labels[val_indices])
            print(f"Train samples: {len(self.train_dataset)}, Val samples: {len(self.val_dataset)}")

        if stage == 'test' or stage is None:
            self.test_dataset = PPGDataset(self.all_windows[test_indices], self.all_labels[test_indices])
            print(f"Test samples: {len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True, num_workers=self.config.NUM_WORKERS, persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False, num_workers=self.config.NUM_WORKERS, persistent_workers=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False, num_workers=self.config.NUM_WORKERS, persistent_workers=True, pin_memory=True)

# --- 4. UNet for Denoising ---
class MultiScaleConv(nn.Module):
    """Multi-scale 1D convolution module."""
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 7, 15, 31]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
    
    def forward(self, x):
        features = [conv(x) for conv in self.convs]
        return torch.cat(features, dim=1)

class PPGDenoisingUNet(nn.Module):
    """UNet-like model for PPG signal denoising."""
    def __init__(self, hidden_size=256):
        super().__init__()
        # Encoder
        self.enc1 = MultiScaleConv(1, hidden_size)
        self.down1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1)
        self.enc2 = MultiScaleConv(hidden_size, hidden_size)
        self.down2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1)
        self.bottleneck = MultiScaleConv(hidden_size, hidden_size)
        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.dec1 = MultiScaleConv(hidden_size * 2, hidden_size)
        self.up2 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.dec2 = MultiScaleConv(hidden_size * 2, hidden_size)
        self.out = nn.Conv1d(hidden_size, 1, kernel_size=1)
    
    def forward(self, x):
        e1 = self.enc1(x)    # (B, hidden_size, 800)
        d1 = self.down1(e1)  # (B, hidden_size, 400)
        e2 = self.enc2(d1)   # (B, hidden_size, 400)
        d2 = self.down2(e2)  # (B, hidden_size, 200)
        b = self.bottleneck(d2)  # (B, hidden_size, 200)
        u1 = self.up1(b)     # (B, hidden_size, 400)
        concat1 = torch.cat([u1, e2], dim=1)  # (B, hidden_size*2, 400)
        d1 = self.dec1(concat1)  # (B, hidden_size, 400)
        u2 = self.up2(d1)    # (B, hidden_size, 800)
        concat2 = torch.cat([u2, e1], dim=1)  # (B, hidden_size*2, 800)
        d2 = self.dec2(concat2)  # (B, hidden_size, 800)
        out = self.out(d2)   # (B, 1, 800)
        return out

class DenoisingUNet(pl.LightningModule):
    """LightningModule for training the denoising UNet."""
    def __init__(self, hidden_size=256, learning_rate=1e-3):
        super().__init__()
        self.model = PPGDenoisingUNet(hidden_size)
        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        noisy, clean = batch
        denoised = self(noisy)
        loss = self.criterion(denoised, clean)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        noisy, clean = batch
        denoised = self(noisy)
        loss = self.criterion(denoised, clean)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

class DenoisingDataset(Dataset):
    """Dataset for denoising task: adds noise to clean PPG windows."""
    def __init__(self, clean_windows, noise_std=0.5):
        self.clean_windows = clean_windows
        self.noise_std = noise_std
    
    def __len__(self):
        return len(self.clean_windows)
    
    def __getitem__(self, idx):
        clean = self.clean_windows[idx]
        noisy = clean + np.random.normal(0, self.noise_std, clean.shape)
        return torch.tensor(noisy, dtype=torch.float).unsqueeze(0), torch.tensor(clean, dtype=torch.float).unsqueeze(0)

# --- 5. Model Definition (Adapted for Regression) ---
class RRTransformer(nn.Module):
    """Transformer model adapted for single-value regression (RR estimation)."""
    def __init__(self, input_size: int, hidden_size: int = 256,
                 num_layers: int = 4, num_heads: int = 8, dropout: float = 0.1,
                 pretrained_path: str = None):
        super().__init__()
        self.hidden_size = hidden_size

        # Multi-scale 1D CNN Feature Extractor
        self.input_convs = nn.ModuleList([
            nn.Conv1d(1, hidden_size // 4, kernel_size=k, padding=k//2)
            for k in [3, 7, 15, 31]
        ])
        self.input_norm = nn.BatchNorm1d(hidden_size)
        self.input_dropout = nn.Dropout(dropout)

        # Load pre-trained weights if provided
        if pretrained_path:
            unet_state_dict = torch.load(pretrained_path)
            for i, conv in enumerate(self.input_convs):
                conv_state_dict = {
                    'weight': unet_state_dict[f'enc1.convs.{i}.weight'],
                    'bias': unet_state_dict[f'enc1.convs.{i}.bias']
                }
                conv.load_state_dict(conv_state_dict)
            print(f"Loaded pre-trained weights from {pretrained_path}")

        # Positional Encoding
        self.pos_encoding = self._create_positional_encoding(input_size, hidden_size)

        # Transformer Encoder
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

        # Regression Head
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def _create_positional_encoding(self, max_len: int, d_model: int):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(self, x):
        B, _, T = x.shape
        conv_features = [conv(x) for conv in self.input_convs]
        x = torch.cat(conv_features, dim=1)  # (B, hidden_size, T)
        x = self.input_norm(x)
        x = self.input_dropout(x)
        x = x.transpose(1, 2)  # -> (B, T, hidden_size)
        x = x + self.pos_encoding[:, :T, :]
        x = self.transformer(x)  # (B, T, hidden_size)
        x = torch.mean(x, dim=1)  # (B, hidden_size)
        output = self.regression_head(x)  # (B, 1)
        return output.squeeze(-1)  # (B,)

# --- 6. PyTorch Lightning Training Module ---
class RREstimator(pl.LightningModule):
    """The LightningModule that orchestrates the training."""
    def __init__(self, config: Config, label_mean: float, label_std: float, pretrained_path: str = None):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.label_mean = label_mean
        self.label_std = label_std

        self.model = RRTransformer(
            input_size=config.WINDOW_SAMPLES,
            hidden_size=config.MODEL_HIDDEN_SIZE,
            num_layers=config.MODEL_NUM_LAYERS,
            num_heads=config.MODEL_NUM_HEADS,
            dropout=config.MODEL_DROPOUT,
            pretrained_path=pretrained_path
        )
        self.loss_fn = nn.MSELoss()
        self.mae = torchmetrics.MeanAbsoluteError()

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, batch_idx):
        x, y_norm = batch
        y_pred_norm = self(x)
        loss = self.loss_fn(y_pred_norm, y_norm)
        y_true = (y_norm * self.label_std) + self.label_mean
        y_pred = (y_pred_norm * self.label_std) + self.label_mean
        mae_val = self.mae(y_pred, y_true)
        return loss, mae_val

    def training_step(self, batch, batch_idx):
        loss, mae = self._shared_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mae', mae, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, mae = self._shared_step(batch, batch_idx)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_mae', mae, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, mae = self._shared_step(batch, batch_idx)
        self.log('test_loss', loss)
        self.log('test_mae', mae)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.LEARNING_RATE)
        return optimizer

# --- 7. Main Execution Block ---
if __name__ == '__main__':
    # Initialize Configuration
    config = Config()

    # Step 1: Analyze Data
    rr_mean, rr_std = analyze_data(config)
    
    # Step 2: Setup DataModule
    datamodule = PPGDataModule(config, label_mean=rr_mean, label_std=rr_std)
    datamodule.prepare_data()
    datamodule.setup()

    # Step 3: Train UNet for Denoising
    print("\n--- Training UNet for Denoising ---")
    noise_std = 0.5  # Noise standard deviation (adjustable)
    denoising_train_dataset = DenoisingDataset(datamodule.train_dataset.windows, noise_std)
    denoising_val_dataset = DenoisingDataset(datamodule.val_dataset.windows, noise_std)
    denoising_train_loader = DataLoader(denoising_train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, persistent_workers=True, pin_memory=True)
    denoising_val_loader = DataLoader(denoising_val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, persistent_workers=True, pin_memory=True)

    unet_model = DenoisingUNet(hidden_size=config.MODEL_HIDDEN_SIZE, learning_rate=config.LEARNING_RATE)
    unet_trainer = pl.Trainer(
        max_epochs=20,  # Number of epochs for UNet training
        accelerator='auto',
        logger=TensorBoardLogger("tb_logs", name="unet_denoising"),
    )
    unet_trainer.fit(unet_model, denoising_train_loader, denoising_val_loader)
    torch.save(unet_model.model.state_dict(), 'unet_denoising.pth')
    print("--- UNet Training Complete ---")

    # Step 4: Setup Transformer Model with Pre-trained Weights
    pretrained_path = 'unet_denoising.pth'
    model = RREstimator(config, label_mean=rr_mean, label_std=rr_std, pretrained_path=pretrained_path)

    # Step 5: Setup Callbacks and Logger
    logger = TensorBoardLogger("tb_logs", name="rr_transformer")
    early_stop_callback = EarlyStopping(
        monitor='val_mae',
        patience=config.PATIENCE,
        verbose=True,
        mode='min'
    )
    checkpoint_callback = ModelCheckpoint(
        monitor='val_mae',
        dirpath='checkpoints',
        filename='best-model-{epoch:02d}-{val_mae:.2f}',
        save_top_k=1,
        mode='min'
    )

    # Step 6: Initialize and Run Trainer for Transformer
    trainer = pl.Trainer(
        max_epochs=config.EPOCHS,
        accelerator='auto',
        logger=logger,
        callbacks=[early_stop_callback, checkpoint_callback],
        log_every_n_steps=10
    )

    print("\n--- 🚀 Starting Transformer Model Training 🚀 ---")
    trainer.fit(model, datamodule)
    print("--- ✅ Training Finished ✅ ---")

    # Step 7: Test the Best Model
    print("\n--- 🧪 Starting Model Testing 🧪 ---")
    test_results = trainer.test(model, datamodule=datamodule, ckpt_path='best')
    print("\n--- Test Results ---")
    print(test_results)
    print("--- ✅ Testing Complete ✅ ---")

    print("\nTo view training logs, run the following command in your terminal:")
    print(f"tensorboard --logdir {os.getcwd()}/tb_logs")