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
# ===================================================================
class Config:
    """Configuration class for all hyperparameters and settings."""
    # Data parameters
    DATA_DIR = './bidmc_data/bidmc_csv/'
    SUBJECTS_TO_EXCLUDE = [13, 15, 19] # Subjects with known NaN issues in RESP

    # Signal processing parameters
    ORIG_FS = 125  # Hz
    NEW_FS = 25    # Hz
    WINDOW_SEC = 32 # seconds
    STEP_SEC = 16    # seconds (32-8 = 24s overlap, i.e., 75%)
    FILTER_LOWCUT = 0.1 # Hz
    FILTER_HIGHCUT = 0.8 # Hz
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
    PATIENCE = 10 # For early stopping
    NUM_WORKERS = 4 # For DataLoader


# --- 2. Exploratory Data Analysis (EDA) ---
# ===================================================================
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
            # Clean column names by stripping whitespace
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

    # Plot distribution
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
# ===================================================================
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

        # Convert to torch tensors
        # Add channel dimension for Conv1d: (seq_len,) -> (1, seq_len)
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
        # --- FIX: Add a flag to track if data has been prepared ---
        self._data_prepared = False

    def prepare_data(self):
        """
        Loads and preprocesses data. Called only on 1 GPU.
        This is where we perform filtering, downsampling, and windowing.
        """
        # --- FIX: Check the flag here. If data is already prepared, do nothing. ---
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
                # Load signals and numerics
                signals_df = pd.read_csv(signal_file)
                numerics_df = pd.read_csv(numerics_file)
                signals_df.columns = signals_df.columns.str.strip()
                numerics_df.columns = numerics_df.columns.str.strip()

                ppg_signal = signals_df['PLETH'].values
                rr_labels = numerics_df['RESP'].values

                # Handle potential NaNs in labels by forward/backward fill
                rr_labels = pd.Series(rr_labels).ffill().bfill().values

                # 1. Bandpass Filtering
                ppg_filtered = bandpass_filter(ppg_signal, self.config.FILTER_LOWCUT,
                                               self.config.FILTER_HIGHCUT, self.config.ORIG_FS,
                                               order=self.config.FILTER_ORDER)

                # 2. Downsampling
                num_samples_new = int(len(ppg_filtered) * self.config.NEW_FS / self.config.ORIG_FS)
                ppg_downsampled = resample(ppg_filtered, num_samples_new)

                # 3. Segmentation with Overlap
                num_windows = (len(ppg_downsampled) - self.config.WINDOW_SAMPLES) // self.config.STEP_SAMPLES + 1

                for i in range(num_windows):
                    start_idx = i * self.config.STEP_SAMPLES
                    end_idx = start_idx + self.config.WINDOW_SAMPLES

                    # Get PPG window
                    window = ppg_downsampled[start_idx:end_idx]

                    # 3a. Z-score normalization per window
                    window = (window - np.mean(window)) / np.std(window)

                    # 3b. Get corresponding label
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
        
        # --- FIX: Set the flag to True after preparation is complete ---
        self._data_prepared = True
        print(f"--- ✅ Data Preparation Complete. Generated {len(self.all_windows)} windows. ✅ ---")

    def setup(self, stage: str = None):
        """Splits data into train, val, test. Called on every GPU."""
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
        return DataLoader(self.train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True, num_workers=self.config.NUM_WORKERS,persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False, num_workers=self.config.NUM_WORKERS,persistent_workers=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False, num_workers=self.config.NUM_WORKERS, persistent_workers=True, pin_memory=True)
    
# --- 4. Model Definition (Adapted for Regression) ---
# ===================================================================
class RRTransformer(nn.Module):
    """
    Transformer model adapted for single-value regression (RR estimation).
    """
    def __init__(self, input_size: int, hidden_size: int = 256,
                 num_layers: int = 4, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size

        # 1. Multi-scale 1D CNN Feature Extractor (as in original)
        self.input_convs = nn.ModuleList([
            nn.Conv1d(1, hidden_size // 4, kernel_size=k, padding=k//2)
            for k in [3, 7, 15, 31]
        ])
        self.input_norm = nn.BatchNorm1d(hidden_size)
        self.input_dropout = nn.Dropout(dropout)

        # 2. Positional Encoding
        self.pos_encoding = self._create_positional_encoding(input_size, hidden_size)

        # 3. Transformer Encoder (as in original)
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

        # 4. Regression Head (MODIFIED PART)
        # This replaces the original decoder. It pools the transformer output
        # and passes it through a simple MLP to get a single value.
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
        # x shape: (B, 1, T)
        B, _, T = x.shape

        # Multi-scale feature extraction
        conv_features = [conv(x) for conv in self.input_convs]
        x = torch.cat(conv_features, dim=1)  # (B, hidden_size, T)
        x = self.input_norm(x)
        x = self.input_dropout(x)
        x = x.transpose(1, 2)  # -> (B, T, hidden_size)

        # Add positional encoding
        x = x + self.pos_encoding[:, :T, :]

        # Transformer processing
        x = self.transformer(x)  # (B, T, hidden_size)

        # --- MODIFICATION FOR REGRESSION ---
        # Pool features over the time dimension to get a single vector per sample
        x = torch.mean(x, dim=1)  # (B, hidden_size)

        # Pass through the regression head to get the final prediction
        output = self.regression_head(x)  # (B, 1)

        return output.squeeze(-1) # -> (B,) for loss calculation


# --- 5. PyTorch Lightning Training Module ---
# ===================================================================
class RREstimator(pl.LightningModule):
    """The LightningModule that orchestrates the training."""
    def __init__(self, config: Config, label_mean: float, label_std: float):
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
            dropout=config.MODEL_DROPOUT
        )
        self.loss_fn = nn.MSELoss()

        # Metrics
        self.mae = torchmetrics.MeanAbsoluteError()

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, batch_idx):
        x, y_norm = batch
        y_pred_norm = self(x)
        loss = self.loss_fn(y_pred_norm, y_norm)

        # De-normalize for interpretable metrics
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


# --- 6. Main Execution Block ---
# ===================================================================
if __name__ == '__main__':
    # Set to 'fork' for multiprocessing on macOS/some Linux distros, not needed on others
    # torch.multiprocessing.set_start_method('fork')
    
    # Initialize Configuration
    config = Config()

    # --- Step 1: Analyze Data ---
    # This also gets the mean/std for label normalization
    rr_mean, rr_std = analyze_data(config)
    
    # --- Step 2: Setup DataModule ---
    datamodule = PPGDataModule(config, label_mean=rr_mean, label_std=rr_std)
    datamodule.prepare_data() # Manually call prepare_data once
    datamodule.setup() # Call setup to create splits

    # --- Step 3: Setup Model ---
    model = RREstimator(config, label_mean=rr_mean, label_std=rr_std)

    # --- Step 4: Setup Callbacks and Logger ---
    # For TensorBoard
    logger = TensorBoardLogger("tb_logs", name="rr_transformer")

    # For Early Stopping
    early_stop_callback = EarlyStopping(
        monitor='val_mae',
        patience=config.PATIENCE,
        verbose=True,
        mode='min'
    )

    # For saving the best model
    checkpoint_callback = ModelCheckpoint(
        monitor='val_mae',
        dirpath='checkpoints',
        filename='best-model-{epoch:02d}-{val_mae:.2f}',
        save_top_k=1,
        mode='min'
    )

    # --- Step 5: Initialize and Run Trainer ---
    trainer = pl.Trainer(
        max_epochs=config.EPOCHS,
        accelerator='auto', # Automatically uses GPU if available
        logger=logger,
        callbacks=[early_stop_callback, checkpoint_callback],
        log_every_n_steps=10
    )

    print("\n--- 🚀 Starting Model Training 🚀 ---")
    trainer.fit(model, datamodule)
    print("--- ✅ Training Finished ✅ ---")

    # --- Step 6: Test the Best Model ---
    print("\n--- 🧪 Starting Model Testing 🧪 ---")
    # The trainer.test() method automatically loads the best checkpoint
    test_results = trainer.test(model, datamodule=datamodule, ckpt_path='best')
    print("\n--- Test Results ---")
    print(test_results)
    print("--- ✅ Testing Complete ✅ ---")

    # To launch TensorBoard, run this in your terminal:
    print("\nTo view training logs, run the following command in your terminal:")
    print(f"tensorboard --logdir {os.getcwd()}/tb_logs")