import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RWKVBlock(nn.Module):
    """RWKV block for time series processing."""
    
    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff or 4 * d_model
        
        # Time mixing (attention-like mechanism)
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, d_model))
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, d_model))
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, d_model))
        
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        
        # Channel mixing (FFN-like mechanism)
        self.channel_mix_k = nn.Parameter(torch.ones(1, 1, d_model))
        self.channel_mix_r = nn.Parameter(torch.ones(1, 1, d_model))
        
        self.key_c = nn.Linear(d_model, self.d_ff, bias=False)
        self.receptance_c = nn.Linear(d_model, d_model, bias=False)
        self.value_c = nn.Linear(self.d_ff, d_model, bias=False)
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, state=None):
        B, T, C = x.shape
        
        # Time mixing
        x_tm = self.ln1(x)
        
        if state is None:
            # Initialize state for the first step
            state = torch.zeros(B, C, device=x.device, dtype=x.dtype)
        
        # Shift and mix with previous state
        x_shifted = torch.cat([state.unsqueeze(1), x_tm[:, :-1]], dim=1)
        
        k = self.key(x_tm * self.time_mix_k + x_shifted * (1 - self.time_mix_k))
        v = self.value(x_tm * self.time_mix_v + x_shifted * (1 - self.time_mix_v))
        r = self.receptance(x_tm * self.time_mix_r + x_shifted * (1 - self.time_mix_r))
        
        # WKV computation (simplified RWKV attention)
        wkv = self._compute_wkv(k, v, r)
        x = x + self.dropout(self.output(wkv))
        
        # Channel mixing
        x_cm = self.ln2(x)
        x_shifted_c = torch.cat([state.unsqueeze(1), x_cm[:, :-1]], dim=1)
        
        k_c = self.key_c(x_cm * self.channel_mix_k + x_shifted_c * (1 - self.channel_mix_k))
        r_c = self.receptance_c(x_cm * self.channel_mix_r + x_shifted_c * (1 - self.channel_mix_r))
        
        vk = self.value_c(F.relu(k_c) ** 2)
        x = x + self.dropout(torch.sigmoid(r_c) * vk)
        
        # Update state for next step
        new_state = x[:, -1]
        
        return x, new_state
    
    def _compute_wkv(self, k, v, r):
        """Compute WKV (weighted key-value) attention."""
        B, T, C = k.shape
        
        # Simplified WKV computation
        # In practice, this would use more efficient algorithms
        w = torch.softmax(k, dim=-1)
        wkv = torch.sigmoid(r) * (w * v)
        
        return wkv


class RWKV(nn.Module):
    """RWKV model for PPG to respiratory waveform estimation."""
    
    def __init__(self, input_size: int, hidden_size: int = 256, 
                 num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv1d(1, hidden_size // 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Conv1d(hidden_size // 2, hidden_size, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # RWKV blocks
        self.rwkv_blocks = nn.ModuleList([
            RWKVBlock(hidden_size, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1)
        )
        
        self.ln_out = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        # x shape: (batch_size, 1, sequence_length)
        
        # Input feature extraction
        x = self.input_proj(x)  # (B, hidden_size, T)
        x = x.transpose(1, 2)   # (B, T, hidden_size)
        
        # RWKV processing
        state = None
        for rwkv_block in self.rwkv_blocks:
            x, state = rwkv_block(x, state)
        
        x = self.ln_out(x)
        
        # Output projection
        output = self.output_proj(x)  # (B, T, 1)
        output = output.transpose(1, 2)  # (B, 1, T)
        
        return output


class ImprovedTransformer(nn.Module):
    """Improved Transformer model with better architecture for signal processing."""
    
    def __init__(self, input_size: int, hidden_size: int = 256, 
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
        
        # Output decoder with skip connections
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout)
            ),
            nn.Sequential(
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.GELU(),
                nn.Dropout(dropout)
            ),
            nn.Linear(hidden_size // 4, 1)
        ])
        
    def _create_positional_encoding(self, max_len: int, d_model: int):
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def forward(self, x):
        # x shape: (batch_size, 1, sequence_length)
        B, _, T = x.shape
        
        # Multi-scale feature extraction
        conv_features = []
        for conv in self.input_convs:
            conv_features.append(conv(x))
        
        # Combine features
        x = torch.cat(conv_features, dim=1)  # (B, hidden_size, T)
        x = self.input_norm(x)
        x = self.input_dropout(x)
        
        # Transpose for transformer: (B, T, hidden_size)
        x = x.transpose(1, 2)
        
        # Add positional encoding
        if T <= self.pos_encoding.size(1):
            x = x + self.pos_encoding[:, :T, :]
        else:
            # Handle sequences longer than max_len
            pe_extended = self.pos_encoding.repeat(1, (T // self.pos_encoding.size(1)) + 1, 1)
            x = x + pe_extended[:, :T, :]
        
        # Transformer processing
        x = self.transformer(x)
        
        # Decoder with skip connections
        skip = x
        for i, layer in enumerate(self.decoder):
            x = layer(x)
            if i == 0:  # Add skip connection after first layer
                x = x + skip[:, :, :x.size(-1)]
        
        # Transpose back: (B, 1, T)
        output = x.transpose(1, 2)
        
        return output


class WaveNet(nn.Module):
    """WaveNet-inspired model for signal-to-signal prediction."""
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 10, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input projection
        self.input_conv = nn.Conv1d(1, hidden_size, kernel_size=1)
        
        # Dilated convolution layers
        self.dilated_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        
        for i in range(num_layers):
            dilation = 2 ** (i % 8)  # Cycle through dilations
            
            # Dilated convolution
            self.dilated_convs.append(
                nn.Conv1d(hidden_size, hidden_size * 2, kernel_size, 
                         dilation=dilation, padding=dilation * (kernel_size - 1) // 2)
            )
            
            # Residual connection
            self.residual_convs.append(
                nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
            )
            
            # Skip connection
            self.skip_convs.append(
                nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
            )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size // 2, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size // 2, 1, kernel_size=1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, 1, sequence_length)
        
        # Input projection
        x = self.input_conv(x)
        
        # Skip connections accumulator
        skip_connections = 0
        
        # Process through dilated convolution layers
        for i in range(self.num_layers):
            # Store input for residual connection
            residual = x
            
            # Dilated convolution with gated activation
            conv_out = self.dilated_convs[i](x)
            tanh_out, sigmoid_out = conv_out.chunk(2, dim=1)
            gated = torch.tanh(tanh_out) * torch.sigmoid(sigmoid_out)
            
            # Residual connection
            x = self.residual_convs[i](gated) + residual
            
            # Skip connection
            skip_connections = skip_connections + self.skip_convs[i](gated)
        
        # Output processing
        output = self.output_layers(skip_connections)
        
        return output
