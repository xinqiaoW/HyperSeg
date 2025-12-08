import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (D,) - wavelengths
        Returns:
            Tensor of shape (D, E) - positional encoded wavelengths
        """
        # Normalize wavelengths to indices [0, max_len)
        indices = (x - x.min()) / (x.max() - x.min() + 1e-8) * (self.pe.size(0) - 1)
        indices = indices.long().clamp(0, self.pe.size(0) - 1)
        return self.pe[indices]


class ChannelProj(nn.Module):
    def __init__(self, embed_dim, hidden_dim=256, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim

        # Positional encoding for wavelengths
        self.pe = PositionalEncoding(d_model=embed_dim)

        # Two-layer MLP for wavelength processing
        self.wv_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

        # Transformer layer
        self.transformer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )

        # Final MLP for projection
        self.final_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor, input_wavelengths: torch.Tensor) -> torch.Tensor:
        """Forward pass through the channel projection layer.

        Args:
            x: Input tensor of shape (B * N, D) where D is number of spectral channels.
            input_wavelengths: Tensor of shape (D,) of wavelengths corresponding to input channels.

        Returns:
            Projected features of shape (B * N, E).
        """
        # Step 1: Apply positional encoding to wavelengths (D,) -> (D, E)
        wvs = self.pe(input_wavelengths)  # (D, E)

        # Step 2: Apply two-layer MLP to wvs (D, E) -> (D, E)
        wvs = self.wv_mlp(wvs)  # (D, E)

        # Step 3: Apply transformer (add batch dimension)
        wvs_batched = wvs.unsqueeze(0)  # (1, D, E)
        transformer_output = self.transformer(wvs_batched)  # (1, D, E)
        transformer_output = transformer_output.squeeze(0)  # (D, E)

        # Step 4: Final projection - x @ mlp(transformer_output)
        projection_weights = self.final_mlp(transformer_output)  # (D, E)

        # Step 5: Apply projection
        output = x @ projection_weights  # (B*N, D) @ (D, E) -> (B*N, E)

        return output