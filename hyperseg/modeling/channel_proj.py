import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelProj(nn.Module):
    def __init__(self, embed_dim, hidden_dim=256, num_layers=4):
        super().__init__()
        # 使用LayerNorm或BatchNorm提高稳定性
        self.mlp_layers = nn.ModuleList()
        in_dim = 1
        
        for i in range(num_layers):
            out_dim = embed_dim if i == num_layers-1 else hidden_dim
            layer = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim) if i < num_layers-1 else nn.Identity(),
                nn.ReLU() if i < num_layers-1 else nn.Identity()
            )
            self.mlp_layers.append(layer)
            in_dim = out_dim
            
        # 可学习的归一化参数
        self.register_buffer('wavelength_min', torch.tensor(400.0))
        self.register_buffer('wavelength_max', torch.tensor(2500.0))
    
    def normalize_wavelengths(self, wavelengths):
        # 更灵活的归一化
        return (wavelengths - self.wavelength_min) / (self.wavelength_max - self.wavelength_min + 1e-8)
    
    def forward(self, x: torch.Tensor, input_wavelengths: torch.Tensor) -> torch.Tensor:
        """Forward pass through the channel projection layer.

        Args:
            x: Input tensor of shape (B * N, D) where D is number of spectral channels.
            input_wavelengths: Tensor of shape (D,) of wavelengths corresponding to input channels.

        Returns:
            Projected features of shape (B * N, E).
        """
        _, D = x.shape

        # Normalize input wavelengths
        norm_wavelengths = self.normalize_wavelengths(input_wavelengths)  # (D,)

        mlp_input = norm_wavelengths.unsqueeze(-1)

        # Generate projection weights using MLP: (D,) -> (D, embed_dim)
        for layer in self.mlp_layers:
            mlp_input = layer(mlp_input)

        weights = mlp_input
        output = x @ weights

        return output