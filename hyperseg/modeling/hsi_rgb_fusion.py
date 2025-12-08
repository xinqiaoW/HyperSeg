import torch
import torch.nn as nn
import torch.nn.functional as F


class ResConv(nn.Module):
    """Residual convolution block with optional channel reduction"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(1, out_channels)  # GroupNorm with 1 group = LayerNorm for conv
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(1, out_channels)

        # Shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.GroupNorm(1, out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out += identity
        out = self.relu(out)

        return out


class HSIRGBFusion(nn.Module):
    """
    Fuses RGB embeddings with spatial (HSI) embeddings using residual convolutions.

    Architecture:
        RGB embeddings (B, C, h, w) → ResConv → (B, C/2, h, w) ─┐
                                                                 ├─ Concat → (B, C, h, w) → ResConv → (B, C, h, w)
        Spatial embeddings (B, C, h, w) → ResConv → (B, C/2, h, w) ─┘
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        half_channels = channels // 2

        # ResConv for RGB embeddings: C → C/2
        self.rgb_resconv = ResConv(channels, half_channels)

        # ResConv for spatial embeddings: C → C/2
        self.spatial_resconv = ResConv(channels, half_channels)

        # ResConv for fused features: C → C
        self.fusion_resconv = ResConv(channels, channels)

    def forward(self, rgb_embeddings, spatial_embeddings):
        """
        Args:
            rgb_embeddings: Tensor of shape (B, C, h, w)
            spatial_embeddings: Tensor of shape (B, C, h, w)

        Returns:
            Fused embeddings of shape (B, C, h, w)
        """
        B, C, h, w = rgb_embeddings.shape

        # Ensure spatial embeddings match RGB embeddings size
        if spatial_embeddings.shape[-2:] != rgb_embeddings.shape[-2:]:
            spatial_embeddings = F.interpolate(
                spatial_embeddings,
                size=(h, w),
                mode='bilinear',
                align_corners=False
            )

        # Process RGB embeddings: (B, C, h, w) → (B, C/2, h, w)
        rgb_features = self.rgb_resconv(rgb_embeddings)

        # Process spatial embeddings: (B, C, h, w) → (B, C/2, h, w)
        spatial_features = self.spatial_resconv(spatial_embeddings)

        # Concatenate along channel dimension: (B, C, h, w)
        fused = torch.cat([rgb_features, spatial_features], dim=1)

        # Final residual convolution: (B, C, h, w) → (B, C, h, w)
        output = self.fusion_resconv(fused)

        return output
