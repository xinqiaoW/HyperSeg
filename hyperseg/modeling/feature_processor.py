"""
Feature Processor Module
Handles feature extraction, fusion, and upsampling operations.
"""
import math
import torch
import torch.nn as nn
import numpy as np
from typing import List
from .common import LayerNorm2d
from .mask_decoder_hq import MLP
from .transformer import TwoWayTransformer
from .scale_aware_PE import get_2d_sincos_pos_embed_with_resolution


class FeatureUpsampler(nn.Module):
    """
    Upsamples feature maps using transposed convolutions.
    
    Args:
        embed_dim: Input embedding dimension
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(embed_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=2, stride=2),
            nn.GELU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Upsampled tensor
        """
        return self.upsample(x)


class SpectralFeatureFusion(nn.Module):
    """
    Fuses spectral features with image embeddings using cross-attention.
    
    Args:
        embed_dim: Embedding dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        mlp_dim: MLP hidden dimension
    """
    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 3,
        num_heads: int = 8,
        mlp_dim: int = 2048
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # Cross-attention fusion
        self.cross_attention_fusion = TwoWayTransformer(
            depth=num_layers,
            embedding_dim=embed_dim,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
        )
        
        # High-frequency token for refinement
        self.hf_token = nn.Embedding(1, embed_dim)
        self.hf_mlp = MLP(
            embed_dim, 
            embed_dim, 
            embed_dim // 8, 
            3, 
            zeros_init=True
        )
        
        # Query encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=0.1,
            batch_first=True
        )
        self.query_encoder = nn.TransformerEncoder(encoder_layer, num_layers=8)
        
        # Position embedding MLP
        self.pos_embed_mlp = MLP(
            embed_dim, 
            embed_dim // 2, 
            embed_dim, 
            3, 
            sigmoid_output=False
        )
        
        # Feature upsampler
        self.upsampler = FeatureUpsampler(embed_dim)
    
    @property
    def device(self) -> torch.device:
        """Get device of the module."""
        return self.upsampler.upsample[0].weight.device
    
    def generate_scale_aware_pos_embed(
        self,
        GSD: torch.Tensor,
        img_size: int,
        patch_size: int
    ) -> torch.Tensor:
        """
        Generate scale-aware positional embeddings.
        
        Args:
            GSD: Ground sampling distance tensor
            img_size: Image size
            patch_size: Patch size
            
        Returns:
            Positional embeddings tensor
        """
        scale_aware_pos_embed_list = [
            get_2d_sincos_pos_embed_with_resolution(
                self.embed_dim, 
                img_size // patch_size, 
                GSD[gsd], 
                device=self.device
            ) 
            for gsd in range(GSD.shape[0])
        ]
        
        scale_aware_pos_embed_list = [
            self.pos_embed_mlp(scale_aware_pos_embed) 
            for scale_aware_pos_embed in scale_aware_pos_embed_list
        ]
        
        scale_aware_pos_embed_list = [
            scale_aware_pos_embed.permute(0, 2, 1).reshape(
                (1, self.embed_dim, img_size // patch_size, img_size // patch_size)
            ) 
            for scale_aware_pos_embed in scale_aware_pos_embed_list
        ]
        
        return torch.cat(scale_aware_pos_embed_list, dim=0)
    
    def fuse_features(
        self,
        queries: torch.Tensor,
        image_embeddings: torch.Tensor,
        keys_pe: torch.Tensor
    ) -> tuple:
        """
        Fuse queries with image embeddings using cross-attention.

        Args:
            queries: Query tensor of shape (B, Q_L, embed_dim)
            image_embeddings: Image embeddings of shape (B, embed_dim, H, W)
            keys_pe: Positional embeddings for keys

        Returns:
            Tuple of (fused_queries, fused_embeddings, hf_token)
        """
        B = queries.shape[0]
        
        # Encode queries
        queries = self.query_encoder(queries)
        
        # Add high-frequency token
        queries = torch.cat([
            queries, 
            self.hf_token.weight[:, None, :].repeat((B, 1, 1))
        ], dim=1)
        
        # Cross-attention fusion
        queries, image_embeddings = self.cross_attention_fusion(
            image_embedding=image_embeddings,
            image_pe=keys_pe,
            point_embedding=queries
        )
        
        # Extract HF token
        out_hf_token = queries[:, -1:, :]
        
        # Upsample image embeddings
        B, HW, C = image_embeddings.shape
        H, W = int(math.sqrt(HW)), int(math.sqrt(HW))
        image_embeddings = self.upsampler(
            image_embeddings.permute(0, 2, 1).view(-1, C, H, W)
        )
        
        return queries, image_embeddings, out_hf_token
    
    def compute_error_correction(
        self,
        hf_token: torch.Tensor,
        image_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute error correction using high-frequency token.
        
        Args:
            hf_token: High-frequency token of shape (B, 1, embed_dim)
            image_embedding: Image embedding of shape (B, C, H, W)
            
        Returns:
            Error correction tensor of shape (B, -, H, W)
        """
        b, c, h, w = image_embedding.shape
        error_correction = (
            self.hf_mlp(hf_token) @ image_embedding.view(b, c, h * w)
        ).view(b, -1, h, w)
        return error_correction
