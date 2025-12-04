"""
Query Processor Module
Handles spectral query extraction and projection for hyperspectral segmentation.
"""
import torch
import torch.nn as nn
from typing import List, Dict, Any
from .channel_proj import ChannelProj


class QueryProcessor(nn.Module):
    """
    Processes spectral queries from hyperspectral images.
    Extracts point-based queries and projects them to appropriate dimensions.
    
    Args:
        input_channels: Number of input spectral channels
        embed_dim: Embedding dimension for queries
        channel_proj: Optional custom channel projection module
    """
    def __init__(
        self, 
        input_channels: int,
        embed_dim: int,
        channel_proj: ChannelProj = None
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        if channel_proj is not None:
            self.channel_proj = channel_proj
        else:
            self.channel_proj = nn.Sequential(
                nn.Linear(input_channels, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
            )
    
    def extract_point_queries(
        self,
        images: List[torch.Tensor],
        batched_input: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """
        Extract spectral queries from point coordinates.

        Args:
            images: List of input images
            batched_input: List of input dictionaries containing point coordinates

        Returns:
            Queries tensor of shape (B, query_len, channels)
        """
        # Extract queries from different images
        queries = torch.stack([
            img[:, x["point_coords"][0, :, 1], x["point_coords"][0, :, 0]]
            for img, x in zip(images, batched_input)
        ], dim=0).permute(0, 2, 1)

        return queries
    
    def extract_feature_queries(
        self,
        image_embeddings: torch.Tensor,
        batched_input: List[Dict[str, Any]],
        img_size: int
    ) -> torch.Tensor:
        """
        Extract queries from feature embeddings at point coordinates.

        Args:
            image_embeddings: Feature embeddings from encoder
            batched_input: List of input dictionaries containing point coordinates
            img_size: Target image size for interpolation

        Returns:
            Queries tensor of shape (B, query_len, embed_dim)
        """
        # Interpolate embeddings to target size
        image_embeddings_scaled = torch.nn.functional.interpolate(
            image_embeddings,
            size=(img_size, img_size),
            mode='bilinear',
            align_corners=False
        )

        queries = torch.stack([
            image_embeddings_scaled[i][:, x["point_coords"][0, :, 1], x["point_coords"][0, :, 0]]
            for i, x in enumerate(batched_input)
        ], dim=0).permute(0, 2, 1)

        return queries
    
    def project_queries(
        self,
        queries: torch.Tensor,
        wavelengths: List[float],
        test_mode: bool = False
    ) -> torch.Tensor:
        """
        Project queries to embedding dimension.
        
        Args:
            queries: Input queries of shape (B, query_len, channels)
            wavelengths: List of wavelengths for spectral channels
            test_mode: Whether in test mode
            
        Returns:
            Projected queries of shape (B, query_len, embed_dim)
        """
        B, Q_L, Embed_dim = queries.shape
        
        if not isinstance(self.channel_proj, ChannelProj):
            # Simple MLP projection
            queries = self.channel_proj(queries.reshape(-1, Embed_dim)).reshape(B, Q_L, -1)
        else:
            # Wavelength-aware projection
            queries = self.channel_proj(
                queries.reshape(-1, Embed_dim), 
                wavelengths, 
                test_mode=test_mode
            ).view(B, Q_L, -1)
        
        return queries
