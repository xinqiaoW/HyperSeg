import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Type
from .common import LayerNorm2d, MLPBlock
import math
import warnings
import numpy as np
import random
from itertools import repeat
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs
else:
    import collections.abc as container_abcs
from ..utils.spectral_process_utils import *
from .scale_aware_PE import get_2d_sincos_pos_embed_with_resolution


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
    

class ImageEncoderViT(nn.Module):
    """Vision Transformer encoder for hyperspectral image segmentation.
    
    This encoder supports multi-scale feature extraction, spectral wavelength matching,
    and adaptive positional embeddings for various spatial resolutions.
    """
    
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = False,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
        in_chans_spectral = 85,
        merge_indexs = [3, 6, 8, 11], # for Vit-b version
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
            in_chans_spectral (int): Number of spectral channels.
            merge_indexs (list): Layer indices where feature merging occurs.
        """
        super().__init__()
        
        # ============ Basic Configuration ============
        self.in_chans = in_chans
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.patch_size = patch_size
        self.out_chans = out_chans
        self.merge_indexs = merge_indexs
        self.global_attn_indexes = global_attn_indexes
        
        # ============ Positional Embedding Module ============
        # MLP for processing scale-aware positional embeddings
        self.pos_embed_mlp = MLP(self.embed_dim, self.embed_dim//2, self.embed_dim, 3, sigmoid_output=False)

        # ============ Transformer Blocks ============
        # Main transformer blocks for feature extraction
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        # ============ Contrastive Learning Modules ============
        # Additional blocks for semantic feature conversion
        self.contras_modules = nn.ModuleList()
        for i in range(2):
            block = Block(
                dim=256,
                num_heads=8,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=16,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.contras_modules.append(block)

        # ============ Output Neck ============
        # Feature dimension reduction and refinement
        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )
    
        # ============ Spectral Wavelength Matching ============
        # Distance threshold for wavelength matching (nm)
        self.nm_dis = 10
        # Pre-compute wavelength matching indices for different data sources
        self.Band_feature_indices_hy, self.unmatch_indices_hy, self.point_bank_indices_hy = find_corresponding_indices(input_wavelengths_hy, spectral_wavelength,self.nm_dis)
        self.Band_feature_indices_mu, self.unmatch_indices_mu, self.point_bank_indices_mu = find_corresponding_indices(input_wavelengths_mu, spectral_wavelength,self.nm_dis)
        self.weight_bank_data_indices_hy, _, self.weight_bank_indices_hy = find_corresponding_indices(input_wavelengths_hy, weight_bank_wavelength,self.nm_dis)
        
        # ============ Spectral Weight Banks ============
        # Learnable weight banks for spectral channel processing
        # Point-wise spectral weights
        self.point_spectral_weight_bank_w = nn.Parameter(torch.randn((self.embed_dim, len(spectral_wavelength), patch_size, patch_size)))
        self.point_spectral_weight_bank_b = nn.Parameter(torch.randn(self.embed_dim))
        # Block-wise spectral weights
        self.block_spectral_weight_bank_w = nn.Parameter(torch.randn((self.embed_dim, len(weight_bank_wavelength), patch_size, patch_size)))
        self.block_spectral_weight_bank_b = nn.Parameter(torch.randn(self.embed_dim))

        # ============ Multi-Scale Feature Merging ============
        # Patch merging layers for hierarchical feature extraction
        self.multi_scale_convs = nn.ModuleList([
            PatchMerging(dim=embed_dim)
            for i in range(len(self.merge_indexs))]) if self.merge_indexs != None else None

    # ============ Utility Methods ============
    
    def find_indices_not_in_A(self, A, B):
        """Find indices of elements in B that are not present in A.
        
        Args:
            A (list): Reference list.
            B (list): Target list to check.
            
        Returns:
            list: Indices of elements in B not found in A.
        """
        set_A = set(A)
        result_indices = []
        for index, element in enumerate(B):
            if element not in set_A:
                result_indices.append(index)
        return result_indices
    
    # ============ Feature Processing Methods ============
    
    def convert_semantic_feature(self, backbone_features):
        """Convert backbone features to semantic features using contrastive modules.
        
        Args:
            backbone_features (torch.Tensor): Features from backbone [B, C, H, W].
            
        Returns:
            torch.Tensor: Converted semantic features [B, C, H, W].
        """
        # Transform to [B, H, W, C] for transformer processing
        backbone_features = backbone_features.permute((0,2,3,1))

        # Apply contrastive transformer blocks
        for i, blk in enumerate(self.contras_modules):
            backbone_features = blk(backbone_features)
        
        # Transform back to [B, C, H, W]
        contras_features = (backbone_features.permute(0, 3, 1, 2))
        return contras_features


    # ============ Forward Pass ============
    
    def forward(self, x: torch.Tensor, test_mode=False, input_wavelength=None, GSD=None) -> torch.Tensor:
        """Forward pass through the encoder.
        
        Args:
            x (torch.Tensor): Input image tensor [B, C, H, W].
            test_mode (bool): If True, use all input channels. If False, randomly select 40 channels.
            input_wavelength (list): Wavelengths for each hyperspectral channel.
            GSD (list or torch.Tensor): Ground sampling distance (m/pixel), e.g., [1.0].

        Returns:
            list: Multi-stage backbone features at different resolutions.
        """

        # -------- 1. Determine Input Data Type --------
        # Distinguish between hyperspectral (hy) and multispectral (mu) data
        is_hyperspectral = x.shape[1] >= 20  # Hyperspectral: many channels (>=20)
        is_multispectral = x.shape[1] < 20  # Multispectral: fewer channels (<20)
        
        # Handle specific case: 224 channels in training mode
        if x.shape[1] == 224 and not test_mode:
            x = x[:,4:224,:,:]  # Skip first 4 channels

        # -------- 2. Update Wavelength Matching Indices --------
        # Dynamically compute indices if input wavelengths are provided
        if input_wavelength != None and is_hyperspectral:
            input_wavelengths = input_wavelength
            # Match input wavelengths to spectral weight bank wavelengths
            self.key_feature_data_indices, self.unmatched_indices, self.key_feature_bank_indices = find_corresponding_indices(
                input_wavelengths, spectral_wavelength, self.nm_dis)
            # Match input wavelengths to auxiliary weight bank wavelengths
            self.aux_feature_data_indices, _, self.aux_feature_bank_indices = find_corresponding_indices(
                input_wavelengths, weight_bank_wavelength, self.nm_dis)
        elif input_wavelength != None and is_multispectral:
            input_wavelengths = input_wavelength
            # For multispectral, only use key features
            self.key_feature_data_indices_mu, self.unmatched_indices_mu, self.key_feature_bank_indices_mu = find_corresponding_indices(
                input_wavelengths, spectral_wavelength, self.nm_dis)

        # -------- 3. Prepare Channel Indices --------
        # Select channels based on data type and mode
        # indices format: [data_key_indices, bank_key_indices, data_aux_indices, bank_aux_indices]
        if is_hyperspectral:
            if not test_mode:
                # Training mode: randomly select 40 auxiliary channels for data augmentation
                num_aux_channels = 40
                random_aux_indices = generate_random_indices(len(self.aux_feature_data_indices)-1, num_aux_channels)
                random_aux_indices.sort()
                channel_indices = [
                    self.key_feature_data_indices,  # Key wavelength indices in input data
                    self.key_feature_bank_indices,  # Key wavelength indices in weight bank
                    np.array(self.aux_feature_data_indices)[random_aux_indices].tolist(),  # Random aux indices in data
                    np.array(self.aux_feature_bank_indices)[random_aux_indices].tolist()  # Random aux indices in bank
                ]
            else:
                # Test mode: use all available channels
                channel_indices = [
                    self.key_feature_data_indices,
                    self.key_feature_bank_indices,
                    self.aux_feature_data_indices,
                    self.aux_feature_bank_indices
                ]
            # Remove overlapping indices between key and auxiliary features to avoid duplication
            non_overlapping_indices = self.find_indices_not_in_A(channel_indices[0], channel_indices[2])
            channel_indices[2] = np.array(channel_indices[2])[non_overlapping_indices].tolist()
            channel_indices[3] = np.array(channel_indices[3])[non_overlapping_indices].tolist()
            self.last_channel_indices = channel_indices
        elif is_multispectral:
            # Multispectral data: only use key features (no auxiliary features)
            channel_indices = [self.key_feature_data_indices_mu, self.key_feature_bank_indices_mu, [], []]
            self.last_channel_indices = channel_indices

        # -------- 4. Process Ground Sampling Distance (GSD) --------
        if GSD == None:
            GSD = [1.0]  # Default GSD value (meters per pixel)
        if not torch.is_tensor(GSD):
            GSD = torch.tensor(GSD)
        
        # -------- 5. Spectral Feature Extraction --------
        # Extract key spectral features using point-wise convolution
        key_spectral_features = F.conv2d(
            x[:, channel_indices[0], :, :],  # Select key wavelength channels
            weight=self.point_spectral_weight_bank_w[:, channel_indices[1], :, :],
            bias=self.point_spectral_weight_bank_b,
            stride=(self.patch_size, self.patch_size),
            padding=(0, 0)
        )
        
        # Extract auxiliary spectral features (if available)
        if len(channel_indices[2]) > 0:
            aux_spectral_features = F.conv2d(
                x[:, channel_indices[2], :, :],  # Select auxiliary wavelength channels
                weight=self.block_spectral_weight_bank_w[:, channel_indices[3], :, :],
                bias=self.block_spectral_weight_bank_b,
                stride=(self.patch_size, self.patch_size),
            )

        # Combine key and auxiliary features
        if len(channel_indices[2]) > 0:
            combined_features = key_spectral_features + aux_spectral_features
        else:
            combined_features = key_spectral_features

        # -------- 6. Add Scale-Aware Positional Embeddings --------
        num_patches_per_side = int(self.img_size / self.patch_size)
        # Generate scale-aware sinusoidal positional embeddings
        scale_aware_pos_embed = get_2d_sincos_pos_embed_with_resolution(
            self.embed_dim, num_patches_per_side, GSD, device=x.device)
        # Process through MLP for better positional encoding
        scale_aware_pos_embed = self.pos_embed_mlp(scale_aware_pos_embed)
        scale_aware_pos_embed = scale_aware_pos_embed.reshape(
            (1, num_patches_per_side, num_patches_per_side, self.embed_dim))
        
        # Transform features to [B, H, W, C] format for transformer processing
        patch_tokens = combined_features.permute((0, 2, 3, 1))  # [B, C, H, W] -> [B, H, W, C]
        patch_tokens = patch_tokens + scale_aware_pos_embed  # Add positional information
        
        # -------- 7. Multi-Stage Feature Extraction --------
        hierarchical_features = []  # Store features at different resolutions
        merge_layer_index = 0
        
        for layer_idx, transformer_block in enumerate(self.blocks):
            # Apply transformer block with gradient checkpointing for memory efficiency
            patch_tokens = torch.utils.checkpoint.checkpoint(
                transformer_block, patch_tokens, use_reentrant=True)
            
            # Collect multi-stage features and apply patch merging
            if self.merge_indexs != None:
                # Save features at specific milestone layers
                if layer_idx in [self.merge_indexs[0], self.global_attn_indexes[0], self.global_attn_indexes[2]]:
                    hierarchical_features.append(patch_tokens.permute(0, 3, 1, 2))  # [B, H, W, C] -> [B, C, H, W]

                # Apply patch merging at specified layers for hierarchical processing
                if layer_idx in self.merge_indexs:
                    patch_tokens = self.multi_scale_convs[merge_layer_index](patch_tokens)
                    merge_layer_index += 1
            elif layer_idx in self.global_attn_indexes:
                # Save features at global attention layers
                hierarchical_features.append(patch_tokens.permute(0, 3, 1, 2))

        # -------- 8. Apply Output Neck --------
        # Final feature refinement and dimension adjustment
        final_features = self.neck(patch_tokens.permute(0, 3, 1, 2))
        hierarchical_features.append(final_features)

        return hierarchical_features


def to_2tuple(x):
    if isinstance(x, container_abcs.Iterable):
        return x
    return tuple(repeat(x, 2))


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    
    """Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        B, H, W, C = x.shape
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)
        x = x.view(B, H // 2, W // 2, C)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x