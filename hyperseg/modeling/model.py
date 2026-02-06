import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List, Tuple, Optional

from .image_encoder_rgb import ImageEncoderRGB, PatchEmbed, Block
from .mask_decoder_hq import MaskDecoderHQ, MLP
from .prompt_encoder import PromptEncoder
from ..utils.spectral_process_utils import interpolate_hyperspectral_image_transform_matrix
from ..utils.transforms import ResizeLongestSide
import copy


class ZeroConv2d(nn.Module):
    """Zero-initialized 2D convolution for ControlNet-style conditioning."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        # Initialize weights and bias to zero
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ZeroMLP(nn.Module):
    """Zero-initialized MLP for ControlNet-style conditioning."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        # Initialize the last layer to zero
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SpectralQueryFusion(nn.Module):
    """
    Fuses spectral queries into a single token for mask decoder.

    Process:
    1. MLP to project num_bands -> embed_dim
    2. Transformer layer for query interaction
    3. Fusion to single token
    4. Zero-initialized MLP before adding to prompt tokens
    """

    def __init__(
        self,
        num_bands: int = 224,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 2048,
    ):
        super().__init__()
        self.num_bands = num_bands
        self.embed_dim = embed_dim

        # MLP to project spectral bands to embed_dim
        self.band_projection = nn.Sequential(
            nn.Linear(num_bands, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
        )

        # Transformer layer for query interaction
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            activation='gelu',
            batch_first=True,
        )

        # Learnable fusion token
        self.fusion_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.fusion_token, std=0.02)

        # Zero-initialized MLP for final output
        self.zero_mlp = ZeroMLP(embed_dim, mlp_dim, embed_dim, num_layers=2)

    def forward(self, spectral_queries: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spectral_queries: (B, num_queries, num_bands)

        Returns:
            fused_token: (B, 1, embed_dim) - ready to add to prompt tokens
        """
        B, num_queries, _ = spectral_queries.shape

        # Project bands to embed_dim: (B, num_queries, embed_dim)
        queries = self.band_projection(spectral_queries)

        # Add fusion token: (B, num_queries + 1, embed_dim)
        fusion_token = self.fusion_token.expand(B, -1, -1)
        queries_with_fusion = torch.cat([fusion_token, queries], dim=1)

        # Transformer layer for interaction
        queries_out = self.transformer_layer(queries_with_fusion)

        # Extract fused token: (B, 1, embed_dim)
        fused_token = queries_out[:, :1, :]

        # Apply zero-initialized MLP
        fused_token = self.zero_mlp(fused_token)

        return fused_token


class HSIPatchEmbed(nn.Module):
    """HSI Patch Embedding for fixed input channels."""

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        fixed_channels: int = 224,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.fixed_channels = fixed_channels
        self.embed_dim = embed_dim

        # Patch embedding for HSI
        self.proj = nn.Conv2d(
            fixed_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: HSI image (B, C, H, W) with fixed_channels
        Returns:
            HSI tokens (B, H/patch_size, W/patch_size, embed_dim)
        """
        x = self.proj(x)
        # B C H W -> B H W C (same as RGB PatchEmbed)
        x = x.permute(0, 2, 3, 1)
        return x


class ImageEncoderHSI(nn.Module):
    """HSI Encoder that mirrors RGB encoder structure for ControlNet-style fusion.

    This encoder processes HSI images and produces features at each layer
    that can be fused with the RGB encoder via ZeroMLPs.
    """

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        fixed_channels: int = 224,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: type = nn.LayerNorm,
        act_layer: type = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ):
        super().__init__()
        self.img_size = img_size
        self.fixed_channels = fixed_channels
        self.depth = depth

        # HSI-specific patch embedding
        self.patch_embed = HSIPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            fixed_channels=fixed_channels,
            embed_dim=embed_dim,
        )

        # Positional embedding (will be copied from RGB encoder)
        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        # Transformer blocks (will be copied from RGB encoder)
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

        # Zero MLPs for each layer (initialized to output zeros)
        self.zero_mlps = nn.ModuleList([
            ZeroMLP(embed_dim, embed_dim * 2, embed_dim, num_layers=2)
            for _ in range(depth)
        ])

    def _select_and_interpolate_bands(
        self,
        x: torch.Tensor,
        start_band: Optional[int],
        num_bands: Optional[int],
    ) -> torch.Tensor:
        """
        Select continuous bands and interpolate to fixed_channels.

        Args:
            x: HSI image (B, C, H, W) with variable channels
            start_band: Starting band index (None means use all bands)
            num_bands: Number of continuous bands to select (None means use all bands)

        Returns:
            Interpolated HSI (B, fixed_channels, H, W)
        """
        B, C, H, W = x.shape

        # If no band selection specified, use all bands
        if start_band is None or num_bands is None:
            selected = x
        else:
            # Clamp to valid range
            start_band = max(0, min(start_band, C - 1))
            num_bands = min(num_bands, C - start_band)
            selected = x[:, start_band:start_band + num_bands, :, :]

        # Interpolate to fixed_channels
        _, C_sel, _, _ = selected.shape
        if C_sel == self.fixed_channels:
            return selected

        # Reshape for interpolation: (B, 1, C_sel, H*W)
        x_flat = selected.view(B, C_sel, H * W).unsqueeze(1)
        x_interp = F.interpolate(
            x_flat,
            size=(self.fixed_channels, H * W),
            mode='bilinear',
            align_corners=False
        )
        # Reshape back: (B, fixed_channels, H, W)
        x_interp = x_interp.squeeze(1).view(B, self.fixed_channels, H, W)

        return x_interp

    def forward(
        self,
        x: torch.Tensor,
        start_band: Optional[int] = None,
        num_bands: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """
        Process HSI and return features at each layer for fusion.

        Args:
            x: HSI image (B, C, H, W) with variable channels
            start_band: Starting band index for continuous band selection (None = use all)
            num_bands: Number of continuous bands to select (None = use all)

        Returns:
            List of tensors, one per layer: (B, H/patch, W/patch, embed_dim)
            Each tensor has been passed through ZeroMLP (initially outputs zeros)
        """
        # Select and interpolate bands
        x = self._select_and_interpolate_bands(x, start_band, num_bands)

        # Patch embedding
        x = self.patch_embed(x)

        # Add positional embedding
        if self.pos_embed is not None:
            x = x + self.pos_embed

        # Process through blocks and collect features
        layer_features = []
        for i, blk in enumerate(self.blocks):
            x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            # Apply ZeroMLP to get fusion features (initially zeros)
            fusion_feat = self.zero_mlps[i](x)
            layer_features.append(fusion_feat)

        return layer_features


class HyperSeg(nn.Module):
    """HyperSeg: Hyperspectral Image Segmentation Model with ControlNet-like Structure.

    This model uses a ControlNet-inspired architecture where:
    1. RGB encoder (frozen) processes RGB images
    2. HSI encoder (trainable, cloned from RGB encoder) processes HSI images
    3. At each layer, HSI features go through ZeroMLP and are added to RGB features

    Args:
        image_encoder_rgb (ImageEncoderRGB): RGB image encoder backbone (SAM ViT, frozen).
        image_encoder_hsi (ImageEncoderHSI): HSI image encoder (cloned from RGB, trainable).
        prompt_encoder (PromptEncoder): SAM-style prompt encoder.
        mask_decoder_hq (MaskDecoderHQ): High-quality mask decoder.
        pixel_mean (List[float]): Mean values for RGB normalization.
        pixel_std (List[float]): Standard deviation for RGB normalization.
        fixed_hsi_channels (int): Fixed number of channels for HSI interpolation.
        image_format (str): Image format (default: "RGB").
    """

    mask_threshold: float = 0.0

    def __init__(
            self,
            image_encoder_rgb: ImageEncoderRGB,
            image_encoder_hsi: ImageEncoderHSI,
            prompt_encoder: PromptEncoder,
            mask_decoder_hq: MaskDecoderHQ,
            pixel_mean: List[float] = [123.675, 116.28, 103.53],
            pixel_std: List[float] = [58.395, 57.12, 57.375],
            fixed_hsi_channels: int = 224,
            image_format: str = "RGB",
    ) -> None:
        super(HyperSeg, self).__init__()

        # ============ RGB Normalization Parameters ============
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # ============ Dual Encoder Architecture ============
        self.rgb_encoder = image_encoder_rgb  # Frozen
        self.hsi_encoder = image_encoder_hsi  # Trainable

        # ============ Decoder Modules ============
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder_hq

        # ============ Model Architecture Parameters ============
        self.img_size = image_encoder_rgb.img_size
        self.patch_size = image_encoder_rgb.patch_embed.proj.kernel_size[0]
        self.image_format = image_format
        self.embed_dim = self.prompt_encoder.embed_dim
        self.fixed_hsi_channels = fixed_hsi_channels
        self.vit_embed_dim = image_encoder_rgb.patch_embed.proj.out_channels

        # ============ Spectral Query Fusion ============
        self.spectral_query_fusion = SpectralQueryFusion(
            num_bands=fixed_hsi_channels,
            embed_dim=self.embed_dim,
            num_heads=8,
            mlp_dim=2048,
        )

        # ============ Image Preprocessing Utilities ============
        self.transform = ResizeLongestSide(self.img_size)

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return self.pixel_mean.device

    def forward(
            self,
            batched_input: List[Dict[str, Any]],
            wavelengths: List[float],
            multimask_output: bool,
            start_band: Optional[int] = None,
            num_bands: Optional[int] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks using ControlNet-like dual encoder architecture.

        The forward pass:
        1. Convert HSI to RGB for the frozen RGB encoder
        2. Process HSI through HSI encoder, collecting features at each layer
        3. Process RGB through RGB encoder, adding HSI features (via ZeroMLP) at each layer
        4. Use fused features for mask prediction

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys:
              'image': The HSI image as a torch tensor in CxHxW format.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          wavelengths (list): List of wavelengths corresponding to the input image.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.
          start_band (int, optional): Starting band index for HSI encoder (None = use all).
          num_bands (int, optional): Number of continuous bands to select (None = use all).

        Returns:
          (list(dict)): A list over input images, where each element is
            a dictionary with the following keys:
              'masks': (torch.Tensor) Batched binary mask predictions.
              'iou_predictions': (torch.Tensor) The model's predictions of mask quality.
              'low_res_logits': (torch.Tensor) Low resolution logits.
        """
        # Transform the input images to the correct size
        for image_record in batched_input:
            image_record["image"] = self.transform.apply_image_torch(
                image_record["image"].unsqueeze(0)
            ).squeeze(0)
            input_size = image_record["image"].shape[-2:]
            image_record["point_coords"] = self.transform.apply_coords_torch(
                image_record["point_coords"], image_record["original_size"]
            ).long()

        # Stack HSI images: (B, C, H, W)
        hsi_images = torch.stack([self.preprocess_hsi(x["image"]) for x in batched_input], dim=0)

        # ============ Step 1: Convert HSI to RGB ============
        with torch.no_grad():
            transform_matrix = interpolate_hyperspectral_image_transform_matrix(
                np.array(wavelengths), np.array([[700, 546.1, 438.8]])
            )[0]
            rgb_images = torch.stack([
                self.preprocess(
                    255 * torch.einsum(
                        's n, n w h -> s w h',
                        torch.tensor(transform_matrix).to(x["image"].device),
                        x["image"]
                    )
                )
                for x in batched_input
            ], dim=0)

        # ============ Step 2: Get HSI features from HSI encoder ============
        # HSI encoder returns features at each layer (already passed through ZeroMLP)
        hsi_layer_features = self.hsi_encoder(hsi_images, start_band=start_band, num_bands=num_bands)

        # ============ Step 3: Process RGB through RGB encoder with HSI fusion ============
        # RGB patch embedding: (B, H/patch, W/patch, embed_dim)
        x = self.rgb_encoder.patch_embed(rgb_images)

        # Add positional embedding
        if self.rgb_encoder.pos_embed is not None:
            x = x + self.rgb_encoder.pos_embed

        # Process through RGB encoder blocks with layer-wise HSI fusion
        for i, blk in enumerate(self.rgb_encoder.blocks):
            x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            # Add HSI features (ZeroMLP output) to RGB features
            x = x + hsi_layer_features[i]

        # Apply neck: (B, out_chans, H/patch, W/patch)
        image_embeddings = self.rgb_encoder.neck(x.permute(0, 3, 1, 2))

        # ============ Step 4: Extract spectral queries ============
        # Extract spectral queries at point locations
        spectral_queries = self._extract_spectral_queries(hsi_images, batched_input)

        # Fuse spectral queries into single token: (B, 1, embed_dim)
        fused_spectral_token = self.spectral_query_fusion(spectral_queries)

        # ============ Step 5: Process each image for mask prediction ============
        predictions = []
        for batch_idx, image_record in enumerate(batched_input):
            # Extract point prompts
            if "point_coords" in image_record:
                coords = image_record["point_coords"]
                labels = image_record["point_labels"]
                # Ensure batch dimension exists
                if coords.dim() == 2:
                    coords = coords.unsqueeze(0)
                if labels.dim() == 1:
                    labels = labels.unsqueeze(0)
                points = (coords, labels)
            else:
                points = None

            # Encode prompts
            sparse_prompt_embeddings, dense_prompt_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )

            # Add fused spectral token to sparse prompt embeddings
            spectral_token = fused_spectral_token[batch_idx:batch_idx+1]
            sparse_prompt_embeddings = sparse_prompt_embeddings + spectral_token.expand_as(sparse_prompt_embeddings)

            # Decode masks
            low_res_mask_logits, iou_predictions = self.mask_decoder(
                image_embeddings_rgb=image_embeddings[batch_idx].unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_prompt_embeddings,
                dense_prompt_embeddings=dense_prompt_embeddings,
                multimask_output=multimask_output,
                sam_token_only=False,
                only_hsi_module=False,
            )

            # Postprocess masks to original image size
            full_res_logits = self.postprocess_masks(
                low_res_mask_logits,
                input_size=input_size,
                original_size=image_record["original_size"],
            )
            binary_masks = full_res_logits > self.mask_threshold

            predictions.append({
                "masks": binary_masks,
                "low_res_logits": low_res_mask_logits,
                "rgb": rgb_images,
                "logits": full_res_logits,
                "iou_pred": iou_predictions,
                "low_res_masks": low_res_mask_logits,
            })

        return predictions

    def _extract_spectral_queries(
        self,
        hsi: torch.Tensor,
        batched_input: List[Dict[str, Any]],
    ) -> torch.Tensor:
        """
        Extract spectral queries from HSI at point locations.

        Args:
            hsi: (B, C, H, W) HSI images
            batched_input: List of input dictionaries with point_coords

        Returns:
            spectral_queries: (B, num_queries, C)
        """
        B, C, H, W = hsi.shape
        queries_list = []

        for batch_idx, image_record in enumerate(batched_input):
            if "point_coords" in image_record:
                coords = image_record["point_coords"][0]  # (N, 2)
                # Clamp coordinates to valid range
                x_coords = coords[:, 0].clamp(0, W - 1).long()
                y_coords = coords[:, 1].clamp(0, H - 1).long()

                # Extract spectral values at point locations: (N, C)
                spectral_values = hsi[batch_idx, :, y_coords, x_coords].T
                queries_list.append(spectral_values)
            else:
                # If no points, use zeros
                queries_list.append(torch.zeros(1, C, device=hsi.device))

        # Pad to same number of queries and stack
        max_queries = max(q.shape[0] for q in queries_list)
        padded_queries = []
        for q in queries_list:
            if q.shape[0] < max_queries:
                padding = torch.zeros(max_queries - q.shape[0], C, device=q.device)
                q = torch.cat([q, padding], dim=0)
            padded_queries.append(q)

        return torch.stack(padded_queries, dim=0)  # (B, max_queries, C)

    def postprocess_masks(
            self,
            masks: torch.Tensor,
            input_size: Tuple[int, ...],
            original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """Remove padding and upscale masks to the original image size."""
        masks = F.interpolate(
            masks,
            (self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize RGB pixel values and pad to square input."""
        x = (x - self.pixel_mean) / self.pixel_std
        height, width = x.shape[-2:]
        pad_height = self.img_size - height
        pad_width = self.img_size - width
        x = F.pad(x, (0, pad_width, 0, pad_height))
        return x

    def preprocess_hsi(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize hyperspectral image and pad to square input."""
        x = x / (x.max() + 1e-8)
        height, width = x.shape[-2:]
        pad_height = self.img_size - height
        pad_width = self.img_size - width
        x = F.pad(x, (0, pad_width, 0, pad_height))
        return x
