import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List, Tuple

from .image_encoder_rgb import ImageEncoderRGB
from .image_encoder_spectral import ImageEncoderViT
from .mask_decoder_hq import MaskDecoderHQ
from .channel_proj import ChannelProj
from .prompt_encoder import PromptEncoder
from .positional_encoding import PositionalEncoding
from .query_processor import QueryProcessor
from .feature_processor import SpectralFeatureFusion
from ..utils.spectral_process_utils import interpolate_hyperspectral_image_transform_matrix
from ..utils.transforms import ResizeLongestSide


class HyperSeg(nn.Module):
    """HyperSeg: Hyperspectral Image Segmentation Model.
    
    This model combines RGB and hyperspectral encoders with spectral query processing
    for accurate segmentation of hyperspectral images. It supports multi-modal feature
    fusion and scale-aware processing.

    Args:
        image_encoder_rgb (ImageEncoderRGB): RGB image encoder backbone.
        prompt_encoder (PromptEncoder): SAM-style prompt encoder.
        mask_decoder_hq (MaskDecoderHQ): High-quality mask decoder.
        image_encoder_all_channel (ImageEncoderViT): Hyperspectral image encoder.
        pixel_mean (List[float]): Mean values for RGB normalization.
        pixel_std (List[float]): Standard deviation for RGB normalization.
        input_channels (int): Number of input spectral channels.
        channel_proj_spectral (ChannelProj): Optional channel projection module.
        image_format (str): Image format (default: "RGB").
        ignore_spectral_query (bool): If True, skip spectral query processing.
        ignore_hsi_module (bool): If True, skip HSI module in decoder.
        feature_as_query (bool): If True, use spatial features as queries.
        only_hsi_module (bool): If True, use only HSI module in decoder.
    """
    
    mask_threshold: float = 0.0

    def __init__(
            self,
            image_encoder_rgb: ImageEncoderRGB,
            prompt_encoder: PromptEncoder,
            mask_decoder_hq: MaskDecoderHQ,
            image_encoder_all_channel: ImageEncoderViT,
            pixel_mean: List[float] = [123.675, 116.28, 103.53],
            pixel_std: List[float] = [58.395, 57.12, 57.375],
            input_channels: int = 112,
            channel_proj_spectral: ChannelProj = None,
            image_format: str = "RGB",
            ignore_spectral_query: bool = False,
            ignore_hsi_module: bool = False,
            feature_as_query: bool = False,
            only_hsi_module: bool = False,
    ) -> None:
        super(HyperSeg, self).__init__()

        # ============ RGB Normalization Parameters ============
        # Register as buffers (not model parameters)
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # ============ Model Configuration Flags ============
        self.ignore_spectral_query = ignore_spectral_query  # Skip spectral query fusion
        self.ignore_hsi_module = ignore_hsi_module  # Use only RGB features in decoder
        self.feature_as_query = feature_as_query  # Use spatial features instead of point queries
        self.only_hsi_module = only_hsi_module  # Use only HSI features in decoder

        # ============ Core Encoder and Decoder Modules ============
        self.rgb_encoder = image_encoder_rgb  # RGB image encoder
        self.spectral_encoder = image_encoder_all_channel  # Hyperspectral encoder
        self.prompt_encoder = prompt_encoder  # SAM prompt encoder
        self.mask_decoder = mask_decoder_hq  # High-quality mask decoder

        # ============ Model Architecture Parameters ============
        self.img_size = image_encoder_rgb.img_size  # Target image size
        self.patch_size = image_encoder_all_channel.patch_size  # Patch size for tokenization
        self.image_format = image_format  # Image channel format
        self.embed_dim = self.prompt_encoder.embed_dim  # Embedding dimension

        # ============ Spectral Query Processing ============
        # Module for extracting and projecting spectral queries from point prompts
        self.query_processor = QueryProcessor(
            input_channels=input_channels,
            embed_dim=self.embed_dim,
            channel_proj=channel_proj_spectral
        )

        # ============ Multi-Modal Feature Fusion ============
        # Cross-attention based fusion of spectral queries and spatial features
        self.feature_fusion = SpectralFeatureFusion(
            embed_dim=self.embed_dim,
            num_layers=3,
            num_heads=8,
            mlp_dim=2048
        )

        # ============ Image Preprocessing Utilities ============
        self.transform = ResizeLongestSide(self.img_size)  # Image resizing transform
        self.positional_encoder = PositionalEncoding(d_model=self.embed_dim)  # Positional encoding


    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return self.feature_fusion.device


    def forward(
            self,
            batched_input: List[Dict[str, Any]],
            wavelengths: List[float],
            multimask_output: bool,
            test_mode: bool = False,
            GSD: torch.Tensor = torch.tensor([1.0])
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.
          wavelengths (list): List of wavelengths corresponding to the input image.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        # transform the input image to the correct size and format
        for image_record in batched_input:
            # Apply the transform and remove batch dimension
            image_record["image"] = self.transform.apply_image_torch(image_record["image"].unsqueeze(0)).squeeze(0)
            input_size = image_record["image"].shape[-2:]
            image_record["point_coords"] = self.transform.apply_coords_torch(
                image_record["point_coords"], (image_record["original_size"])
            ).long()

        # Extract and process queries
        input_images = torch.stack([self.preprocess_hsi(x["image"]) for x in batched_input], dim=0)
        if not self.feature_as_query:
            queries = self.query_processor.extract_point_queries(
                [x["image"] for x in batched_input], batched_input
            )

        # Project queries to embedding dimension
        if not self.feature_as_query:
            # Ensure wavelengths are provided as a tensor for ChannelProj
            if isinstance(wavelengths, torch.Tensor):
                wavelength_tensor = wavelengths.to(device=queries.device, dtype=queries.dtype)
            else:
                wavelength_tensor = torch.as_tensor(
                    wavelengths, device=queries.device, dtype=queries.dtype
                )
            queries = self.query_processor.project_queries(queries, wavelength_tensor, test_mode)

        # Process the input images to get RGB embeddings
        with torch.no_grad():
            transform_matrix = interpolate_hyperspectral_image_transform_matrix(
                np.array(wavelengths), np.array([[700, 546.1, 438.8]])
            )[0]
            input_images_rgb = torch.stack([
                self.preprocess(
                    255 * torch.einsum('s n, n w h -> s w h', torch.tensor(transform_matrix).to(x["image"].device),
                                       x["image"]))
                for x in batched_input
            ], dim=0)

            image_embeddings_rgb, _ = self.rgb_encoder(input_images_rgb)
            B, C, H, W = image_embeddings_rgb.shape

        # Process the input images to get spectral embeddings (multi-stage features)
        multi_stage_features = []
        for batch_idx in range(GSD.shape[0]):
            stage_features = self.spectral_encoder(
                input_images[batch_idx].unsqueeze(0),
                test_mode=test_mode,
                input_wavelength=wavelengths,
                GSD=GSD[batch_idx]
            )
            # Collect features from different stages
            for stage_idx in range(len(stage_features)):
                if multi_stage_features == []:
                    multi_stage_features = [[] for _ in range(len(stage_features))]
                multi_stage_features[stage_idx].append(stage_features[stage_idx])

        # Concatenate features from all batches
        for stage_idx, stage_feature in enumerate(multi_stage_features):
            multi_stage_features[stage_idx] = torch.cat(stage_feature, dim=0)

        # Use the last stage features as main spatial embeddings
        spatial_embeddings = multi_stage_features[-1]
        # Extract queries from spatial features if using feature-based queries
        if self.feature_as_query:
            with torch.no_grad():
                queries = self.query_processor.extract_feature_queries(
                    spatial_embeddings, batched_input, self.img_size
                )

        # Fuse spectral queries with spatial embeddings using cross-attention
        if not self.ignore_spectral_query:
            # Generate scale-aware positional embeddings for keys
            key_positional_embeds = self.feature_fusion.generate_scale_aware_pos_embed(
                GSD, self.img_size, self.patch_size
            )

            # Fuse queries with spatial features and get high-frequency correction token
            queries, fused_embeddings, high_freq_token = self.feature_fusion.fuse_features(
                queries, spatial_embeddings, key_positional_embeds
            )
        else:
            # Skip query fusion, directly upsample spatial embeddings
            fused_embeddings = self.feature_fusion.upsampler(spatial_embeddings)

        # Process each image in the batch
        predictions = []
        for batch_idx, image_record in enumerate(batched_input):
            # Extract point prompts if available
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None

            # Use pre-computed fused embeddings
            current_fused_embedding = fused_embeddings[batch_idx].unsqueeze(0)

            # Encode prompts (points, boxes, masks)
            sparse_prompt_embeddings, dense_prompt_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )

            # Decode masks using both RGB and spectral features
            low_res_mask_logits, iou_predictions = self.mask_decoder(
                image_embeddings_rgb=image_embeddings_rgb[batch_idx].unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_prompt_embeddings,
                dense_prompt_embeddings=dense_prompt_embeddings,
                multimask_output=multimask_output,
                sam_token_only=self.ignore_hsi_module,
                only_hsi_module=self.only_hsi_module,
            )

            # Apply high-frequency error correction
            if not self.ignore_spectral_query:
                error_correction_map = self.feature_fusion.compute_error_correction(
                    high_freq_token[batch_idx].unsqueeze(0), current_fused_embedding
                )
                low_res_mask_logits = low_res_mask_logits + error_correction_map

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
                "rgb": input_images_rgb,
                "logits": full_res_logits,
                "iou_pred": iou_predictions,
                "low_res_masks": low_res_mask_logits,
            })
        return predictions, multi_stage_features

    # ============ Debugging Utilities ============
    
    def module_grad_hook(self):
        """Register gradient hooks for debugging gradient flow."""
        self.spectral_encoder.register_full_backward_hook(
            lambda module, grad_input, grad_output: print(
                f"spectral_encoder - grad_output: {grad_output[0]} grad_input: {grad_input[0]}"
            )
        )
        self.feature_fusion.cross_attention_fusion.register_full_backward_hook(
            lambda module, grad_input, grad_output: print(
                f"cross_attention_fusion - grad_output: {grad_output[0]} grad_input: {grad_input[0]}"
            )
        )
        self.feature_fusion.upsampler.register_full_backward_hook(
            lambda module, grad_input, grad_output: print(
                f"upsampler - grad_output: {grad_output[0]} grad_input: {grad_input[0]}"
            )
        )
        self.feature_fusion.query_encoder.register_full_backward_hook(
            lambda module, grad_input, grad_output: print(
                f"query_encoder - grad_output: {grad_output[0]} grad_input: {grad_input[0]}"
            )
        )
        self.feature_fusion.cross_attention_fusion.register_forward_hook(
            lambda module, input, output: print(
                f"cross_attention_fusion - input: {input[0]} output: {output[0]}"
            )
        )

    # ============ Postprocessing Methods ============
    
    def postprocess_masks(
            self,
            masks: torch.Tensor,
            input_size: Tuple[int, ...],
            original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """Remove padding and upscale masks to the original image size.

        Args:
            masks (torch.Tensor): Batched mask logits from decoder [B, C, H, W].
            input_size (Tuple[int, int]): Size of image input to model (H, W).
            original_size (Tuple[int, int]): Original image size before resizing (H, W).

        Returns:
            torch.Tensor: Upscaled masks in original resolution [B, C, H, W].
        """
        # First, upsample to model's standard size
        masks = F.interpolate(
            masks,
            (self.spectral_encoder.img_size, self.spectral_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        # Remove padding
        masks = masks[..., : input_size[0], : input_size[1]]
        # Upsample to original size
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    # ============ Preprocessing Methods ============
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize RGB pixel values and pad to square input.

        Args:
            x (torch.Tensor): Input RGB image tensor [C, H, W].

        Returns:
            torch.Tensor: Normalized and padded image tensor.
        """
        # Normalize using ImageNet statistics
        x = (x - self.pixel_mean) / self.pixel_std
        
        # Pad to square size
        height, width = x.shape[-2:]
        pad_height = self.spectral_encoder.img_size - height
        pad_width = self.spectral_encoder.img_size - width
        x = F.pad(x, (0, pad_width, 0, pad_height))
        return x

    def preprocess_hsi(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize hyperspectral image and pad to square input.

        Args:
            x (torch.Tensor): Input hyperspectral image tensor [C, H, W].

        Returns:
            torch.Tensor: Normalized and padded image tensor.
        """
        # Normalize to [0, 1] range
        x = x / x.max()
        
        # Pad to square size
        height, width = x.shape[-2:]
        pad_height = self.spectral_encoder.img_size - height
        pad_width = self.spectral_encoder.img_size - width
        x = F.pad(x, (0, pad_width, 0, pad_height))
        return x
