import torch

from functools import partial

from .modeling import MaskDecoderHQ, PromptEncoder, HyperSeg, TwoWayTransformer, ImageEncoderRGB


def _set_requires_grad(module, requires_grad: bool):
    """Utility to toggle gradient computation for a module."""
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = requires_grad


def build_seg_vit_h(sam_checkpoint=None, fixed_hsi_channels=224):
    return _build_seg(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        sam_checkpoint=sam_checkpoint,
        fixed_hsi_channels=fixed_hsi_channels,
    )


build_seg = build_seg_vit_h


def _build_seg(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    sam_checkpoint=None,
    fixed_hsi_channels=224,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    seg = HyperSeg(
        image_encoder_rgb=ImageEncoderRGB(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder_hq=MaskDecoderHQ(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            vit_dim=encoder_embed_dim,
        ),
        fixed_hsi_channels=fixed_hsi_channels,
    )

    # Load SAM checkpoint for RGB encoder, prompt encoder, and mask decoder
    new_state_dict = {}
    if sam_checkpoint is not None:
        with open(sam_checkpoint, "rb") as f:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            state_dict = torch.load(f, map_location=device)

            for k in list(state_dict.keys()):
                if k.startswith("image_encoder."):
                    new_state_dict["rgb_encoder." + k[14:]] = state_dict[k]
                else:
                    new_state_dict[k] = state_dict[k]

        info = seg.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded SAM checkpoint: {info}")

    # Freeze pretrained modules (SAM RGB encoder + prompt encoder)
    _set_requires_grad(seg.rgb_encoder, False)
    _set_requires_grad(seg.prompt_encoder, False)

    # Ensure learnable modules keep gradients enabled
    _set_requires_grad(seg.hsi_patch_embed, True)
    _set_requires_grad(seg.spectral_query_fusion, True)
    _set_requires_grad(seg.mask_decoder, True)

    return seg
