import torch
import copy

from functools import partial

from .modeling import MaskDecoderHQ, PromptEncoder, HyperSeg, TwoWayTransformer, ImageEncoderRGB
from .modeling.model import ImageEncoderHSI


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

    # Create RGB encoder (will be frozen)
    rgb_encoder = ImageEncoderRGB(
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
    )

    # Create HSI encoder (will be trainable, cloned from RGB encoder)
    hsi_encoder = ImageEncoderHSI(
        img_size=image_size,
        patch_size=vit_patch_size,
        fixed_channels=fixed_hsi_channels,
        embed_dim=encoder_embed_dim,
        depth=encoder_depth,
        num_heads=encoder_num_heads,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        act_layer=torch.nn.GELU,
        use_abs_pos=True,
        use_rel_pos=True,
        rel_pos_zero_init=True,
        window_size=14,
        global_attn_indexes=encoder_global_attn_indexes,
    )

    seg = HyperSeg(
        image_encoder_rgb=rgb_encoder,
        image_encoder_hsi=hsi_encoder,
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
    if sam_checkpoint is not None:
        with open(sam_checkpoint, "rb") as f:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            state_dict = torch.load(f, map_location=device)

            new_state_dict = {}
            for k in list(state_dict.keys()):
                if k.startswith("image_encoder."):
                    new_state_dict["rgb_encoder." + k[14:]] = state_dict[k]
                else:
                    new_state_dict[k] = state_dict[k]

        info = seg.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded SAM checkpoint: {info}")

        # Clone RGB encoder weights to HSI encoder (blocks and pos_embed)
        print("Cloning RGB encoder weights to HSI encoder...")

        # Clone positional embedding
        if seg.rgb_encoder.pos_embed is not None and seg.hsi_encoder.pos_embed is not None:
            seg.hsi_encoder.pos_embed.data.copy_(seg.rgb_encoder.pos_embed.data)

        # Clone transformer blocks
        for i, (rgb_block, hsi_block) in enumerate(zip(seg.rgb_encoder.blocks, seg.hsi_encoder.blocks)):
            hsi_block.load_state_dict(rgb_block.state_dict())

        print("HSI encoder initialized from RGB encoder weights")

    # Freeze RGB encoder (ControlNet-style: main branch is frozen)
    _set_requires_grad(seg.rgb_encoder, False)
    print("RGB encoder frozen")

    # Keep HSI encoder trainable (ControlNet-style: control branch is trainable)
    _set_requires_grad(seg.hsi_encoder, True)
    print("HSI encoder trainable")

    # Keep other modules trainable
    _set_requires_grad(seg.prompt_encoder, True)
    _set_requires_grad(seg.spectral_query_fusion, True)
    _set_requires_grad(seg.mask_decoder, True)

    return seg
