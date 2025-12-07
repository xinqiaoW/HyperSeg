import torch

from functools import partial

from .modeling import ImageEncoderViT, MaskDecoderHQ, PromptEncoder, HyperSeg, TwoWayTransformer, TinyViT, ImageEncoderRGB, ChannelProj

spectral_wavelength = [400, 412.5, 429.5, 443, 455, 467.5, 473.375, 481.25, 488.25, 
                       500, 520, 531, 536, 545, 550.5, 561.25, 564.75, 565.5, 575, 580, 
                       596, 605, 610, 612, 626, 627.5, 630, 635, 640, 645, 650, 655, 656, 
                       660, 664.5, 665, 667, 671.25, 677.5, 686, 700, 705, 710, 716, 725, 
                       730, 740, 748.5, 760, 764.25, 776, 783, 790, 808, 820, 825, 830, 
                       835.3125, 842, 850, 858.5, 865, 866, 869.5, 880, 896, 905, 910, 926, 
                       938, 945, 950, 959, 1240, 1375, 1575, 1575.5, 1610, 1640, 1650, 2050.25, 
                       2130, 2195, 2217.5,2500]

def load_and_resize_params(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_dict = model.state_dict()
    
    for k, v in checkpoint.items():
        if k in model_dict:
            if v.shape != model_dict[k].shape:
                if 'image_encoder.pos_embed' in k:
                    v = F.interpolate(v.permute((0,3,1,2)), size=(model_dict[k].shape[1], model_dict[k].shape[2]), mode='nearest').permute((0,2,3,1))
                elif 'rel_pos' in k:
                    v = F.interpolate(v.unsqueeze(0).unsqueeze(0), size=(model_dict[k].shape[0], model_dict[k].shape[1]),).squeeze(0).squeeze(0)
                elif 'weight_bank' in k:
                    v = F.interpolate(v, size=(model_dict[k].shape[2], model_dict[k].shape[3]), mode='nearest')

            model_dict[k] = v
    
    model.load_state_dict(model_dict, strict=False)
    return model

def _set_requires_grad(module, requires_grad: bool):
    """Utility to toggle gradient computation for a module."""
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = requires_grad


def build_seg_vit_h(sam_checkpoint=None,
                    hyperfree_checkpoint=None,
                    channel_proj_spectral=False, 
                    ignore_hsi_module=False,
                    ignore_spectral_query=False,
                    feature_as_query=False,
                    only_hsi_module=False):
    return _build_seg(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        sam_checkpoint=sam_checkpoint,
        hyperfree_checkpoint=hyperfree_checkpoint,
        channel_proj_spectral=channel_proj_spectral,
        ignore_hsi_module=ignore_hsi_module,
        ignore_spectral_query=ignore_spectral_query,
        feature_as_query=feature_as_query,
        only_hsi_module=only_hsi_module
    )


build_seg = build_seg_vit_h


def _build_seg(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    sam_checkpoint=None, # Image Encoder RGB | MaskDecoder HQ | Prompter Encoder
    hyperfree_checkpoint=None,# Image Encoder All Channel
    channel_proj_spectral=False,
    ignore_spectral_query=False,
    ignore_hsi_module=False,
    feature_as_query=False,
    only_hsi_module=False
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    seg = HyperSeg(
        ignore_hsi_module=ignore_hsi_module,
        ignore_spectral_query=ignore_spectral_query,
        feature_as_query=feature_as_query,
        only_hsi_module=only_hsi_module,
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
        image_encoder_all_channel=ImageEncoderViT(
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
            merge_indexs=None,
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
        channel_proj_spectral=None if (not channel_proj_spectral) else ChannelProj(
            embed_dim=256,
            hidden_dim=256,
            num_layers=4
        ),
        # pixel_mean=[123.675, 116.28, 103.53],
        # pixel_std=[58.395, 57.12, 57.375],
    )
    # seg.eval()
    # seg._init_weights()
    new_state_dict = {}
    if sam_checkpoint is not None:
        with open(sam_checkpoint, "rb") as f:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            state_dict = torch.load(f, map_location=device)

            for k in list(state_dict.keys()):
                # print(k)
                if k.startswith("image_encoder."):
                    new_state_dict["rgb_encoder." + k[14:]] = state_dict[k]
                    del state_dict[k]
                # else:
                #     new_state_dict[k] = state_dict[k]
        
        # info = seg.load_state_dict(new_state_dict, strict=False)
        # print(info)
    
    if hyperfree_checkpoint is not None:
        with open(hyperfree_checkpoint, "rb") as f:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            state_dict = torch.load(f, map_location=device)
        
        for k in list(state_dict.keys()):
            if k.startswith("image_encoder"):
                new_state_dict["spectral_encoder." + k[14:]] = state_dict[k]
            else:
                new_state_dict[k] = state_dict[k]
    
    info = seg.load_state_dict(new_state_dict, strict=False)

    # Freeze pretrained modules (SAM RGB encoder + HyperFree modules)
    _set_requires_grad(seg.rgb_encoder, False)
    _set_requires_grad(seg.spectral_encoder, False)
    _set_requires_grad(seg.mask_decoder, False)
    _set_requires_grad(seg.prompt_encoder, False)

    # Ensure learnable modules keep gradients enabled
    _set_requires_grad(seg.query_processor, True)
    _set_requires_grad(seg.feature_fusion, True)

    return seg