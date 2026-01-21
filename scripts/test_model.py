"""
Test script for the new ControlNet-like HyperSeg model structure.
Uses random HSI tensors to verify the model works correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
from hyperseg import build_seg_vit_h


def test_hyperseg_forward():
    """Test the forward pass of HyperSeg with random HSI data."""
    print("=" * 60)
    print("Testing HyperSeg with ControlNet-like structure")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Build model (without loading SAM checkpoint for testing)
    print("\n[1] Building model...")
    model = build_seg_vit_h(sam_checkpoint=None, fixed_hsi_channels=224)
    model = model.to(device)
    model.eval()
    print(f"    Model built successfully!")
    print(f"    - img_size: {model.img_size}")
    print(f"    - patch_size: {model.patch_size}")
    print(f"    - embed_dim: {model.embed_dim}")
    print(f"    - vit_embed_dim: {model.vit_embed_dim}")
    print(f"    - fixed_hsi_channels: {model.fixed_hsi_channels}")

    # Create random HSI image with variable channels
    batch_size = 2
    num_channels = 112  # Variable number of channels (will be interpolated to 224)
    height, width = 256, 256

    print(f"\n[2] Creating random HSI data...")
    print(f"    - Batch size: {batch_size}")
    print(f"    - HSI channels: {num_channels} (will be interpolated to {model.fixed_hsi_channels})")
    print(f"    - Image size: {height}x{width}")

    # Create random HSI images
    hsi_images = torch.rand(batch_size, num_channels, height, width).to(device)

    # Create random wavelengths
    wavelengths = [400.0 + (2100.0 / num_channels) * i for i in range(num_channels)]

    # Create random point prompts - shape should be (N, 2) for each sample
    # where N is number of points
    num_points = 2
    point_coords = torch.randint(0, min(height, width), (batch_size, num_points, 2)).float().to(device)
    point_labels = torch.ones(batch_size, num_points).to(device)

    # Build batched input
    batched_input = []
    for i in range(batch_size):
        batched_input.append({
            "image": hsi_images[i],
            "original_size": (height, width),
            "point_coords": point_coords[i],  # (N, 2)
            "point_labels": point_labels[i],  # (N,)
        })

    print(f"\n[3] Running forward pass...")
    with torch.no_grad():
        outputs = model(batched_input, wavelengths=wavelengths, multimask_output=False)

    print(f"    Forward pass successful!")
    print(f"\n[4] Output shapes:")
    for i, out in enumerate(outputs):
        print(f"    Sample {i}:")
        print(f"      - masks: {out['masks'].shape}")
        print(f"      - logits: {out['logits'].shape}")
        print(f"      - low_res_logits: {out['low_res_logits'].shape}")
        print(f"      - iou_pred: {out['iou_pred'].shape}")
        print(f"      - rgb: {out['rgb'].shape}")

    # Test with different number of HSI channels
    print(f"\n[5] Testing with different HSI channel counts...")
    for num_ch in [50, 112, 224, 300]:
        hsi_test = torch.rand(1, num_ch, 128, 128).to(device)
        wavelengths_test = [400.0 + (2100.0 / num_ch) * i for i in range(num_ch)]
        point_test = torch.randint(0, 128, (1, 1, 2)).float().to(device)
        label_test = torch.ones(1, 1).to(device)

        batched_input_test = [{
            "image": hsi_test[0],
            "original_size": (128, 128),
            "point_coords": point_test[0],
            "point_labels": label_test[0],
        }]

        with torch.no_grad():
            out_test = model(batched_input_test, wavelengths=wavelengths_test, multimask_output=False)
        print(f"    {num_ch} channels -> mask shape: {out_test[0]['masks'].shape} ✓")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


def test_model_components():
    """Test individual model components."""
    print("\n" + "=" * 60)
    print("Testing individual model components")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Import components
    from hyperseg.modeling.model import (
        ZeroConv2d, ZeroMLP, HSIPatchEmbed, SpectralQueryFusion
    )

    # Test ZeroConv2d
    print("\n[1] Testing ZeroConv2d...")
    zero_conv = ZeroConv2d(256, 256, kernel_size=1).to(device)
    x = torch.rand(2, 256, 64, 64).to(device)
    out = zero_conv(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6), "ZeroConv should output zeros initially"
    print(f"    Input: {x.shape} -> Output: {out.shape}")
    print(f"    Output is zero: {torch.allclose(out, torch.zeros_like(out), atol=1e-6)} ✓")

    # Test ZeroMLP
    print("\n[2] Testing ZeroMLP...")
    zero_mlp = ZeroMLP(256, 1024, 256, num_layers=2).to(device)
    x = torch.rand(2, 10, 256).to(device)
    out = zero_mlp(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6), "ZeroMLP should output zeros initially"
    print(f"    Input: {x.shape} -> Output: {out.shape}")
    print(f"    Output is zero: {torch.allclose(out, torch.zeros_like(out), atol=1e-6)} ✓")

    # Test HSIPatchEmbed
    print("\n[3] Testing HSIPatchEmbed...")
    hsi_embed = HSIPatchEmbed(
        img_size=1024,
        patch_size=16,
        fixed_channels=224,
        embed_dim=1280
    ).to(device)

    # Test with different channel counts
    for num_ch in [50, 112, 224, 300]:
        x = torch.rand(2, num_ch, 256, 256).to(device)
        out = hsi_embed(x)
        expected_h = 256 // 16
        expected_w = 256 // 16
        assert out.shape == (2, 1280, expected_h, expected_w), f"Shape mismatch: {out.shape}"
        print(f"    Input: {x.shape} -> Output: {out.shape} ✓")

    # Test channel interpolation
    print("\n[4] Testing channel interpolation...")
    x = torch.rand(2, 100, 64, 64).to(device)
    x_interp = hsi_embed.interpolate_channels(x)
    assert x_interp.shape == (2, 224, 64, 64), f"Shape mismatch: {x_interp.shape}"
    print(f"    Input: {x.shape} -> Interpolated: {x_interp.shape} ✓")

    # Test SpectralQueryFusion
    print("\n[5] Testing SpectralQueryFusion...")
    query_fusion = SpectralQueryFusion(
        num_bands=224,
        embed_dim=256,
        num_heads=8,
        mlp_dim=2048
    ).to(device)

    queries = torch.rand(2, 5, 224).to(device)  # 5 queries per sample
    fused = query_fusion(queries)
    assert fused.shape == (2, 1, 256), f"Shape mismatch: {fused.shape}"
    print(f"    Input queries: {queries.shape} -> Fused token: {fused.shape} ✓")

    print("\n" + "=" * 60)
    print("All component tests passed!")
    print("=" * 60)


def test_gradient_flow():
    """Test that gradients flow through the trainable components."""
    print("\n" + "=" * 60)
    print("Testing gradient flow")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build model
    model = build_seg_vit_h(sam_checkpoint=None, fixed_hsi_channels=224)
    model = model.to(device)
    model.train()

    # Create dummy data
    hsi = torch.rand(1, 112, 128, 128).to(device)
    wavelengths = [400.0 + (2100.0 / 112) * i for i in range(112)]
    point_coords = torch.randint(0, 128, (1, 1, 2)).float().to(device)
    point_labels = torch.ones(1, 1).to(device)

    batched_input = [{
        "image": hsi[0],
        "original_size": (128, 128),
        "point_coords": point_coords[0],
        "point_labels": point_labels[0],
    }]

    # Forward pass
    outputs = model(batched_input, wavelengths=wavelengths, multimask_output=False)

    # Compute dummy loss
    loss = outputs[0]["logits"].mean()

    # Backward pass
    loss.backward()

    # Check gradients for trainable modules
    print("\n[1] Checking gradients for trainable modules...")

    # HSI Patch Embed
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.hsi_patch_embed.parameters() if p.requires_grad)
    print(f"    hsi_patch_embed has gradients: {has_grad} ✓" if has_grad else f"    hsi_patch_embed has gradients: {has_grad} ✗")

    # Spectral Query Fusion
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.spectral_query_fusion.parameters() if p.requires_grad)
    print(f"    spectral_query_fusion has gradients: {has_grad} ✓" if has_grad else f"    spectral_query_fusion has gradients: {has_grad} ✗")

    # Mask Decoder
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.mask_decoder.parameters() if p.requires_grad)
    print(f"    mask_decoder has gradients: {has_grad} ✓" if has_grad else f"    mask_decoder has gradients: {has_grad} ✗")

    print("\n[2] Checking frozen modules have no gradients...")

    # RGB Encoder (should be frozen)
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.rgb_encoder.parameters())
    print(f"    rgb_encoder has gradients: {has_grad} (should be False) {'✓' if not has_grad else '✗'}")

    # Prompt Encoder (should be frozen)
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.prompt_encoder.parameters())
    print(f"    prompt_encoder has gradients: {has_grad} (should be False) {'✓' if not has_grad else '✗'}")

    print("\n" + "=" * 60)
    print("Gradient flow test completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_model_components()
    test_hyperseg_forward()
    test_gradient_flow()
