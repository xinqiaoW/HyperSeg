import argparse
import os
import random
import subprocess
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from hyperseg import build_seg_vit_h
from hyperseg.modeling.dataset import CustomDataset, SpaceNetDataset
from hyperseg.utils.focal_loss import loss_masks
from hyperseg.utils.tools import (
    Meaner,
    build_input_for_hyperseg,
    command,
    create_point_coords,
    seg_call,
    transform_output_seg,
    update_points,
)


def log_module_param_memory(
    model: nn.Module,
    log_dir: str = './logs',
    filename: str = 'module_param_memory.txt'
) -> None:
    """Log parameter and buffer memory usage for each module."""
    os.makedirs(log_dir, exist_ok=True)
    lines = []
    total_bytes = 0

    for name, module in model.named_modules():
        param_bytes = sum(p.numel() * p.element_size() for p in module.parameters(recurse=False))
        buffer_bytes = sum(b.numel() * b.element_size() for b in module.buffers(recurse=False))
        module_bytes = param_bytes + buffer_bytes

        if module_bytes == 0:
            continue

        total_bytes += module_bytes
        module_name = name if name else '<root>'
        lines.append(
            f"{module_name}: {module_bytes / (1024 ** 2):.4f} MB "
            f"(params {param_bytes / (1024 ** 2):.4f} MB, buffers {buffer_bytes / (1024 ** 2):.4f} MB)"
        )

    lines.append(f"TOTAL: {total_bytes / (1024 ** 2):.4f} MB")

    with open(os.path.join(log_dir, filename), 'w') as f:
        f.write('\n'.join(lines))


def compute_best_mask_loss(
    logits: torch.Tensor,
    bmasks: torch.Tensor,
    gt: torch.Tensor,
) -> tuple[torch.Tensor, float, float, list[int], torch.Tensor]:
    """
    Compute loss on all masks and select the best mask per sample.

    Args:
        logits: Predicted logits [B, 3, H, W]
        bmasks: Predicted masks [B, 3, H, W]
        gt: Ground truth masks [B, H, W]

    Returns:
        total_loss: Sum of best losses for all samples (for backprop)
        avg_mask_loss: Average mask loss across batch
        avg_dice_loss: Average dice loss across batch
        best_mask_indices: List of best mask index for each sample
        best_bmasks: Best masks selected for each sample [B, H, W]
    """
    B = gt.shape[0]
    gt_expanded = gt.float().unsqueeze(1)  # [B, 1, H, W]

    total_loss = 0
    total_mask_loss = 0
    total_dice_loss = 0
    best_mask_indices = []

    for b in range(B):
        sample_losses = []
        for mask_idx in range(3):
            mask_logits = logits[b:b+1, mask_idx:mask_idx+1, :, :]  # [1, 1, H, W]
            sample_gt = gt_expanded[b:b+1]  # [1, 1, H, W]
            loss_mask_i, loss_dice_i = loss_masks(mask_logits, sample_gt, num_masks=1)
            loss_i = loss_mask_i + loss_dice_i
            sample_losses.append((loss_i, loss_mask_i, loss_dice_i, mask_idx))

        # Find the mask with smallest loss for this sample
        min_loss, min_mask_loss, min_dice_loss, best_idx = min(sample_losses, key=lambda x: x[0].item())
        total_loss = total_loss + min_loss
        total_mask_loss += min_mask_loss.item()
        total_dice_loss += min_dice_loss.item()
        best_mask_indices.append(best_idx)

    # Select best masks for each sample
    best_bmasks = torch.stack([bmasks[b, best_mask_indices[b], :, :] for b in range(B)], dim=0)

    return total_loss, total_mask_loss / B, total_dice_loss / B, best_mask_indices, best_bmasks


def register_activation_memory_hooks(model: nn.Module) -> tuple[dict, list]:
    """Register forward hooks to track activation memory usage."""
    module_to_name = {module: (name if name else '<root>') for name, module in model.named_modules()}
    activation_mem = {}

    def accumulate_tensor_mem(obj) -> int:
        if isinstance(obj, torch.Tensor):
            return obj.numel() * obj.element_size()
        if isinstance(obj, (list, tuple)):
            return sum(accumulate_tensor_mem(x) for x in obj)
        if isinstance(obj, dict):
            return sum(accumulate_tensor_mem(v) for v in obj.values())
        return 0

    def hook(module, inputs, outputs):
        name = module_to_name.get(module, '<unnamed>')
        mem = accumulate_tensor_mem(outputs)
        if mem > activation_mem.get(name, 0):
            activation_mem[name] = mem

    hooks = []
    for module in model.modules():
        if not list(module.children()):
            hooks.append(module.register_forward_hook(hook))

    return activation_mem, hooks


def _log_first_batch_memory(
    activation_mem: dict,
    activation_hooks: list,
    device: str,
    log_dir: str = './logs'
) -> None:
    """Log CUDA and activation memory after first batch."""
    try:
        if torch.cuda.is_available():
            mem_summary = torch.cuda.memory_summary(device=device, abbreviated=False)
            with open(os.path.join(log_dir, 'cuda_mem_summary_first_batch.txt'), 'w') as f:
                f.write(mem_summary)

        if activation_mem:
            lines = []
            total_bytes = 0
            for name, bytes_ in activation_mem.items():
                total_bytes += bytes_
                lines.append(f"{name}: {bytes_ / (1024 ** 2):.4f} MB")
            lines.append(f"TOTAL: {total_bytes / (1024 ** 2):.4f} MB")
            with open(os.path.join(log_dir, 'module_activation_memory_first_batch.txt'), 'w') as f:
                f.write('\n'.join(lines))
    except Exception as e:
        print(f"Failed to write CUDA memory summary: {e}")
    finally:
        for h in activation_hooks:
            h.remove()


def _get_checkpoint_name(save_name: str | None, epoch: int, batch_idx: int | None = None) -> str:
    """Generate checkpoint filename."""
    base = save_name if save_name else "net"
    if batch_idx is not None:
        return f"{base}_{epoch}_{batch_idx}.pth"
    return f"{base}_{epoch}.pth"


def train(
    net: nn.Module,
    lr: float,
    epochs: int,
    dataloader,
    optimizer,
    wavelengths,
    point_num: int | None,
    build_input,
    transform_output,
    net_call,
    save_checkpoint_path: str,
    save_name: str | None = None,
    device: str = 'cuda',
    show_rgb: bool = False,
    start_validation: bool = False,
    dataset_type: str = 'custom',
    num_iterations: int = 2,
    out_images: str = './outputs/out_images',
    select_bands: int | None = None,
) -> None:
    """
    Train the segmentation network.

    Args:
        net: The neural network model
        lr: Learning rate
        epochs: Number of training epochs
        dataloader: Data loader for training data
        optimizer: Optimizer instance
        wavelengths: Wavelength information for HSI
        point_num: Number of prompt points (unused, kept for API compatibility)
        build_input: Function to build model input
        transform_output: Function to transform model output
        net_call: Function to call the model
        save_checkpoint_path: Directory to save checkpoints
        save_name: Base name for saved checkpoints
        device: Device to train on
        show_rgb: Whether to save RGB visualizations
        start_validation: Whether to run validation during training
        dataset_type: Type of dataset ('custom' or 'spacenet')
        num_iterations: Number of point refinement iterations per batch
        out_images: Directory to save output images
        select_bands: Number of continuous bands to randomly select (None = use all)
    """
    net.train()
    total_batches = len(dataloader) * epochs
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=total_batches, eta_min=1e-6)

    mean_loss = Meaner()
    mean_dice_loss = Meaner()
    mean_mask_loss = Meaner()

    first_batch_logged = False
    activation_mem, activation_hooks = register_activation_memory_hooks(net)

    for epoch in range(epochs):
        for batch_idx, batch_data in enumerate(dataloader):
            # Load batch data based on dataset type
            if dataset_type == 'spacenet':
                images, gt = batch_data
                images = images.to(device)
                gt = gt.to(device)
            else:
                (images, _gsd), gt = batch_data
                images = images.to(device)
                gt = gt.to(device)

            B, H, W = gt.shape
            C = images.shape[1]  # Number of HSI channels

            # Generate random band selection if enabled
            start_band, num_bands = None, None
            if select_bands is not None and select_bands < C:
                max_start = C - select_bands
                start_band = random.randint(0, max_start)
                num_bands = select_bands

            # Generate point prompts
            point_coords = create_point_coords(gt, num_points=1).to(device)
            point_labels = torch.ones((B, point_coords.shape[1]), device=device)

            mean_loss.reset()
            mean_dice_loss.reset()
            mean_mask_loss.reset()

            optimizer.zero_grad()

            # Iterative point refinement
            rgb, mask, gt_show = None, None, None
            for _ in range(num_iterations):
                batched_input = build_input(
                    images, shape=(H, W),
                    point_coords=point_coords,
                    point_labels=point_labels,
                    device=device
                )
                batched_output = net_call(
                    net, batched_input, wavelengths=wavelengths, multimask_output=True,
                    start_band=start_band, num_bands=num_bands
                )
                # logits: [B, 3, H, W], bmasks: [B, 3, H, W]
                logits, bmasks = transform_output(batched_output, multimask_output=True)

                # Compute best mask loss per sample and aggregate
                total_loss, avg_mask_loss, avg_dice_loss, best_mask_indices, best_bmasks = \
                    compute_best_mask_loss(logits, bmasks, gt)
                loss = total_loss / (B * num_iterations)

                mean_loss.update(loss.item())
                mean_dice_loss.update(avg_dice_loss)
                mean_mask_loss.update(avg_mask_loss)

                if show_rgb:
                    rgb = torch.flip(batched_output[0]["rgb"][0].permute(1, 2, 0), dims=[-1]).detach().cpu().numpy()
                    mask = batched_output[0]["masks"][0, best_mask_indices[0], :, :].detach().int().cpu().numpy()
                    gt_show = gt[0].detach().cpu().numpy()

                del batched_input, batched_output
                loss.backward()
                point_coords, point_labels = update_points(point_coords, point_labels, best_bmasks, gt)

            optimizer.step()
            lr_scheduler.step()

            # Log memory usage after first batch
            if not first_batch_logged:
                _log_first_batch_memory(activation_mem, activation_hooks, device)
                first_batch_logged = True

            print(
                f"Batch {batch_idx} | "
                f"mask_loss: {mean_mask_loss.avg:.4f} | "
                f"dice_loss: {mean_dice_loss.avg:.4f} | "
                f"total_loss: {mean_loss.avg:.4f} | "
                f"lr: {optimizer.param_groups[0]['lr']:.6f}"
            )

            # Save checkpoints at intervals
            if batch_idx % 50 == 0:
                # Run validation every 400 batches
                if batch_idx % 400 == 0 and start_validation:
                    temp_path = os.path.join(save_checkpoint_path, f"temp_{epoch}_{batch_idx}.pth")
                    torch.save(net.state_dict(), temp_path)
                    subprocess.Popen(command(temp_path), shell=True)

                # Save RGB visualizations
                if show_rgb and rgb is not None:
                    cv2.imwrite(os.path.join(out_images, f"rgb_{epoch}_{batch_idx}.png"), (rgb * 255).astype(np.uint8))
                    cv2.imwrite(os.path.join(out_images, f"mask_{epoch}_{batch_idx}.png"), (mask * 255).astype(np.uint8))
                    cv2.imwrite(os.path.join(out_images, f"gt_{epoch}_{batch_idx}.png"), (gt_show * 255).astype(np.uint8))

                # Save periodic checkpoint every 5000 batches
                if batch_idx % 5000 == 0:
                    ckpt_name = _get_checkpoint_name(save_name, epoch, batch_idx)
                    torch.save(net.state_dict(), os.path.join(save_checkpoint_path, ckpt_name))

        # Save checkpoint after each epoch
        ckpt_name = _get_checkpoint_name(save_name, epoch)
        torch.save(net.state_dict(), os.path.join(save_checkpoint_path, ckpt_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HyperSeg model")

    # Model arguments
    parser.add_argument(
        '--sam_checkpoint', type=str, default='./checkpoints/sam_vit_h_4b8939.pth',
        help='path to the SAM checkpoint'
    )
    parser.add_argument(
        '--fixed_hsi_channels', type=int, default=224,
        help='fixed number of HSI channels for interpolation'
    )
    parser.add_argument(
        '--wavelengths', type=str,
        default=[400.0 + (2100.0 / 224.0) * i for i in range(0, 224)],
        help='wavelengths of the input image'
    )

    # Training arguments
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--num_iterations', type=int, default=2, help='point refinement iterations per batch')
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers for data loading')
    parser.add_argument('--point_num', type=int, default=None, help='number of points to sample')

    # Checkpoint arguments
    parser.add_argument('--save_checkpoint_path', type=str, default='./checkpoints', help='path to save checkpoints')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/net_2_10000.pth', help='checkpoint to load')
    parser.add_argument('--load_state_dict', action='store_true', help='load state dict from checkpoint')
    parser.add_argument('--save_name', type=str, default=None, help='base name for saved checkpoints')
    parser.add_argument('--start_epoch', type=int, default=0, help='starting epoch number')

    # Model configuration
    parser.add_argument('--parallel', action='store_true', help='use DataParallel')
    parser.add_argument('--frozen', action='store_true', help='freeze the whole model')

    # Output arguments
    parser.add_argument('--out_images', type=str, default='./outputs/out_images', help='path to save output images')
    parser.add_argument('--show_rgb', action='store_true', help='save RGB visualizations')
    parser.add_argument('--start_validation', action='store_true', help='run validation during training')

    # Custom dataset arguments
    parser.add_argument('--hsi_path', type=str, default="/data2/pl/HSITask/data/HyperFree/data_compressed", help='path to HSI images')
    parser.add_argument('--gt_path', type=str, default="/data2/pl/HSITask/data/HyperFree/labels_hf_nms", help='path to ground truth masks')
    parser.add_argument('--index_file', type=str, default='../data/hyperfree_index.json', help='path to index file')

    # SpaceNet dataset arguments
    parser.add_argument('--dataset_type', type=str, default='custom', choices=['custom', 'spacenet'], help='dataset type')
    parser.add_argument('--spacenet_gt_dir', type=str, default='./data/SpaceNet/ground_truth', help='SpaceNet ground truth directory')
    parser.add_argument('--spacenet_hsi_dir', type=str, default='./data/SpaceNet/hsi_info', help='SpaceNet HSI info directory')
    parser.add_argument('--spacenet_endmember_lib', type=str, default='./data/SpaceNet/hsi_info/endmember_libraries.npz', help='SpaceNet endmember library')
    parser.add_argument('--spacenet_dataset', type=str, default='all', choices=['SN1', 'SN2', 'all'], help='SpaceNet dataset subset')

    # Band selection arguments
    parser.add_argument('--select_bands', type=int, default=None, help='number of continuous bands to randomly select (None = use all)')

    args = parser.parse_args()

    # Create output directories
    os.makedirs(args.save_checkpoint_path, exist_ok=True)
    os.makedirs(args.out_images, exist_ok=True)
    os.makedirs('./logs', exist_ok=True)

    # Build model
    seg = build_seg_vit_h(
        sam_checkpoint=args.sam_checkpoint,
        fixed_hsi_channels=args.fixed_hsi_channels
    )

    if args.parallel:
        seg = nn.DataParallel(seg, device_ids=[0, 1])
    seg = seg.to(args.device)

    # Load checkpoint if specified
    if args.load_state_dict:
        if args.checkpoint_path is None:
            raise ValueError("checkpoint_path must be specified when load_state_dict is True")
        model_state_dict = torch.load(args.checkpoint_path)
        seg.load_state_dict(model_state_dict, strict=False)
        del model_state_dict

    # Create dataset
    if args.dataset_type == 'spacenet':
        print(f"Using SpaceNet dataset: {args.spacenet_dataset}")
        dataset = SpaceNetDataset(
            ground_truth_dir=args.spacenet_gt_dir,
            hsi_info_dir=args.spacenet_hsi_dir,
            endmember_lib_path=args.spacenet_endmember_lib,
            dataset=args.spacenet_dataset,
        )
    else:
        dataset = CustomDataset(args.hsi_path, args.gt_path, index_file=args.index_file)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )

    # Freeze model if specified
    if args.frozen:
        for param in seg.parameters():
            param.requires_grad = False

    torch.cuda.empty_cache()
    optimizer = optim.Adam(seg.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    log_module_param_memory(seg, log_dir='./logs', filename='module_param_memory.txt')

    train(
        net=seg,
        lr=args.lr,
        epochs=args.epochs,
        dataloader=dataloader,
        optimizer=optimizer,
        wavelengths=args.wavelengths,
        point_num=args.point_num,
        build_input=build_input_for_hyperseg,
        transform_output=transform_output_seg,
        net_call=seg_call,
        save_checkpoint_path=args.save_checkpoint_path,
        save_name=args.save_name,
        device=args.device,
        show_rgb=args.show_rgb,
        start_validation=args.start_validation,
        dataset_type=args.dataset_type,
        num_iterations=args.num_iterations,
        out_images=args.out_images,
        select_bands=args.select_bands,
    )
