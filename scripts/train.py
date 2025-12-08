import sys
import os

from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import random
from hyperseg import build_seg_vit_h
import argparse
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import List
import PIL
import cv2
from hyperseg.modeling.dataset import CustomDataset
from hyperseg.utils.focal_loss import loss_masks
from hyperseg.autoMetric import SegMetric
import os
from hyperseg.utils.tools import create_point_coords, mask_iou, build_input_for_hyperseg, seg_call, transform_output_seg, pca_dr, command, update_points, Meaner
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim
import subprocess


def log_module_param_memory(model, log_dir='./logs', filename='module_param_memory.txt'):
    os.makedirs(log_dir, exist_ok=True)
    lines = []
    total_bytes = 0
    for name, module in model.named_modules():
        param_bytes = 0
        for p in module.parameters(recurse=False):
            param_bytes += p.numel() * p.element_size()
        buffer_bytes = 0
        for b in module.buffers(recurse=False):
            buffer_bytes += b.numel() * b.element_size()
        module_bytes = param_bytes + buffer_bytes
        if module_bytes == 0:
            continue
        total_bytes += module_bytes
        module_name = name if name != '' else '<root>'
        lines.append(f"{module_name}: {module_bytes / (1024 ** 2):.4f} MB (params {param_bytes / (1024 ** 2):.4f} MB, buffers {buffer_bytes / (1024 ** 2):.4f} MB)")
    lines.append(f"TOTAL: {total_bytes / (1024 ** 2):.4f} MB")
    with open(os.path.join(log_dir, filename), 'w') as f:
        f.write('\n'.join(lines))


def register_activation_memory_hooks(model):
    module_to_name = {}
    for name, module in model.named_modules():
        module_to_name[module] = name if name != '' else '<root>'

    activation_mem = {}

    def accumulate_tensor_mem(obj):
        mem = 0
        if isinstance(obj, torch.Tensor):
            mem += obj.numel() * obj.element_size()
        elif isinstance(obj, (list, tuple)):
            for x in obj:
                mem += accumulate_tensor_mem(x)
        elif isinstance(obj, dict):
            for v in obj.values():
                mem += accumulate_tensor_mem(v)
        return mem

    def hook(module, inputs, outputs):
        name = module_to_name.get(module, '<unnamed>')
        mem = accumulate_tensor_mem(outputs)
        prev = activation_mem.get(name, 0)
        if mem > prev:
            activation_mem[name] = mem

    hooks = []
    for module in model.modules():
        if len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(hook))
    return activation_mem, hooks


def train(net, lr, epochs, dataloader, optimizer, wavelengths, point_num, build_input, transform_output, net_call, save_checkpoint_path, save_name=None, device='cuda', feature_mapping=False, show_rgb=False, start_validation=False):
    net.train()
    total_batches = len(dataloader) * epochs # total batches
    lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_batches,  # 每个epoch一个完整周期
            eta_min=lr * 0.01     # 最小学习率
        )

    # mean loss notes
    mean_l = Meaner()
    mean_dice_l = Meaner()
    mean_masks_l = Meaner()
    first_batch_logged = False
    activation_mem, activation_hooks = register_activation_memory_hooks(net)

    for epoch in range(epochs):
        for batch_idx, ((images, gsd), gt) in enumerate(dataloader):
            # data preprocess
            images = images.to(device)
            gt = gt.to(device)
            B, H, W = gt.shape

            # generate point coords as prompt
            point_coords = create_point_coords(gt, num_points=1).to(device)
            point_labels = torch.ones((B, point_coords.shape[1])).to(device)
            
            # reset meaner
            mean_l.reset()
            mean_dice_l.reset()
            mean_masks_l.reset()

            optimizer.zero_grad()

            for iteration in range(args.num_iterations):
                # build input
                batched_input = build_input(images, shape=(H, W), point_coords=point_coords, point_labels=point_labels, device=device)
                # forward
                batched_output, all_features = net_call(net, batched_input, wavelengths=wavelengths, multimask_output=False, GSD=gsd)
                # get logits from output
                logits, bmasks = transform_output(batched_output)
                # compute loss
                loss_mask, loss_dice = loss_masks(logits.unsqueeze(1), gt.float().unsqueeze(1), num_masks=B)
                l = (loss_mask + loss_dice) / args.num_iterations

                mean_l.update(l.item())
                mean_dice_l.update(loss_dice.item())
                mean_masks_l.update(loss_mask.item())

                if feature_mapping:
                    feature_map = all_features[-1] # B, C, H', W'
                    for k in range(B):
                        fe = feature_map[k] # C, H', W'
                        fmap = pca_dr(fe.detach().cpu().numpy())
                        cv2.imwrite(os.path.join(args.out_images, f"feature_map_{epoch}_{batch_idx}_{k}.png"), (fmap * 255).astype(np.uint8))
                
                if show_rgb:
                    rgb = torch.flip(batched_output[0]["rgb"][0].permute(1, 2, 0), dims=[-1]).detach().cpu().numpy()
                    # cv2.imwrite(os.path.join(args.out_images, f"rgb_{epoch}_{batch_idx}_{i}.png"), (rgb * 255).astype(np.uint8))
                    mask = batched_output[0]["masks"][0, 0, :, :].detach().int().cpu().numpy()
                    gt_show = gt[0].detach().cpu().numpy()
                    # cv2.imwrite(os.path.join(args.out_images, f"mask_{epoch}_{batch_idx}_{i}.png"), (mask * 255).astype(np.uint8))
                del batched_input, batched_output
                l.backward()
                point_coords, point_labels = update_points(point_coords, point_labels, bmasks, gt)

            optimizer.step()
            lr_scheduler.step()

            if not first_batch_logged:
                try:
                    if torch.cuda.is_available():
                        mem_summary = torch.cuda.memory_summary(device=device, abbreviated=False)
                        with open(os.path.join('./logs', 'cuda_mem_summary_first_batch.txt'), 'w') as f:
                            f.write(mem_summary)
                    if activation_mem is not None:
                        lines = []
                        total_bytes = 0
                        for name, bytes_ in activation_mem.items():
                            total_bytes += bytes_
                            lines.append(f"{name}: {bytes_ / (1024 ** 2):.4f} MB")
                        lines.append(f"TOTAL: {total_bytes / (1024 ** 2):.4f} MB")
                        with open(os.path.join('./logs', 'module_activation_memory_first_batch.txt'), 'w') as f:
                            f.write('\n'.join(lines))
                except Exception as e:
                    print(f"Failed to write CUDA memory summary: {e}")
                finally:
                    for h in activation_hooks:
                        h.remove()
                first_batch_logged = True

            print(f"Loss Batch{batch_idx} focal loss or binary entropy loss:{mean_masks_l.avg} dice loss:{mean_dice_l.avg} loss:{mean_l.avg} Current lr:{optimizer.param_groups[0]['lr']}")
            
            # save model
            if batch_idx % 50 == 0:
                if batch_idx % 400 == 0:
                    if start_validation:
                        torch.save(net.state_dict(), os.path.join(save_checkpoint_path, f"temp_{epoch}_{batch_idx}.pth"))
                        com = command(os.path.join(save_checkpoint_path, f"temp_{epoch}_{batch_idx}.pth"))
                        subprocess.Popen(com, shell=True)
                if show_rgb:
                    cv2.imwrite(os.path.join(args.out_images, f"rgb_{epoch}_{batch_idx}_{0}.png"), (rgb * 255).astype(np.uint8))
                    cv2.imwrite(os.path.join(args.out_images, f"mask_{epoch}_{batch_idx}_{0}.png"), (mask * 255).astype(np.uint8))
                    cv2.imwrite(os.path.join(args.out_images, f"gt_{epoch}_{batch_idx}_{0}.png"), (gt_show * 255).astype(np.uint8))
                if batch_idx % 5000 == 0:
                    if save_name is not None:
                        torch.save(net.state_dict(), os.path.join(save_checkpoint_path, f"{save_name}_{epoch}_{batch_idx}.pth"))
                    else:
                        torch.save(net.state_dict(), os.path.join(save_checkpoint_path, f"net_{epoch}_{batch_idx}.pth"))
    # save checkpoint after each epoch
    if save_name is not None:
        torch.save(net.state_dict(), os.path.join(save_checkpoint_path, f"{save_name}_{epoch}.pth"))
    else:
        torch.save(net.state_dict(), os.path.join(save_checkpoint_path, f"net_{epoch}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--wavelengths', type=str, default=[400.0 + (2100.0 / 112.0) * i for i in range(1, 113)]
        , help='path wavelengths of the input image'
    )
    parser.add_argument(
        '--num_iterations', type=int, default=2,
        )
    parser.add_argument(
        '--sam_checkpoint', type=str, default='./checkpoints/sam_vit_h_4b8939.pth',
        help='path to the SAM checkpoint'
    )
    parser.add_argument(
        '--hyperfree_checkpoint', type=str, default='./checkpoints/HyperFree-h.pth',
        help='path to the HyperFree checkpoint'
    )
    parser.add_argument(
        '--device', type=str, default='cuda:7',
        help='device to use'
    )
    parser.add_argument(
        '--hsi_path', type=str, default="/data2/pl/HSITask/data/HyperFree/data_compressed",
        help='path to the hsi images'
    )
    parser.add_argument(
        '--gt_path', type=str, default="/data2/pl/HSITask/data/HyperFree/labels_hf_nms",
        help='path to the gt mask'
    )
    parser.add_argument(
        '--index_file', type=str, default='../data/hyperfree_index.json',
    )
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='batch size'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-4,
        help='learning rate'
    )
    parser.add_argument(
        '--epochs', type=int, default=5,
        help='number of epochs'
    )
    parser.add_argument(
        '--save_checkpoint_path', type=str, default='./checkpoints',
        help='path where checkpoint saves'
    )
    parser.add_argument(
        '--out_images', type=str, default='./outputs/out_images',
        help='path where images saves'
    )
    parser.add_argument(
        '--channel_proj_spectral', action='store_true', default=True,
        help='use spectral channel projection'
    )
    parser.add_argument(
        '--point_num', type=int, default=None,
        help='number of points to sample'
    )
    parser.add_argument(
        '--load_state_dict', action='store_true',
        help='load state dict'
    )
    parser.add_argument(
        '--frozen', action='store_true',help='whether to freeze the whole model'
    )
    parser.add_argument(
        '--checkpoint_path', type=str, default='./checkpoints/net_1_5000.pth',
        help='path to the checkpoint to load'
    )
    parser.add_argument(
        '--ignore_hsi_module', action='store_true',help='whether to ignore hsi module, for ablation study'
    )
    parser.add_argument(
        '--ignore_spectral_query', action='store_true',help='whether to ignore spectral query, for ablation study'
    )
    parser.add_argument(
        '--feature_as_query', action='store_true',help='whether to use feature as query'
    )
    parser.add_argument(
        '--save_name', type=str, default=None,
    )
    parser.add_argument(
        '--start_epoch', type=int, default=0,
    )
    
    parser.add_argument(
        '--save_feature_mapping', action='store_true',help='whether to save feature mapping for visualization'
    )
    parser.add_argument(
        '--parallel', action='store_true',help='whether to use DataParallel'
    )
    parser.add_argument(
        '--num_workers', type=int, default=1,
        help='number of workers for data loading'
    )
    parser.add_argument(
        '--only_hsi_module', action='store_true',help='whether to only use hsi module'
    )
    parser.add_argument(
        '--show_rgb', action='store_true',help='whether to show rgb image'
    )
    parser.add_argument(
        '--start_validation', action='store_true',help='whether to start validation after each epoch'
    )
    args = parser.parse_args()
    os.makedirs(args.save_checkpoint_path, exist_ok=True)
    os.makedirs(args.out_images, exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    seg = build_seg_vit_h(sam_checkpoint=args.sam_checkpoint,
                          hyperfree_checkpoint=args.hyperfree_checkpoint,
                          channel_proj_spectral=args.channel_proj_spectral,
                          ignore_hsi_module=args.ignore_hsi_module, 
                          ignore_spectral_query=args.ignore_spectral_query,
                          feature_as_query=args.feature_as_query,
                          only_hsi_module=args.only_hsi_module)
    
    if args.parallel:
        # device_ids = args.device.split(',')
        seg = nn.DataParallel(seg, device_ids=[0, 1])
        # seg = seg.cuda(device=int(device_ids[0]))
    seg = seg.to(args.device)

    # load state dict
    if args.load_state_dict:
        assert args.checkpoint_path is not None, f"When load_state_dict is True, checkpoint_path should not be None"
        model_state_dict = torch.load(args.checkpoint_path)
        seg.load_state_dict(model_state_dict, strict=False)
        del model_state_dict
    
    dataset = CustomDataset(args.hsi_path, args.gt_path, index_file=args.index_file)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    # different config, different 'save name'.This is for ablation study
    if args.ignore_hsi_module:
        save_name = 'without_hsi_module'
    elif args.ignore_spectral_query:
        save_name = 'without_spectral_query'
    elif args.only_hsi_module:
        save_name = 'only_hsi_module'
    else:
        save_name = args.save_name


    if args.frozen: # higher priority
        for name, param in seg.named_parameters():
            param.requires_grad = False

    torch.cuda.empty_cache()
    optimizer = optim.Adam(seg.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    
    log_module_param_memory(seg, log_dir='./logs', filename='module_param_memory.txt')

    train(seg, args.lr, args.epochs, dataloader, optimizer, wavelengths=args.wavelengths, point_num=args.point_num, build_input=build_input_for_hyperseg, transform_output=transform_output_seg, net_call=seg_call, save_checkpoint_path=args.save_checkpoint_path, device=args.device, save_name=save_name, feature_mapping=args.save_feature_mapping, show_rgb=args.show_rgb, start_validation=args.start_validation)


