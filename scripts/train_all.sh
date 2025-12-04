#!/bin/bash
set -e
DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT="${DIR}/.."

mkdir -p "${ROOT}/checkpoints" "${ROOT}/outputs/out_images" "${ROOT}/outputs/hyperspectral_classification" "${ROOT}/logs"

CUDA_VISIBLE_DEVICES=3 nohup python "${ROOT}/scripts/train.py" \
  --batch_size 3 --lr 0.0008 --epochs 20 \
  --sam_checkpoint /data2/pl/checkpoints/sam-hq/pretrained_checkpoint/sam_vit_h_4b8939.pth \
  --channel_proj_spectral \
  --hyperfree_checkpoint "${ROOT}/checkpoints/HyperFree-h.pth" \
  --gt_path /data2/pl/HSITask/data/HyperFree/my_gt_version_2 \
  --save_checkpoint_path "${ROOT}/checkpoints" --out_images "${ROOT}/outputs/out_images" \
  > "${ROOT}/logs/train_default.log" 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python "${ROOT}/scripts/train.py" \
  --batch_size 3 --lr 0.0008 --epochs 20 \
  --sam_checkpoint /data2/pl/checkpoints/sam-hq/pretrained_checkpoint/sam_vit_h_4b8939.pth \
  --channel_proj_spectral --ignore_spectral_query \
  --hyperfree_checkpoint "${ROOT}/checkpoints/HyperFree-h.pth" \
  --gt_path /data2/pl/HSITask/data/HyperFree/my_gt_version_2 \
  --save_checkpoint_path "${ROOT}/checkpoints" --out_images "${ROOT}/outputs/out_images" \
  > "${ROOT}/logs/train_no_spectral_query.log" 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python "${ROOT}/scripts/train.py" \
  --batch_size 3 --lr 0.0008 --epochs 20 \
  --sam_checkpoint /data2/pl/checkpoints/sam-hq/pretrained_checkpoint/sam_vit_h_4b8939.pth \
  --channel_proj_spectral --ignore_hsi_module \
  --hyperfree_checkpoint "${ROOT}/checkpoints/HyperFree-h.pth" \
  --gt_path /data2/pl/HSITask/data/HyperFree/my_gt_version_2 \
  --save_checkpoint_path "${ROOT}/checkpoints" --out_images "${ROOT}/outputs/out_images" \
  > "${ROOT}/logs/train_no_hsi_module.log" 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python "${ROOT}/scripts/train.py" \
  --batch_size 3 --lr 0.0008 --epochs 20 \
  --sam_checkpoint /data2/pl/checkpoints/sam-hq/pretrained_checkpoint/sam_vit_h_4b8939.pth \
  --channel_proj_spectral --ignore_hsi_module --ignore_spectral_query \
  --hyperfree_checkpoint "${ROOT}/checkpoints/HyperFree-h.pth" \
  --gt_path /data2/pl/HSITask/data/HyperFree/my_gt_version_2 \
  --save_checkpoint_path "${ROOT}/checkpoints" --out_images "${ROOT}/outputs/out_images" \
  > "${ROOT}/logs/train_ablation_both_ignored.log" 2>&1 &