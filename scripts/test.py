import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from hyperseg import build_seg_vit_h, SamPredictor
from hyperfree import HyperFree_model_registry, SamAutomaticMaskGenerator as HyperFreeMaskGenerator
import argparse
import random

from hyperseg.modeling.dataset import CustomDataset
from hyperseg.automatic_mask_generator import SamAutomaticMaskGenerator
import cv2
from skimage import io
from sklearn.metrics import average_precision_score
import scipy.io as sio
import h5py
from hyperseg.utils.tools import extract_connected_components
from hyperseg.autoMetric import SegMetric
from hyperseg.prompt_mask_feature_interaction import hyperspectral_classification, show_anns

from hyperseg.utils.tools import pca_dr
import matplotlib.pyplot as plt
from PIL import Image

from hyperseg.utils.tools import rsshow


def compute_ap(gt: torch.Tensor, preds: torch.Tensor):
    if(preds.shape[2]!=gt.shape[2] or preds.shape[3]!=gt.shape[3]):
        postprocess_preds = F.interpolate(preds, size=gt.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    return average_precision_score(gt[0, 0, :, :].flatten().cpu().detach().numpy(), postprocess_preds[0, 0, :, :].flatten().cpu().detach().numpy(), average='macro')


def create_point_coords(gt: torch.Tensor, num_points: int = 5):
    B, H, W = gt.shape
    point_coords = torch.zeros(B, num_points, 2)
    for i in range(B):
        idx = torch.nonzero(gt[i])
        n = min(num_points, idx.shape[0])
        rand_perm = torch.randperm(idx.shape[0])[:n]
        idx = idx[rand_perm]
        idx = torch.flip(idx, dims=[1])
        point_coords[i] = idx

    return point_coords


def mask_iou(pred_label, label):
    '''
    calculate mask iou for pred_label and gt_label
    '''
    assert len(pred_label.shape) == len(label.shape) == 3 # 1, H, W
    pred_label = (pred_label > 0)[0].int()
    label = label[0].int()

    intersection = ((label * pred_label)).sum()
    union = ((label + pred_label) > 0).sum()
    return intersection / union


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='LongKou')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model', type=str, default='seg', choices=['seg', 'hyperfree'])
    parser.add_argument('--hyperfree_type', type=str, default='vit_b', choices=['vit_b', 'vit_l', 'vit_h'])
    parser.add_argument('--point_num', type=int, default=1)
    parser.add_argument('--sam_checkpoint', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--remove', action='store_true')
    parser.add_argument('--samples', type=int, default=0)
    parser.add_argument('--fixed_hsi_channels', type=int, default=224)
    parser.add_argument('--feature_index_id', type=int, default=1, help='Which stage of encoder features to use for HyperFree')

    args = parser.parse_args()

    # set random seed
    if args.seed is not None:
        random.seed(args.seed)

    if args.dataset == 'LongKou':
        # LongKou: 270 bands, 550x400, wavelength 400-1000nm
        wavelengths = [400.0 + (600.0 / 270) * i for i in range(270)]
        GSD = 0.463 # Ground sampling distance (m/pixel)
        data_path = "./data/LongKou/WHU_Hi_LongKou.mat"
        gt_path = "./data/LongKou/WHU_Hi_LongKou_gt.mat"

    elif args.dataset == 'HanChuan':
        # HanChuan: 274 bands, 1217x303, wavelength 400-1000nm
        wavelengths = [400.0 + (600.0 / 274) * i for i in range(274)]
        GSD = 0.109
        data_path = "./data/HanChuan/WHU_Hi_HanChuan.mat"
        gt_path = "./data/HanChuan/WHU_Hi_HanChuan_gt.mat"

    elif args.dataset == 'HongHu':
        # HongHu: 270 bands, 940x475, wavelength 400-1000nm
        wavelengths = [400.0 + (600.0 / 270) * i for i in range(270)]
        GSD = 0.043
        data_path = "./data/HongHu/WHU_Hi_HongHu.mat"
        gt_path = "./data/HongHu/WHU_Hi_HongHu_gt.mat"

    elif args.dataset == "Houston13":
        # Houston 2013: 144 bands, 349x1905, wavelength 380-1050nm (ITRES CASI-1500)
        wavelengths = [380.0 + (1050.0 - 380.0) / 144 * i for i in range(144)]
        GSD = 2.5  # approximate GSD for airborne HSI
        data_path = "./data/Houston/Houston13.mat"
        gt_path = "./data/Houston/Houston13_7gt.mat"

    elif args.dataset == "Houston18":
        # Houston 2018: 48 bands, 601x2384, wavelength 380-1050nm
        wavelengths = [380.0 + (1050.0 - 380.0) / 48 * i for i in range(48)]
        GSD = 1.0  # approximate GSD
        data_path = "./data/Houston/Houston18.mat"
        gt_path = "./data/Houston/Houston18_7gt.mat"

    elif args.dataset == "IndianPines":
        # Indian Pines: 200 bands, 145x145, wavelength 400-2500nm (AVIRIS)
        wavelengths = [400.0 + (2500.0 - 400.0) / 200 * i for i in range(200)]
        GSD = 20.0  # 20m GSD for AVIRIS
        data_path = "./data/IndianPines/Indian_pines_corrected.mat"
        gt_path = "./data/IndianPines/Indian_pines_gt.mat"

    elif args.dataset == "PaviaU":
        # Pavia University: 103 bands, 610x340, wavelength 430-860nm (ROSIS)
        wavelengths = [430.0 + (860.0 - 430.0) / 103 * i for i in range(103)]
        GSD = 1.3  # 1.3m GSD for ROSIS
        data_path = "./data/PaviaU/PaviaU.mat"
        gt_path = "./data/PaviaU/PaviaU_gt.mat"

    elif args.dataset == "PaviaC":
        # Pavia Centre: 102 bands, 1096x715, wavelength 430-860nm (ROSIS)
        wavelengths = [430.0 + (860.0 - 430.0) / 102 * i for i in range(102)]
        GSD = 1.3  # 1.3m GSD for ROSIS
        data_path = "./data/PaviaC/Pavia.mat"
        gt_path = "./data/PaviaC/Pavia_gt.mat"

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    pred_iou_thresh = 0.3  # Controling the model's predicted mask quality in range [0, 1].
    stability_score_thresh = 0.4  # Controling the stability of the mask in range [0, 1].
    feature_index_id = 1  # Deciding which stage of encoder features to use

    ROOT = os.path.dirname(os.path.dirname(__file__))
    save_dir = os.path.join(ROOT, "outputs", "hyperspectral_classification")
    os.makedirs(save_dir, exist_ok=True)

    # Load data based on dataset type
    if args.dataset in ["Houston13", "Houston18"]:
        # HDF5 format (.mat v7.3) - MATLAB column-major order needs transpose
        with h5py.File(data_path, 'r') as f:
            img = f['ori_data'][:].astype(np.float64)  # raw: (C, W, H)
            img = np.transpose(img, (0, 2, 1))  # -> (C, H, W)
        with h5py.File(gt_path, 'r') as f:
            gt = f['map'][:].astype(np.uint8).T  # raw: (W, H) -> (H, W)
    elif args.dataset == "IndianPines":
        # scipy.io format, (H, W, C) -> transpose to (C, H, W)
        img = sio.loadmat(data_path)['indian_pines_corrected'].astype(np.float64)
        img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        gt = sio.loadmat(gt_path)['indian_pines_gt'].astype(np.uint8)
    elif args.dataset == "PaviaU":
        # scipy.io format, (H, W, C) -> transpose to (C, H, W)
        img = sio.loadmat(data_path)['paviaU'].astype(np.float64)
        img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        gt = sio.loadmat(gt_path)['paviaU_gt'].astype(np.uint8)
    elif args.dataset == "PaviaC":
        # scipy.io format, (H, W, C) -> transpose to (C, H, W)
        img = sio.loadmat(data_path)['pavia'].astype(np.float64)
        img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        gt = sio.loadmat(gt_path)['pavia_gt'].astype(np.uint8)
    elif args.dataset in ["LongKou", "HanChuan", "HongHu"]:
        # WHU datasets: scipy.io format, (H, W, C) -> transpose to (C, H, W)
        mat_keys = {
            "LongKou": ("WHU_Hi_LongKou", "WHU_Hi_LongKou_gt"),
            "HanChuan": ("WHU_Hi_HanChuan", "WHU_Hi_HanChuan_gt"),
            "HongHu": ("WHU_Hi_HongHu", "WHU_Hi_HongHu_gt"),
        }
        data_key, gt_key = mat_keys[args.dataset]
        img = sio.loadmat(data_path)[data_key].astype(np.float64)
        img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        gt = sio.loadmat(gt_path)[gt_key].astype(np.uint8)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    height, width = img.shape[-2], img.shape[-1]

    ratio = 1024 / (max(height, width))
    GSD = GSD / ratio
    GSD = torch.tensor([[GSD]])

    # Normalization
    img_normalized = (img - img.min()) / (img.max() - img.min())
    img_torch = torch.from_numpy(img_normalized).float()
    img_uint8 = (255 * img_normalized).astype(np.uint8)

    # Interpolate HSI channels to fixed_hsi_channels
    C_in, H_img, W_img = img_torch.shape
    if C_in != args.fixed_hsi_channels:
        # Reshape to (H*W, 1, C_in) for 1D interpolation along channel dimension
        img_torch = img_torch.permute(1, 2, 0).reshape(-1, 1, C_in)  # (H*W, 1, C_in)
        img_torch = F.interpolate(img_torch, size=args.fixed_hsi_channels, mode='linear', align_corners=False)
        img_torch = img_torch.reshape(H_img, W_img, args.fixed_hsi_channels).permute(2, 0, 1)  # (C_out, H, W)

    # model
    if args.model == 'seg':
        seg = build_seg_vit_h(
            sam_checkpoint=args.sam_checkpoint,
            fixed_hsi_channels=args.fixed_hsi_channels
        ).to(args.device)
        if args.checkpoint is not None:
            info = seg.load_state_dict(torch.load(args.checkpoint, map_location=args.device), strict=False)
        seg.eval()
        mask_generator = SamAutomaticMaskGenerator(
            seg,
            points_per_side=32,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
        )
    elif args.model == 'hyperfree':
        hyperfree = HyperFree_model_registry[args.hyperfree_type](
            checkpoint=args.checkpoint,
            encoder_global_attn_indexes=-1,
            merge_indexs=None
        ).to(args.device)
        hyperfree.eval()
        mask_generator = HyperFreeMaskGenerator(
            model=hyperfree,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            points_per_side=32,
        )


    def similarity(vector1, vector2):
        dot_product = np.dot(vector1, vector2)

        norm_v1 = np.linalg.norm(vector1)
        norm_v2 = np.linalg.norm(vector2)
        # cosine similarity
        consine_similarity = dot_product / (norm_v1 * norm_v2)

        # L1 norm
        l1_norm = 1 / np.sum(np.abs(vector1 - vector2))
        print(f"cosine similarity: {consine_similarity}, l1_norm: {l1_norm}")
        return consine_similarity


    best_score = -1e9
    gt_torch = torch.from_numpy(gt).long().to(args.device)
    num_classes = int(gt_torch.max().item())

    if args.model == 'hyperfree':
        # HyperFree workflow: generate masks and assign to classes using feature similarity
        # Prepare few_shots from ground truth (sample one point per class)
        few_shots = []
        for i in range(num_classes):
            few_shot = np.zeros((height, width))
            class_mask = (gt == (i + 1))  # gt class labels are 1-indexed
            if class_mask.sum() > 0:
                # Sample one point from this class
                locs = np.where(class_mask)
                idx = np.random.randint(len(locs[0]))
                few_shot[locs[0][idx], locs[1][idx]] = 1
            few_shots.append(few_shot)

        # Prepare image for HyperFree (H, W, C) in range [0, 255]
        img_hwc = img_uint8.transpose(1, 2, 0) if img_uint8.ndim == 3 and img_uint8.shape[0] < img_uint8.shape[-1] else img_uint8
        if img_hwc.ndim == 3 and img_hwc.shape[2] > img_hwc.shape[0]:
            img_hwc = img_uint8.transpose(1, 2, 0)

        # Run hyperspectral classification
        classification_maps_each_class, classification_map = hyperspectral_classification(
            mask_generator=mask_generator,
            image=img_hwc,
            few_shots=few_shots,
            spectral_lengths=wavelengths,
            feature_index_id=args.feature_index_id,
            GSD=GSD[0]
        )

        # Convert classification_map to prediction (class indices are 0-indexed in classification_map)
        # Need to shift by 1 to match gt format (1-indexed)
        ultimate_gt = torch.from_numpy(classification_map.astype(np.int64) + 1).float()
        # Set background (where classification_map == 0 originally) back to 0
        ultimate_gt[classification_map == 0] = 0

        # Save classification results
        show_anns(classification_maps_each_class, save_dir)

    else:
        # HyperSeg workflow: point-prompted segmentation
        for _ in range(1):
            anns = torch.zeros((height, width))
            predictor = SamPredictor(seg, wavelengths)
            gt_list = []
            for i in range(gt_torch.max() + 1):
                gt_torch_i = gt_torch.clone()
                gt_torch_i[gt_torch != i] = 0
                gt_torch_i[gt_torch == i] = 1
                gt_list.append(gt_torch_i)

            ultimate_gt = torch.zeros((height, width))
            for i in range(len(gt_list)):
                print(f"Class {i}")
                if i == 0:
                    continue # skip background
                mean_ap = 0
                mean_iou = 0
                gt_list_i = gt_list[i]
                components, labeled_array, num_components = extract_connected_components(gt_list_i.detach().cpu().numpy())
                for l in range(num_components):
                    gt_in = torch.tensor(components[l]).float().to(args.device)
                    if gt_in.sum() < 100:
                        continue

                    point_coords = create_point_coords(gt_in.unsqueeze(0), num_points=args.point_num)

                    best_point_score = -1e9
                    for point_sq in range(point_coords.shape[1]):
                        masks, _, _, _ = predictor.predict_torch(
                            images = img_torch.to(args.device).unsqueeze(0).permute(0, 2, 3, 1),
                            point_coords = point_coords[:, point_sq:(point_sq+1), :].to(args.device),
                            point_labels = torch.ones((1, 1)).to(args.device),
                            multimask_output=True,
                            wavelengths = wavelengths,
                            return_logits=False,
                        )
                        # masks: [1, 3, H, W] with multimask_output=True
                        # Select best mask out of 3 based on score against ground truth
                        best_mask_score = -1e9
                        best_single_mask = None
                        for mask_idx in range(masks.shape[1]):
                            mask_candidate = masks[0, mask_idx:mask_idx+1, :, :]  # [1, H, W]
                            mask_bool = (mask_candidate > 0)
                            score = (mask_bool.int() * gt_list[i]).sum() / (gt_list[i].sum() + mask_bool.int().sum())
                            if score > best_mask_score:
                                best_mask_score = score
                                best_single_mask = mask_bool

                        mask = best_single_mask
                        # mask_gt
                        max_score = (mask.int() * gt_list[i]).sum() / (gt_list[i].sum() + mask.int().sum())

                        if max_score > best_point_score:
                            best_point_score = max_score
                            print(best_point_score)
                            best_mask = mask

                        cv2.imwrite(os.path.join(save_dir, f"mask_{i}_{l}.png"), (mask[0, :, :].detach().cpu().numpy() * 255).astype(np.uint8))
                    ultimate_gt[best_mask[0, :, :]] = i

    # Compute metrics
    seg_metric = SegMetric(
        gt = gt_torch.detach().cpu(),
        pred_mask = ultimate_gt,
        num_classes = num_classes
    )
    oa, aa, ka = seg_metric.compute_metrics()
    print(f"Overall Accuracy: {oa}")
    print(f"Average Accuracy: {aa}")
    print(f"Kappa Coefficient: {ka}")

    if oa > best_score:
        best_score = oa
        img_labelled = torch.zeros((height, width, 3))
        for i in range(1, ultimate_gt.int().max().item() + 1):
            img_labelled[ultimate_gt == i] = torch.rand(1, 1, 3)
        cv2.imwrite(os.path.join(save_dir, f"labelled.png"), (img_labelled.detach().cpu().numpy() * 255).astype(np.uint8))
    print(f"best: {best_score}")

    if args.remove and args.checkpoint is not None:
        os.remove(args.checkpoint)
