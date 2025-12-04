import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from hyperseg import build_seg_vit_h, SamPredictor
import argparse
import random

from hyperseg.modeling.dataset import CustomDataset
from hyperseg.automatic_mask_generator import SamAutomaticMaskGenerator
import cv2
from skimage import io
from sklearn.metrics import average_precision_score
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
    parser.add_argument('--channel_proj_spectral', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model', type=str, default='seg')
    parser.add_argument('--point_num', type=int, default=1)
    parser.add_argument('--sam_checkpoint', type=str, default=None)
    parser.add_argument('--hyperfree_checkpoint', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--ignore_hsi_module', action='store_true')
    parser.add_argument('--ignore_spectral_query', action='store_true')
    parser.add_argument('--remove', action='store_true')
    parser.add_argument('--feature_as_query', action='store_true')
    parser.add_argument('--samples', type=int, default=0)

    args = parser.parse_args()

    # set random seed
    if args.seed is not None:
        random.seed(args.seed)

    if args.dataset == 'LongKou':
        wavelengths = [
    401.809998, 404.031006, 406.252014, 408.472992, 410.694000, 412.915009,
    415.135986, 417.356995, 419.578003, 421.799988, 424.020996, 426.242004,
    428.463013, 430.683990, 432.904999, 435.126007, 437.346985, 439.567993,
    441.789001, 444.010010, 446.230988, 448.451996, 450.674011, 452.894989,
    455.115997, 457.337006, 459.558014, 461.778992, 464.000000, 466.221008,
    468.441986, 470.662994, 472.884003, 475.105011, 477.326996, 479.548004,
    481.769012, 483.989990, 486.210999, 488.432007, 490.653015, 492.873993,
    495.095001, 497.316010, 499.536987, 501.757996, 503.979004, 506.200989,
    508.421997, 510.643005, 512.864014, 515.085022, 517.306030, 519.526978,
    521.747986, 523.968994, 526.190002, 528.411011, 530.632019, 532.854004,
    535.075012, 537.296021, 539.517029, 541.737976, 543.958984, 546.179993,
    548.401001, 550.622009, 552.843018, 555.064026, 557.284973, 559.505981,
    561.728027, 563.948975, 566.169983, 568.390991, 570.612000, 572.833008,
    575.054016, 577.275024, 579.495972, 581.716980, 583.937988, 586.158997,
    588.380981, 590.601990, 592.822998, 595.044006, 597.265015, 599.486023,
    601.706970, 603.927979, 606.148987, 608.369995, 610.591003, 612.812012,
    615.033020, 617.255005, 619.476013, 621.697021, 623.918030, 626.138977,
    628.359985, 630.580994, 632.802002, 635.023010, 637.244019, 639.465027,
    641.685974, 643.908020, 646.129028, 648.349976, 650.570984, 652.791992,
    655.013000, 657.234009, 659.455017, 661.676025, 663.896973, 666.117981,
    668.338989, 670.559998, 672.781982, 675.002991, 677.223999, 679.445007,
    681.666016, 683.887024, 686.107971, 688.328979, 690.549988, 692.770996,
    694.992004, 697.213013, 699.434998, 701.656006, 703.877014, 706.098022,
    708.318970, 710.539978, 712.760986, 714.981995, 717.203003, 719.424011,
    721.645020, 723.866028, 726.086975, 728.309021, 730.530029, 732.750977,
    734.971985, 737.192993, 739.414001, 741.635010, 743.856018, 746.077026,
    748.297974, 750.518982, 752.739990, 754.961975, 757.182983, 759.403992,
    761.625000, 763.846008, 766.067017, 768.288025, 770.508972, 772.729980,
    774.950989, 777.171997, 779.393005, 781.614014, 783.835999, 786.057007,
    788.278015, 790.499023, 792.719971, 794.940979, 797.161987, 799.382996,
    801.604004, 803.825012, 806.046021, 808.267029, 810.489014, 812.710022,
    814.931030, 817.151978, 819.372986, 821.593994, 823.815002, 826.036011,
    828.257019, 830.478027, 832.698975, 834.919983, 837.140991, 839.362976,
    841.583984, 843.804993, 846.026001, 848.247009, 850.468018, 852.689026,
    854.909973, 857.130981, 859.351990, 861.572998, 863.794006, 866.015991,
    868.237000, 870.458008, 872.679016, 874.900024, 877.120972, 879.341980,
    881.562988, 883.783997, 886.005005, 888.226013, 890.447021, 892.668030,
    894.890015, 897.111023, 899.331970, 901.552979, 903.773987, 905.994995,
    908.216003, 910.437012, 912.658020, 914.879028, 917.099976, 919.320984,
    921.543030, 923.763977, 925.984985, 928.205994, 930.427002, 932.648010,
    934.869019, 937.090027, 939.310974, 941.531982, 943.752991, 945.973999,
    948.195007, 950.416992, 952.638000, 954.859009, 957.080017, 959.301025,
    961.521973, 963.742981, 965.963989, 968.184998, 970.406006, 972.627014,
    974.848022, 977.070007, 979.291016, 981.512024, 983.732971, 985.953979,
    988.174988, 990.395996, 992.617004, 994.838013, 997.059021, 999.280029]
        GSD = 0.463 # Ground sampling distance (m/pixel)
        data_path = "./Data/LongKou/WHU-Hi-LongKou.tif"
        gt_path = "./Data/LongKou/WHU-Hi-LongKou_gt.tif"
        sample_path = "./Data/LongKou/Test"
        sample_list = os.listdir(sample_path)
    
    elif args.dataset == 'HanChuan':
        wavelengths = [400.0 + ((600.0 * i)/ 274.0) for i in range(1, 275)]
        GSD = 0.109
        data_path = "/data2/pl/ImageTask/wxq/HyperFree_copy/Data/Tiff_format/WHU-Hi-HanChuan/WHU-Hi-HanChuan.tif"
        gt_path = "/data2/pl/ImageTask/wxq/HyperFree_copy/Data/Tiff_format/WHU-Hi-HanChuan/WHU-Hi-HanChuan_gt.tif"
        

pred_iou_thresh = 0.0  # Controling the model's predicted mask quality in range [0, 1].
stability_score_thresh = 0.0  # Controling the stability of the mask in range [0, 1].
feature_index_id = 1  # Deciding which stage of encoder features to use

# ckpt_pth = "./checkpoints/HyperFree-h.pth"
ROOT = os.path.dirname(os.path.dirname(__file__))
save_dir = os.path.join(ROOT, "outputs", "hyperspectral_classification")
os.makedirs(save_dir, exist_ok=True)

gt = io.imread(gt_path)
img = io.imread(data_path)

height, width = img.shape[-2], img.shape[-1]

ratio = 1024 / (max(height, width))
GSD = GSD / ratio
GSD = torch.tensor([[GSD]])

# Normalization
img_normalized = (img - img.min()) / (img.max() - img.min())
img_torch = torch.from_numpy(img_normalized).float()
img_uint8 = (255 * img_normalized).astype(np.uint8)
#################################################################
# gt_torch = torch.from_numpy(gt).long().to(args.device)
# gt_list = []
# for i in range(gt_torch.max() + 1):
#     gt_torch_i = gt_torch.clone()
#     gt_torch_i[gt_torch != i] = 0
#     gt_torch_i[gt_torch == i] = 1
#     gt_list.append(gt_torch_i)

# for i, gt_i in enumerate(gt_list):
#     print(f"i{i}")
#     idx_y, idx_x = torch.where(gt_i == 1)
#     length = len(idx_x)
#     for k in range(30):
#         print(f"k{k}")
#         if length - k > 0:
#             random_idx = random.randint(0, length - 1)
#             plt.plot(np.arange(len(wavelengths)), img_normalized[:, idx_y[random_idx], idx_x[random_idx]], 'b-')
#     plt.savefig(os.path.join(save_dir, "spectral_plot_" + str(i) + ".png"))
#     plt.close()
#################################################################
# model
if args.model == 'seg':
    seg = build_seg_vit_h(channel_proj_spectral=args.channel_proj_spectral, ignore_hsi_module=args.ignore_hsi_module, ignore_spectral_query=args.ignore_spectral_query, feature_as_query=args.feature_as_query, hyperfree_checkpoint=args.hyperfree_checkpoint, sam_checkpoint=args.sam_checkpoint).to(args.device)
    if args.checkpoint is not None:
        info = seg.load_state_dict(torch.load(args.checkpoint, map_location=args.device), strict=False)
    seg.eval()
    mask_generator = SamAutomaticMaskGenerator(
        seg,
        points_per_side=32,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
    )
elif seg.model == 'hyperfree':
    pass

# few_shots = []  # Storing binary maps for each class, where the non-zero location represents the corresponding sample
# num_classes = 9
# for i in range(num_classes):
#     few_shots.append(np.zeros((height, width)))

# few_shots[0][388, 42] = 1
# few_shots[1][231, 42] = 1
# few_shots[2][159, 18] = 1
# few_shots[3][40, 329] = 1
# few_shots[4][192, 285] = 1
# few_shots[5][360, 280] = 1
# few_shots[6][265, 174] = 1
# few_shots[7][355, 108] = 1
# few_shots[8][34, 238] = 1
# max_oa = -1e9
# max_aa = -1e9
# max_ka = -1e9
# while True:
#     few_shots = []
#     for i in range(num_classes):
#         few_shots.append(np.zeros((height, width)))
#     for j in range(num_classes):
#         idx = np.where(gt == j + 1)
#         k = random.randint(0, len(idx) - 1)
#         few_shots[j][idx[k]] = 1

#     classification_maps_each_class, classification_map = hyperspectral_classification(
#         mask_generator=mask_generator,
#         image=img_uint8.transpose((1, 2, 0)), # (H, W, C)
#         few_shots=few_shots,
#         spectral_lengths=wavelengths,
#         feature_index_id=feature_index_id,
#         GSD=torch.tensor([1.0]),
#     )

#     show_anns(classification_maps_each_class, save_dir)

#     segmetric = SegMetric(
#     gt = torch.tensor(gt),
#     pred_mask = torch.tensor(classification_map),
#     num_classes = num_classes
#     )

#     oa, aa, ka = segmetric.compute_metrics()
#     print(f"Overall Accuracy: {oa}")
#     print(f"Average Accuracy: {aa}")
#     print(f"Kappa Coefficient: {ka}")
#     if oa > max_oa:
#         max_oa = oa
#         print("New Max Overall Accuracy:", oa)
#     if aa > max_aa:
#         max_aa = aa
#         print("New Max Average Accuracy:", aa)
#     if ka > max_ka:
#         max_ka = ka
#         print("New Max Kappa Coefficient:", ka)
def similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)

    norm_v1 = np.linalg.norm(vector1)
    norm_v2 = np.linalg.norm(vector2)
    # 计算余弦相似度
    consine_similarity = dot_product / (norm_v1 * norm_v2)

    # 计算 1 范数
    l1_norm = 1 / np.sum(np.abs(vector1 - vector2))
    print(f"cosine similarity: {consine_similarity}, l1_norm: {l1_norm}")
    return consine_similarity


best_score = -1e9
best_mask = Image.open(os.path.join(save_dir, "labelled.png"))
image_array = np.array(best_mask)

image_tensor = torch.from_numpy(image_array).float().to(args.device)

while True:
    ultimate_gt = torch.zeros((height, width))
    anns = torch.zeros((height, width))
    predictor = SamPredictor(seg, wavelengths)
    gt_torch = torch.from_numpy(gt).long().to(args.device)
    gt_list = []
    for i in range(gt_torch.max() + 1):
        gt_torch_i = gt_torch.clone()
        gt_torch_i[gt_torch != i] = 0
        gt_torch_i[gt_torch == i] = 1
        gt_list.append(gt_torch_i)

    #############################
    # few_shots = []
    # few_points = []
    # few_vectors = []
    # num_classes = 9
    # for i in range(num_classes):
    #     few_shots.append(np.zeros((height, width)))
    # for j in range(num_classes):
    #     idx = np.where(gt == j + 1)
    #     k = random.randint(0, len(idx[0]) - 1)
    #     # few_shots[j][idx[k]] = 1
    #     few_points.append(torch.tensor([[idx[1][k], idx[0][k]]]))
    ####
    # few_points = []
    # few_vectors = []
    # few_points.append(torch.tensor([[363, 111]]))
    # few_points.append(torch.tensor([[ 15, 249]]))
    # few_points.append(torch.tensor([[ 38, 154]]))
    # few_points.append(torch.tensor([[342,  31]]))
    # few_points.append(torch.tensor([[259, 193]]))
    # few_points.append(torch.tensor([[274, 394]]))
    # few_points.append(torch.tensor([[221, 140]]))
    # few_points.append(torch.tensor([[ 90, 215]]))
    # few_points.append(torch.tensor([[236, 122]]))

    # ###
    # for i, point in enumerate(few_points):
    #     masks, _, _, all_features = predictor.predict_torch(
    #             images = img_torch.to(args.device).unsqueeze(0).permute(0, 2, 3, 1),
    #             point_coords = point.to(args.device).unsqueeze(0),
    #             point_labels = torch.ones((1, point.shape[0])).to(args.device),
    #             multimask_output=False,
    #             wavelengths = wavelengths,
    #             return_logits=False,
    #             GSD = torch.tensor([1.0])
    #         )
    #     few_masks = (masks[0] > 0).int() * img_torch.to(args.device)
    #     few_vectors.append(few_masks.mean(dim=[1, 2]))
    #############################
    # ap_list = []
    # iou_list = []

    # ultimate_gt = torch.zeros((height, width))
    # x = torch.arange(0, width, 5).unsqueeze(0).repeat((height // 5, 1))
    # y = torch.arange(0, height, 5).unsqueeze(1).repeat((1, width // 5))

    # point_coords = torch.stack([x, y], dim=-1).reshape(-1, 2)
    # best_cls_point_score = [-1e9, -1e9, -1e9, -1e9, -1e9, -1e9, -1e9, -1e9, -1e9]
    # for point_sq in range(point_coords.shape[0]):
    #     print(point_sq)
    #     masks, _, _, all_features = predictor.predict_torch(
    #                 images = img_torch.to(args.device).unsqueeze(0).permute(0, 2, 3, 1),
    #                 point_coords = point_coords[point_sq:(point_sq+1), :].to(args.device).unsqueeze(0),
    #                 point_labels = torch.ones((1, 1)).to(args.device),
    #                 multimask_output=False,
    #                 wavelengths = wavelengths,
    #                 return_logits=False,
    #                 GSD = torch.tensor([1.0])
    #             )
    #     max_score = -1e9
    #     mask = (masks[0] > 0)
    #     numer = -1
    #     for cls_num in range(1, gt_torch.max() + 1):
    #         score = 0
    #         mask_gt =  mask.int() * gt_list[cls_num]
    #         score = mask_gt.sum()
    #         if score > max_score:
    #             max_score = score
    #             numer = cls_num - 1
    #             best_mask = (masks[0] > 0)

    #     if max_score > best_cls_point_score[numer]:
    #         best_cls_point_score[numer] = max_score
    #         ultimate_gt[best_mask[0, :, :]] = numer
        
    #     del masks
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
                # point_coords = create_point_coords(gt_in.unsqueeze(0), num_points=20)
            
            best_point_score = -1e9
            for point_sq in range(point_coords.shape[1]):
                # point_coords = predictor.transform.apply_coords_torch(point_coords.to(args.device), img.shape[:2])
                masks, _, _, all_features = predictor.predict_torch(
                    images = img_torch.to(args.device).unsqueeze(0).permute(0, 2, 3, 1),
                    point_coords = point_coords[:, point_sq:(point_sq+1), :].to(args.device),
                    point_labels = torch.ones((1, 1)).to(args.device),
                    multimask_output=False,
                    wavelengths = wavelengths,
                    return_logits=False,
                    GSD = torch.tensor(GSD) 
                )
                masks = masks[0]
                mask = (masks > 0)
                # mask_gt
                max_score = (mask.int() * gt_list[i]).sum() / (gt_list[i].sum() + mask.int().sum())
                
                if max_score > best_point_score:
                    best_point_score = max_score
                    print(best_point_score)
                    best_mask = mask

                cv2.imwrite(os.path.join(save_dir, f"mask_{i}_{l}.png"), (mask[0, :, :].detach().cpu().numpy() * 255).astype(np.uint8))
            ultimate_gt[best_mask[0, :, :]] = i

    seg_metric = SegMetric(
        gt = gt_torch.detach().cpu(),
        pred_mask = ultimate_gt, 
        num_classes = gt_torch.max()
    )
    oa, aa, ka = seg_metric.compute_metrics()
    print(f"Overall Accuracy: {oa}")
    print(f"Average Accuracy: {aa}")
    print(f"Kappa Coefficient: {ka}")
    print(f"AP")
    if oa > best_score:
        best_score = oa
        img_labelled = torch.zeros((height, width, 3))
        for i in range(1, ultimate_gt.int().max().item() + 1):
            img_labelled[ultimate_gt == i] = torch.rand(1, 1, 3)
        cv2.imwrite(os.path.join(save_dir, f"labelled.png"), (img_labelled.detach().cpu().numpy() * 255).astype(np.uint8))
    print(f"best: {best_score}")
    if args.remove:
            os.remove(args.checkpoint)

            