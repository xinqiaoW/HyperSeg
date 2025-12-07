import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import json
import os
import tifffile
from scipy.interpolate import interp1d
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2
import clip
import torchvision
from PIL import Image
import scipy.io
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import find_boundaries
from typing import Tuple, Optional
from hyperseg.modeling.dataset import remove_trailing_digits
import argparse


def get_buffer_region(mask, k=5):
    """
    计算掩码边界向外扩展 k 个像素的区域 (Buffer Region)。

    Args:
        mask (torch.Tensor): 原始二值掩码，形状 (H, W) 或 (1, H, W)。
        k (int): 向外扩展的像素数。

    Returns:
        torch.Tensor: 缓冲区二值掩码 R_Buffer，形状 (H, W)。
    """
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)  # 转换为 (1, 1, H, W)
    elif mask.dim() == 3 and mask.shape[0] == 1:
        mask = mask.unsqueeze(0)  # 转换为 (1, 1, H, W)
    else:
        # 处理可能的批次或通道维度，确保是 (N, C, H, W) 形状
        # 假设输入已经是 (H, W) 或 (1, H, W)
        raise ValueError("Mask must be 2D (H, W) or 3D (1, H, W)")

    # 1. 定义结构元素 (Kernel)
    # 尺寸为 (2k + 1) x (2k + 1) 的全 1 方形内核
    kernel_size = 2 * k + 1
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)
    
    # 2. 执行膨胀 (Dilation)
    # PyTorch 没有内置的二值形态学操作，我们使用 Max Pooling 或 Conv2d 来模拟膨胀。
    # 膨胀可以被看作是使用全 1 内核进行 Max Pooling，但需要 Padding 确保尺寸不变。
    
    # 2a. 使用 Max Pool 模拟膨胀
    # 为了使用 Max Pool，需要将 mask 转换为浮点型
    mask_float = mask.float()
    
    # 计算填充 (Padding)
    padding = k 
    
    # Max Pool 模拟膨胀：窗口内最大值 (即是否有任何前景像素)
    mask_dilated = F.max_pool2d(
        mask_float, 
        kernel_size=kernel_size, 
        stride=1, 
        padding=padding
    )
    
    # 3. 计算差集：膨胀后的掩码 - 原始掩码
    # 将膨胀后的结果重新二值化 (> 0)
    mask_dilated_binary = (mask_dilated > 0.5).int()
    
    # 缓冲区 R_Buffer 是膨胀区域减去原始区域
    buffer_region = mask_dilated_binary - mask.int()
    
    # 确保结果是二值的 (如果原始掩码和膨胀掩码有重叠，可能需要clip或使用逻辑操作)
    buffer_region = (buffer_region > 0).squeeze().int()

    return buffer_region


def compute_gradient_modulus_sum_vectorized(hyperspectral_image):
    """
    向量化版本，同时计算所有波段的梯度
    """
    C, H, W = hyperspectral_image.shape
    
    # Sobel滤波器
    sobel_x = torch.tensor([[-1, 0, 1], 
                           [-2, 0, 2], 
                           [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat((1, C, 1, 1))
    sobel_y = torch.tensor([[-1, -2, -1], 
                           [0, 0, 0], 
                           [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat((1, C, 1, 1))
    
    device = hyperspectral_image.device
    sobel_x = sobel_x.to(device)
    sobel_y = sobel_y.to(device)
    
    # 为所有波段添加batch维度
    input_4d = hyperspectral_image.unsqueeze(0)  # 形状: (1, C, H, W)
    
    # 计算所有波段的梯度
    grad_x = F.conv2d(input_4d, sobel_x, padding=1, groups=1)  # 形状: (1, C, H, W)
    grad_y = F.conv2d(input_4d, sobel_y, padding=1, groups=1)  # 形状: (1, C, H, W)
    
    # 移除channel维度并计算模长
    grad_x = grad_x.squeeze(0)  # 形状: (C, H, W)
    grad_y = grad_y.squeeze(0)  # 形状: (C, H, W)
    
    # 计算每个波段的梯度模长并求和
    modulus = torch.sqrt(grad_x**2 + grad_y**2)  # 形状: (C, H, W)
    G = torch.sum(modulus, dim=0)  # 形状: (H, W)
    
    return G.detach().cpu().numpy()


def compute_foreground_spectral_variance(
    hyperspectral_image: torch.Tensor, 
    mask: torch.Tensor
) -> float:
    """
    计算前景区域的光谱方差指标 I_Var，使用协方差矩阵的迹 (Trace) 作为衡量标准。
    I_Var = Trace(Sigma_F) = Sum(Var_b)

    Args:
        hyperspectral_image (torch.Tensor): 原始高光谱图像 (C, H, W)，C是波段数。
        mask (torch.Tensor): 待评估的二值掩码 (H, W)，前景为True/1，背景为False/0。

    Returns:
        float: 前景光谱方差指标 I_Var。值越小，前景内部光谱越纯净。
    """
    
    # 1. 确保掩码和图像在同一设备上
    if hyperspectral_image.device != mask.device:
        mask = mask.to(hyperspectral_image.device)

    # 2. 提取前景光谱向量
    
    # 将 mask 展平并转换为布尔类型
    mask_flat = mask.flatten()
    
    # 将 HSI 重塑为 (C, H*W)
    H_flat = hyperspectral_image.view(hyperspectral_image.shape[0], -1)
    
    # 使用布尔索引提取前景像素的光谱向量
    # result_flat 的形状是 (C, N_F)，其中 N_F 是前景像素数量
    foreground_spectra = H_flat[:, mask_flat.bool()]

    # 3. 检查前景像素数量
    N_F = foreground_spectra.shape[1]
    if N_F <= 1:
        # 如果前景像素太少或只有1个，无法计算方差/协方差
        return 0.0 # 或者返回一个很大的值作为惩罚，这里返回0.0表示完美纯净（无方差）
    
    # 4. 计算前景光谱协方差矩阵 (Sigma_F)
    
    # torch.cov 要求输入形状为 (N, C)，即 N 个样本，C 个特征
    # 我们当前是 (C, N_F)，需要转置为 (N_F, C)
    
    # PyTorch 的 cov 函数默认计算样本协方差 (除以 N-1)
    # 结果 Sigma_F 的形状是 (C, C)
    Sigma_F = torch.cov(foreground_spectra.T)

    # 5. 计算协方差矩阵的迹 (Trace)
    
    # 协方差矩阵的迹是其对角线元素之和：Trace(Sigma_F) = sum(Sigma_F[i, i])
    # 这等价于所有波段方差的和：sum(Var_b)
    I_Var = torch.trace(Sigma_F)
    
    return I_Var.item() / N_F


def compute_relaxed_edge_consistency(
    hyperspectral_image: torch.Tensor,
    mask: torch.Tensor,
    d_tol: float = 2.0,
    buffer_k: int = 5,
    percentile_threshold: float = 50
) -> float:
    """
    计算基于容忍度的边缘一致性指标 (Relaxed Edge Consistency, Ĉ_Edge)。

    Args:
        hyperspectral_image (torch.Tensor): 原始高光谱图像 (C, H, W)。
        mask (torch.Tensor): 待评估的二值掩码 (H, W)。
        d_tol (float): 容忍距离，掩码边缘点距离图像边缘的最大距离（像素）。
        buffer_k (int): 用于计算局部阈值的缓冲区扩展像素数。
        percentile_threshold (float): 用于确定局部图像边缘的百分位阈值。

    Returns:
        float: 容忍度边缘一致性得分 Ĉ_Edge ([0, 1])。
    """
    # 1. 初始化和类型转换
    H, W = mask.shape
    
    # 将 mask 转换为 NumPy 数组进行边界提取
    mask_np = mask.cpu().numpy().astype(np.int8)

    # 2. 提取掩码边缘 E_Mask
    # 使用 skimage.segmentation.find_boundaries 来获取像素级的掩码边界
    E_Mask = find_boundaries(mask_np, mode='inner', connectivity=1)
    
    # 检查掩码是否为空，如果为空则无法计算，返回 0.0 或 NaN
    if not np.any(E_Mask):
        return 0.0 
    
    # 3. 计算高光谱梯度图 G 和图像边缘 E_Image (基于局部阈值)
    G_np = compute_gradient_modulus_sum_vectorized(hyperspectral_image)
    
    # 3a. 计算局部阈值区域 R_Buffer
    # 我们使用 dilate 来模拟缓冲区，为了简洁，这里使用 scipy 的 binary_dilation
    from scipy.ndimage import binary_dilation
    
    # 扩展掩码以确定局部区域
    struct_element = np.ones((2 * buffer_k + 1, 2 * buffer_k + 1))
    mask_dilated = binary_dilation(mask_np, structure=struct_element)
    
    # 缓冲区 R 是 (膨胀区域 OR 原始掩码)，也可以仅使用膨胀区域
    R_Buffer = mask_dilated
    
    # 3b. 确定局部阈值 tau_Local
    G_R = G_np[R_Buffer] # 获取缓冲区内的梯度值
    if len(G_R) == 0:
         # 缓冲区为空，可能是掩码太小
         return 0.0
    if 0 <= percentile_threshold <= 1:
        tau_Local = np.percentile(G_R, percentile_threshold * 100)
    else:
        tau_Local = percentile_threshold
    
    # 3c. 生成图像边缘 E_Image
    # 在整个 G_np 上应用局部阈值，生成二值图
    E_Image = (G_np > tau_Local).astype(np.int8)
    
    # 4. 计算距离图 D_Image
    # distance_transform_edt 计算到最近前景像素（E_Image=1）的欧氏距离
    # 注意：如果 E_Image 为空，这个函数会返回一个最大值矩阵
    if not np.any(E_Image):
        # 如果图像中没有高于阈值的边缘，则我们没有可比较的真实边缘
        return 0.0
        
    D_Image = distance_transform_edt(1 - E_Image) # 1 - E_Image 使得 E_Image 的前景变为 0

    # 5. 计算松弛匹配点的数量 |M_Relaxed|
    
    # 获取掩码边缘点对应的距离值
    # D_Image[E_Mask] 得到一个数组，其中包含 E_Mask 上所有点到 E_Image 的最短距离
    distances_at_mask_edge = D_Image[E_Mask]

    # 匹配条件: 距离 <= 容忍度
    M_Relaxed_count = np.sum(distances_at_mask_edge <= d_tol)
    
    # 6. 计算最终指标
    E_Mask_count = np.sum(E_Mask)
    E_Image_count = np.sum(E_Image)
    
    relaxed_consistency_score = M_Relaxed_count / E_Mask_count
    ratio = E_Mask_count / E_Image_count
    
    return relaxed_consistency_score, ratio


def compute_spectral_purity(hsi_image, gt):
    """
    计算高光谱图像中特定区域（gt==1）的光谱纯度（一致性）
    
    参数:
        hsi_image: 高光谱图像，形状为 (C, H, W)
        gt: 地面真实标签，形状为 (H, W)，值为1的位置表示目标区域
        
    返回:
        spectral_purity: 光谱纯度值（平均余弦相似度）
    """
    # 获取目标区域的掩码（展平后）
    gt_flat = gt.reshape(-1)
    mask = (gt_flat == 1)
    
    # 如果目标区域内没有像素，返回0
    if torch.sum(mask) == 0:
        return 0.0
    
    # 重塑高光谱数据并提取目标光谱
    spectra = hsi_image.reshape(hsi_image.shape[0], -1)[:, mask]  # 形状 (C, k)
    
    # 归一化光谱向量（按列归一化）
    norm = torch.linalg.norm(spectra, axis=0, keepdims=True)
    # 避免除以零
    norm[norm == 0] = 1e-10
    spectra_norm = spectra / norm
    
    # 计算所有光谱对之间的余弦相似度矩阵
    cosine_sim_matrix = torch.mm(spectra_norm.T, spectra_norm)  # 形状 (k, k)
    
    # 计算非对角线元素的平均值（排除自相似）
    k = cosine_sim_matrix.shape[0]
    if k > 1:
        # 提取非对角线元素
        off_diag_sum = torch.sum(cosine_sim_matrix) - torch.trace(cosine_sim_matrix)
        num_pairs = k * (k - 1)
        spectral_purity = off_diag_sum / num_pairs
    else:
        spectral_purity = 1.0  # 单个光谱的纯度为1
    
    mean_spectra = torch.mean(spectra_norm, axis=1, keepdims=True)
    mean_spectra /= torch.linalg.norm(mean_spectra, axis=0, keepdims=True)

    mean_sim = (spectra_norm * mean_spectra).sum(0)
    less_than = torch.sum((mean_sim < 0.85)) / (torch.sum(mask))
    del mean_spectra, cosine_sim_matrix

    return spectral_purity, less_than

def rsshow(I, scale=0.005):
    low, high = torch.quantile(I, torch.tensor([scale, 1-scale]).to(I.device))
    I[I > high] = high
    I[I < low] = low
    I = (I-low)/(high-low)
    return I


def interpolate_hyperspectral_image_transform_matrix(wave_lib, target_wavelengths):
    if len(target_wavelengths.shape) == 1:
        target_wavelengths = target_wavelengths[None]
    wave_lib_np = wave_lib
    wave_current_np = target_wavelengths
    f = interp1d(wave_lib_np, np.arange(len(wave_lib_np)))
    temp = [f(t) for t in wave_current_np]
    transform_matrix = []
    for t in temp:
        transform_matrix_curr = np.zeros((wave_current_np.shape[1],
                                          len(wave_lib_np)),
                                         dtype=np.float32)
        idx0 = np.arange(wave_current_np.shape[1], dtype=np.int64)
        idx1 = np.floor(t).astype(np.int64)
        idx2 = np.ceil(t).astype(np.int64)
        w1 = 1 - t % 1
        w2 = t % 1
        transform_matrix_curr[idx0, idx2] = w2
        transform_matrix_curr[idx0, idx1] = w1
        transform_matrix.append(transform_matrix_curr)
    transform_matrix = np.stack(transform_matrix)
    return transform_matrix


wavelengths = [400.0 + (2100.0 / 112.0) * i for i in range(1, 113)] # needed to be modified
device = "cuda" if torch.cuda.is_available() else "cpu"
parser = argparse.ArgumentParser()
parser.add_argument('--index_file', type=str, default='/data2/pl/ImageTask/wxq/HyperSeg_lp/data/hyperfree_index.json')
parser.add_argument('--threshold', type=float, default=0.36)
parser.add_argument('--delete', action='store_true')
args = parser.parse_args()

# 读取JSON索引文件
with open(args.index_file, 'r') as f:
    index_data = json.load(f)

samples = index_data['samples']
print(f"Total samples in index: {len(samples)}")

# 遍历JSON中的samples，检查GT是否存在
for i, sample in enumerate(samples):
    img_path = sample['img_path']
    gt_path = sample['gt_path']
    
    # 检查GT是否存在
    if os.path.exists(gt_path):
        gt = torch.tensor(np.load(gt_path))
        if torch.sum(gt) <= 100:
            continue

        image = tifffile.imread(img_path).transpose(2, 0, 1)  # return numpy array | convert to [c,h,w]
        image = torch.tensor(image.astype(np.float32))
        image = (image - image.min()) / (image.max() - image.min())

        if torch.isnan(image).any():
            continue
        elif image.max().item() > 1:
            continue
        elif image.min().item() < 0:
            continue
        

        # compute spectral purity\edge score\variance
        try:
            image = image.to(device)
            gt = gt.to(device)
            spectral_purity, _ = compute_spectral_purity(image, gt)
            relaxed_consistency_score, _ = compute_relaxed_edge_consistency(image, gt, percentile_threshold=0.9)
            var = compute_foreground_spectral_variance(image, gt)
            weighted_score = spectral_purity * 0.35 + relaxed_consistency_score * 0.2  - var * 0.45 * 10

            print(f"{i} {gt_path} {spectral_purity:.4f} {relaxed_consistency_score:.4f} {var:.4f} {weighted_score:.4f}")
            if weighted_score < args.threshold:
                if args.delete:
                    os.remove(gt_path)
            del image, gt
            torch.cuda.empty_cache()
        except:
            print(f"{i} {gt_path} error")
            del image, gt
            continue