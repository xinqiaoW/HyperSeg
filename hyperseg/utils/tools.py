import torch
import spectral as spy
from scipy import ndimage
import numpy as np
import os

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


def create_point_coords(gt: torch.Tensor, num_points: int = 5):
    '''
    create point coords from gt mask. generate prompts for sam
    '''
    B, H, W = gt.shape
    point_coords = torch.zeros(B, num_points, 2, dtype=torch.float)
    for i in range(B):
        idx = torch.nonzero(gt[i])
        n = min(num_points, idx.shape[0])
        rand_perm = torch.randperm(idx.shape[0])[:n]
        idx = idx[rand_perm]
        idx = torch.flip(idx, dims=[1])
        point_coords[i] = idx
    
    return point_coords # (B, num_points, 2)


def build_input_for_hyperseg(images, shape, point_coords, point_labels, device='cuda'):
    '''
    build input for hyperseg
    '''
    batched_input = []
    B = images.shape[0]
    for i in range(B):
        batched_input.append({
            "image": images[i],
            "original_size": shape,
            "point_coords": point_coords[i].unsqueeze(0),
            "point_labels": point_labels[i].unsqueeze(0),
        })
    return batched_input


def transform_output_seg(batched_output):
    '''
    transform output from hyperseg to proper expression
    '''
    return torch.stack([out["logits"][0, 0, :, :] for out in batched_output], dim=0), torch.stack([out["masks"][0, 0, :, :] for out in batched_output], dim=0)


def seg_call(net, batched_input, wavelengths, multimask_output=False, GSD=torch.tensor([1.0])):
    return net(batched_input, wavelengths=wavelengths, multimask_output=multimask_output, GSD=GSD)


def rsshow(I, scale=0.005):
    low, high = torch.quantile(I, torch.tensor([scale, 1-scale]).to(I.device))
    I[I > high] = high
    I[I < low] = low
    I = (I-low)/(high-low)
    return I


def pca_dr(src):
    src = src.transpose((1, 2, 0))  # Change to (H, W, C)
    input_image = src.copy()
    pc = spy.principal_components(src)
    pc_98 = pc.reduce(fraction=0.98)  # 保留98%的特征值
    img_pc = pc_98.transform(input_image)  # 把数据转换到主成分空间
    return img_pc[:, :, :3]


def command(path):
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    test_py = os.path.join(root, "scripts", "test.py")
    log_path = os.path.join(root, "logs", "results.out")
    com = f"CUDA_VISIBLE_DEVICES=6 python {test_py} --channel_proj_spectral --remove --checkpoint {path} >> {log_path}"
    return com


def extract_connected_components(tensor):
    """
    提取二值tensor中的连通区域
    
    参数:
        tensor: 二维numpy数组，只包含0和1
    
    返回:
        components: 列表，每个元素是一个二维numpy数组，表示一个连通区域
        labels: 标记后的数组，每个连通区域有唯一的标签
        num_components: 连通区域的数量
    """
    # 使用4连通性（上下左右）进行标记
    labeled_array, num_components = ndimage.label(tensor, structure=[[0,1,0],[1,1,1],[0,1,0]])
    
    components = []
    for i in range(1, num_components + 1):
        # 为每个连通区域创建一个新的tensor
        component = (labeled_array == i).astype(np.int32)
        components.append(component)
    
    return components, labeled_array, num_components


def extract_point_from_error_region(pred:torch.Tensor, gt:torch.Tensor):
    # [pred (1, H, W), gt (1, H, W) ] or all (H, W)
    if len(pred.shape) == 3:
        pred = pred[0]
        gt = gt[0]
    H, W = pred.shape
    pred_bool = pred.bool().int()
    gt_bool = gt.bool().int()

    false_pos = pred_bool & (~gt_bool)
    false_neg = (~pred_bool) & gt_bool

    false_pos_components, false_pos_labels, num_false_pos_components = extract_connected_components(false_pos.detach().cpu().numpy())
    false_neg_components, false_neg_labels, num_false_neg_components = extract_connected_components(false_neg.detach().cpu().numpy())
    if false_pos_components != [] and false_neg_components != []:
        largest_fp_component = max(false_pos_components, key=lambda x: x.sum())
        largest_fn_component = max(false_neg_components, key=lambda x: x.sum())
        if largest_fp_component.sum() > largest_fn_component.sum():
            largest_component = largest_fp_component
            point_labels = torch.tensor([[0.]])
        else:
            largest_component = largest_fn_component
            point_labels = torch.tensor([[1.]])
    
    elif false_pos_components == [] and false_neg_components == []:
        point_labels = torch.tensor([[1.]])
        largest_component = gt
        
    elif false_neg_components == []:
        largest_fp_component = max(false_pos_components, key=lambda x: x.sum())
        largest_component = largest_fp_component
        point_labels = torch.tensor([[0.]])
    elif false_pos_components == []:
        largest_fn_component = max(false_neg_components, key=lambda x: x.sum())
        largest_component = largest_fn_component
        point_labels = torch.tensor([[1.]])
    
    point_coords = create_point_coords(torch.tensor(largest_component, dtype=torch.float).unsqueeze(0).to(gt.device), num_points=1)

    return point_coords, point_labels

@torch.no_grad()
def update_points(point_coords, point_labels, pred, gt):
    B, _, _ = pred.shape
    coords_list = []
    labels_list = []
    for i in range(B):
        extra_coords, extra_labels = extract_point_from_error_region(pred[i], gt[i])
        coords_list.append(extra_coords)
        labels_list.append(extra_labels)
    coords = torch.cat(coords_list, dim=0).to(gt.device)
    labels = torch.cat(labels_list, dim=0).to(gt.device)

    point_coords = torch.cat([point_coords, coords], dim=1)
    point_labels = torch.cat([point_labels, labels], dim=1)
    return point_coords, point_labels


class Meaner(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count