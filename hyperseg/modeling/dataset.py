import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import tifffile
import re
import json
from scipy import sparse
from ..utils.tools import extract_connected_components


def extract_all_filenames(root_dir):
    """
    提取指定目录及其所有子目录中的文件名
    
    Args:
        root_dir (str): 要遍历的根目录路径
        
    Returns:
        list: 包含所有文件名的列表
    """
    all_filenames = []
    
    # 遍历目录树
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 将当前目录中的文件名添加到列表
        for filename in filenames:
            all_filenames.append(filename)
    
    return all_filenames

def remove_trailing_digits(input_string):
    return re.sub(r'\d+$', '', input_string)

class CustomDataset(Dataset):
    def __init__(self, images_dir=None, gt_dir=None, transform=None, data_name='LongKou', index_file=None):
        """
        :param images_dir: image path (可选，如果提供 index_file 则可为 None)
        :param gt_dir: gt path (可选，如果提供 index_file 则可为 None)
        :param transform: transform
        :param data_name: dataset name
        :param index_file: 预生成的索引文件路径 (JSON格式)，如果提供则直接加载，极快
        """
        self.transform = transform
        self.samples = []
        
        # 优先使用索引文件
        if index_file is not None and os.path.exists(index_file):
            print(f"Loading dataset from index file: {index_file}")
            self._load_from_index(index_file)
        else:
            # 传统方式：扫描目录
            if images_dir is None or gt_dir is None:
                raise ValueError("Either index_file or (images_dir, gt_dir) must be provided")
            
            self.images_dir = images_dir
            self.gt_dir = gt_dir
            self.name = data_name
            self.gsd_dir = os.path.join(self.images_dir, '..', 'gsd')
            print(f"Scanning dataset directories (this may be slow for large datasets)...")
            self._prepare_samples()
        
    
    def _load_from_index(self, index_file):
        """从预生成的索引文件加载数据集信息（极快）"""
        with open(index_file, 'r') as f:
            data = json.load(f)
        
        self.images_dir = data.get('images_dir', '')
        self.gt_dir = data.get('gt_dir', '')
        self.name = data.get('data_name', 'HyperFree')
        self.gsd_dir = data.get('gsd_base_dir', '')
        
        # 加载样本列表
        for sample in data['samples']:
            if sample['type'] == 'npy':
                self.samples.append((
                    sample['img_path'],
                    sample['gt_path'],
                    sample['gsd_path']
                ))
            elif sample['type'] == 'LongKou':
                self.samples.append((
                    'LongKou',
                    sample['img_path'],
                    sample['gt_path'],
                    sample['class_k'],
                    sample['component_idx']
                ))
        
        print(f"✓ Loaded {len(self.samples)} samples from index file")
    
    def _prepare_samples(self):
        """传统方式：扫描目录准备样本（慢，不推荐用于大数据集）"""
        print("Warning: Scanning directories directly. For large datasets, use prepare_dataset_index.py to generate an index file first.")
        
        gt_files = list(os.scandir(self.gt_dir))
 
        for gtfile in gt_files:
            gt_path = gtfile.path
            gt_file = os.path.basename(gt_path)
            
            if gt_file.endswith('.npy'):
                # 与 prepare_dataset_index 中保持一致：
                # 标签/图像基名相同，GSD 基名比它们少 3 个字符（如去掉 "_LR"），仅后缀不同
                name, _ = os.path.splitext(gt_file)

                # 图像：优先 .tif，如不存在则尝试 .tiff
                img_path = os.path.join(self.images_dir, name + '.tif')
                if not os.path.exists(img_path):
                    alt_img_path = os.path.join(self.images_dir, name + '.tiff')
                    if os.path.exists(alt_img_path):
                        img_path = alt_img_path
                    else:
                        continue

                # GSD：先尝试去掉末尾 3 个字符的基名，再回退到完整基名
                if len(name) > 3:
                    gsd_basename = name[:-3]
                else:
                    gsd_basename = name

                gsd_path = os.path.join(self.gsd_dir, gsd_basename + '.txt')
                if not os.path.exists(gsd_path):
                    alt_gsd_path = os.path.join(self.gsd_dir, name + '.txt')
                    if os.path.exists(alt_gsd_path):
                        gsd_path = alt_gsd_path
                    else:
                        continue

                # Only check file existence, defer validation to __getitem__
                if os.path.exists(img_path) and os.path.exists(gsd_path):
                    self.samples.append((img_path, gt_path, gsd_path))
                    
            elif gt_file.endswith('.tif'):
                if self.name == 'LongKou':
                    img_path = './Data/LongKou/WHU-Hi-LongKou.tif'
                    gt_full_path = './Data/LongKou/WHU-Hi-LongKou_gt.tif'
                    
                    # For LongKou, we need to load once to get component info
                    # Cache this to avoid repeated loading
                    if not hasattr(self, '_longkou_cache'):
                        try:
                            gt = tifffile.imread(gt_full_path)
                            gt_torch = torch.tensor(gt.astype(np.float32))
                            
                            gt_min = gt_torch.min()
                            gt_max = gt_torch.max()
                            
                            for k in range(gt_min.int().item(), gt_max.int().item() + 1):
                                gt_in = (gt_torch == k).float()
                                components, labeled_array, num_components = extract_connected_components(gt_in.detach().cpu().numpy())
                                for component_idx, component in enumerate(components):
                                    mask = torch.tensor(component.astype(np.float32))
                                    if mask.sum() <= 100:
                                        continue
                                    self.samples.append(('LongKou', img_path, gt_full_path, k, component_idx))
                            
                            self._longkou_cache = True
                        except Exception as e:
                            print(f"Error processing LongKou data: {e}")
                            self._longkou_cache = True
        
        print(f"✓ Found {len(self.samples)} samples")
    
    def __len__(self):
        """Return the number of samples"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample by index with validation"""
        sample_info = self.samples[idx]
        
        # Handle regular npy samples
        if len(sample_info) == 3:
            img_path, gt_path, gsd_path = sample_info

            # Load ground truth and validate
            temp = np.load(gt_path)
            unique_values = np.unique(temp[temp > 0])

            # Check if there are any positive values
            if len(unique_values) == 0:
                # Return a random valid sample if current one is invalid
                return self.__getitem__((idx + 1) % len(self))

            gt = torch.tensor(temp == np.random.choice(unique_values))
            if torch.sum(gt) <= 100:
                # Return a random valid sample if current one is invalid
                return self.__getitem__((idx + 1) % len(self))
            
            # Load image
            image = tifffile.imread(img_path).transpose(2, 0, 1)  # return numpy array | convert to [c,h,w]
            image = torch.tensor(image.astype(np.float32))
            image = (image - image.min()) / (image.max() - image.min())
            
            # Validate image
            if torch.isnan(image).any() or image.max().item() > 1 or image.min().item() < 0:
                return self.__getitem__((idx + 1) % len(self))
            
            # Load gsd
            with open(gsd_path, 'r') as f:
                gsd = float(f.read())
                gsd = torch.tensor([gsd])
            
            # Apply transform if provided
            if self.transform:
                image = self.transform(image)
            
            return (image, gsd), gt
        
        # Handle LongKou samples
        elif len(sample_info) == 5:
            dataset_name, img_path, gt_full_path, class_k, component_idx = sample_info
            
            # Load full image and ground truth
            img = tifffile.imread(img_path)
            gt = tifffile.imread(gt_full_path)
            
            img = torch.tensor(img.astype(np.float32))
            img = (img - img.min()) / (img.max() - img.min())
            
            gt_torch = torch.tensor(gt.astype(np.float32))
            gt_in = (gt_torch == class_k).float()
            
            # Extract the specific component
            components, labeled_array, num_components = extract_connected_components(gt_in.detach().cpu().numpy())
            component = components[component_idx]
            mask = torch.tensor(component.astype(np.float32))
            
            # Apply transform if provided
            if self.transform:
                img = self.transform(img)
            
            return img, (gt_in * mask).float()
        

    
    def remove(self, idx):
        self.samples.pop(idx)


if __name__ == "__main__":
    dataset = CustomDataset(
        "/data2/pl/HSITask/data/HyperFree/data_compressed",
        "/data2/pl/HSITask/data/HyperFree/gt_compressed/ground_truth"
    )
    print(len(dataset))
    # for i in range(len(dataset)):
    #     image, gt = dataset[i]
    #     print(gt.sum())


class SpaceNetDataset(Dataset):
    """
    Dataset for SpaceNet ground truth and reconstructed HSI from hsi_info.

    Args:
        ground_truth_dir: Directory containing ground truth .npz files (with 'masks' key)
        hsi_info_dir: Directory containing HSI info subdirectories (SN1_*, SN2_*)
        endmember_lib_path: Path to endmember_libraries.npz file
        dataset: Which dataset to use ('SN1', 'SN2', or 'all')
        transform: Optional transform to apply
        target_size: Target size for HSI reconstruction (default: 128)
    """

    # Mapping from ground truth prefix to HSI info prefix
    GT_TO_HSI_PREFIX = {
        '3band_': '8band_',           # SN1
        'RGB-PanSharpen_': 'MUL_',    # SN2
    }

    # Mapping from ground truth pattern to HSI info subdirectory
    GT_TO_HSI_SUBDIR = {
        # SN1
        '3band_AOI_1_RIO_': 'SN1_SN1_train_8band',
        '3band_AOI_2_RIO_': 'SN1_SN1_test_8band',
        # SN2 Vegas
        'RGB-PanSharpen_AOI_2_Vegas_': ['SN2_AOI_2_Vegas_Train', 'SN2_AOI_2_Vegas_Test_public'],
        # SN2 Paris
        'RGB-PanSharpen_AOI_3_Paris_': ['SN2_AOI_3_Paris_Train', 'SN2_AOI_3_Paris_Test_public'],
        # SN2 Shanghai
        'RGB-PanSharpen_AOI_4_Shanghai_': ['SN2_AOI_4_Shanghai_Train', 'SN2_AOI_4_Shanghai_Test_public'],
        # SN2 Khartoum
        'RGB-PanSharpen_AOI_5_Khartoum_': ['SN2_AOI_5_Khartoum_Train', 'SN2_AOI_5_Khartoum_Test_public'],
    }

    def __init__(
        self,
        ground_truth_dir,
        hsi_info_dir,
        endmember_lib_path,
        dataset='all',
        transform=None,
        target_size=128,
    ):
        self.ground_truth_dir = ground_truth_dir
        self.hsi_info_dir = hsi_info_dir
        self.transform = transform
        self.target_size = target_size
        self.dataset = dataset

        # Load endmember libraries
        print(f"Loading endmember libraries from {endmember_lib_path}")
        lib_data = np.load(endmember_lib_path)
        self.A_hsi = lib_data['A_hsi']  # (num_bands, num_endmembers)

        # Collect valid samples
        self.samples = []
        self._collect_samples()
        print(f"SpaceNetDataset: Found {len(self.samples)} valid samples")

    def _get_hsi_name(self, gt_name):
        """Convert ground truth name to HSI info name."""
        for gt_prefix, hsi_prefix in self.GT_TO_HSI_PREFIX.items():
            if gt_name.startswith(gt_prefix):
                return gt_name.replace(gt_prefix, hsi_prefix, 1)
        return None

    def _get_hsi_subdir(self, gt_name):
        """Get HSI info subdirectory for a ground truth file."""
        for pattern, subdirs in self.GT_TO_HSI_SUBDIR.items():
            if gt_name.startswith(pattern):
                if isinstance(subdirs, list):
                    return subdirs
                return [subdirs]
        return []

    def _is_sn1(self, gt_name):
        """Check if ground truth file is from SN1."""
        return gt_name.startswith('3band_')

    def _is_sn2(self, gt_name):
        """Check if ground truth file is from SN2."""
        return gt_name.startswith('RGB-PanSharpen_')

    def _collect_samples(self):
        """Collect all valid (ground_truth, hsi_info) pairs."""
        gt_files = [f for f in os.listdir(self.ground_truth_dir) if f.endswith('.npz')]

        for gt_file in gt_files:
            gt_name = gt_file.replace('.npz', '')

            # Filter by dataset
            if self.dataset == 'SN1' and not self._is_sn1(gt_name):
                continue
            if self.dataset == 'SN2' and not self._is_sn2(gt_name):
                continue

            # Get HSI info name and subdirectories
            hsi_name = self._get_hsi_name(gt_name)
            if hsi_name is None:
                continue

            subdirs = self._get_hsi_subdir(gt_name)

            # Find HSI info files
            hsi_x_path = None
            hsi_scale_path = None

            for subdir in subdirs:
                x_path = os.path.join(self.hsi_info_dir, subdir, f"{hsi_name}_X.npz")
                scale_path = os.path.join(self.hsi_info_dir, subdir, f"{hsi_name}_scale.npz")

                if os.path.exists(x_path) and os.path.exists(scale_path):
                    hsi_x_path = x_path
                    hsi_scale_path = scale_path
                    break

            if hsi_x_path is None:
                continue

            gt_path = os.path.join(self.ground_truth_dir, gt_file)

            # Load ground truth to get number of masks
            gt_data = np.load(gt_path)
            masks = gt_data['masks']
            num_masks = masks.shape[0]

            # Add one sample per mask
            for mask_idx in range(num_masks):
                self.samples.append({
                    'gt_path': gt_path,
                    'hsi_x_path': hsi_x_path,
                    'hsi_scale_path': hsi_scale_path,
                    'mask_idx': mask_idx,
                    'name': gt_name,
                })

    def _load_sparse_matrix(self, npz_path):
        """Load sparse matrix from npz file."""
        data = np.load(npz_path)
        return sparse.csr_matrix(
            (data['data'], data['indices'], data['indptr']),
            shape=tuple(data['shape'])
        )

    def _reconstruct_hsi(self, hsi_x_path, hsi_scale_path):
        """Reconstruct HSI from abundance maps and endmember library."""
        # Load sparse abundance matrix
        X_sparse = self._load_sparse_matrix(hsi_x_path)
        X = X_sparse.toarray()  # (num_endmembers, num_pixels)

        # Load scaling factor
        scale_data = np.load(hsi_scale_path)
        scaling_factor = scale_data['scaling_factor']  # (1, num_pixels)

        # Reconstruct HSI: Y = A @ X * scale
        Y = self.A_hsi @ X  # (num_bands, num_pixels)
        Y = Y * scaling_factor  # Apply scaling

        # Reshape to image
        num_pixels = Y.shape[1]
        h = w = int(np.sqrt(num_pixels))
        if h * w != num_pixels:
            # Try to find correct dimensions
            for i in range(1, int(np.sqrt(num_pixels)) + 1):
                if num_pixels % i == 0:
                    h = i
                    w = num_pixels // i

        hsi = Y.reshape(-1, h, w)  # (num_bands, H, W)
        return hsi

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load and reconstruct HSI
        hsi = self._reconstruct_hsi(sample['hsi_x_path'], sample['hsi_scale_path'])

        # Load mask
        gt_data = np.load(sample['gt_path'])
        masks = gt_data['masks']
        mask = masks[sample['mask_idx']]  # (H, W)

        # Resize mask to match HSI if needed
        if mask.shape != hsi.shape[1:]:
            from scipy.ndimage import zoom
            scale_h = hsi.shape[1] / mask.shape[0]
            scale_w = hsi.shape[2] / mask.shape[1]
            mask = zoom(mask.astype(np.float32), (scale_h, scale_w), order=0)
            mask = (mask > 0.5).astype(np.float32)

        # Convert to tensors
        hsi = torch.tensor(hsi.astype(np.float32))
        mask = torch.tensor(mask.astype(np.float32))

        # Normalize HSI
        hsi = (hsi - hsi.min()) / (hsi.max() - hsi.min() + 1e-8)

        # Apply transform if provided
        if self.transform:
            hsi = self.transform(hsi)

        return hsi, mask

    def get_sample_info(self, idx):
        """Get metadata for a sample."""
        return self.samples[idx]

    @staticmethod
    def select_rgb_bands(num_bands):
        """Select RGB band indices for HSI visualization."""
        if num_bands >= 224:
            return [28, 17, 10]  # AVIRIS RGB bands
        if num_bands >= 100:
            return [int(num_bands * 0.3), int(num_bands * 0.2), int(num_bands * 0.1)]
        return [min(num_bands - 1, 4), min(num_bands - 1, 2), min(num_bands - 1, 1)]

    @staticmethod
    def prepare_rgb(hsi, bands=None):
        """
        Prepare RGB image from HSI for display with percentile stretch.

        Args:
            hsi: HSI tensor or numpy array with shape (C, H, W)
            bands: RGB band indices, if None will auto-select based on num_bands

        Returns:
            RGB image as numpy array with shape (H, W, 3), values in [0, 1]
        """
        if isinstance(hsi, torch.Tensor):
            hsi = hsi.numpy()

        if bands is None:
            bands = SpaceNetDataset.select_rgb_bands(hsi.shape[0])

        rgb = np.stack([hsi[b] for b in bands], axis=-1)
        for i in range(3):
            p2, p98 = np.percentile(rgb[:, :, i], [2, 98])
            if p98 > p2:
                rgb[:, :, i] = np.clip((rgb[:, :, i] - p2) / (p98 - p2), 0, 1)
            else:
                rgb[:, :, i] = rgb[:, :, i] / (np.max(rgb[:, :, i]) + 1e-8)
        return rgb