import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import tifffile  
import re
import json
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
            gt = torch.tensor(temp == np.random.choice(np.unique(temp[temp > 0])))
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