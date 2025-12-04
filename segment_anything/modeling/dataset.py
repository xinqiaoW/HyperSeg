import os
import glob
import numpy as np
import torch
from torch.utils.data import IterableDataset
from PIL import Image
import tifffile  
import re
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

class CustomDataset(IterableDataset):
    def __init__(self, images_dir, gt_dir, transform=None, data_name='LongKou'):
        """
        :param images_dir: image path
        :param gt_dir: gt path
        :param transform: transform
        """
        self.images_dir = images_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.name = data_name
        self.gsd_dir = os.path.join(self.images_dir, '..', 'gsd')
        
    
    def __iter__(self):
        
        # gt_files = glob.glob(os.path.join(self.gt_dir, '*.npy'))
        gt_files = os.scandir(self.gt_dir)
 
        for i, gtfile in enumerate(gt_files):
            # gain basename
            gt_path = gtfile.path
            gt_file = os.path.basename(gt_path)
            if gt_file.endswith('.npy'):
                name, _ = gt_file.split(".")
                # basename = remove_trailing_digits(name) + '.tif'
                # name_1, name_2 = name[:-3], name[-2:]
                basename = name + '.tif'
                print(basename)
                
                if os.path.exists(os.path.join(self.images_dir, basename)):
                    all_gts = torch.tensor(np.load(gt_path))
                    for cls_num in range(all_gts.max().item()):
                        gt = (all_gts == (cls_num+1)).float()
                        if torch.sum(gt) <= 100:
                            continue
                        # self.samples.append((os.path.join(self.images_dir, basename), gt_path))
                        img_path, gt_path = os.path.join(self.images_dir, basename), gt_path
                        print(img_path)

                        image = tifffile.imread(img_path).transpose(2, 0, 1)  # return numpy array | convert to [c,h,w]
                        image = torch.tensor(image.astype(np.float32))
                        image = (image - image.min()) / (image.max() - image.min())

                        if torch.isnan(image).any():
                            continue
                        elif image.max().item() > 1:
                            continue
                        elif image.min().item() < 0:
                            continue

                        if self.transform:
                            image_tensor = self.transform(image_tensor)
                        
                        try:
                            with open(os.path.join("/data2/pl/HSITask/data/HyperFree/gsd", basename[:-7] + '.txt'), 'r') as f:
                                gsd = float(f.read())
                                gsd = torch.tensor([gsd])
                        except:
                            print(f"gsd file does not exist")
                            continue

                        yield (image, gsd), gt
            elif gt_file.endswith('.tif'):
                if self.name == 'LongKou':
                    mask = tifffile.imread(gt_path)
                    img = tifffile.imread('./Data/LongKou/WHU-Hi-LongKou.tif')
                    gt = tifffile.imread('./Data/LongKou/WHU-Hi-LongKou_gt.tif')
                    gt_torch = torch.tensor(gt.astype(np.float32))
                    img = torch.tensor(img.astype(np.float32))
                    img = (img - img.min()) / (img.max() - img.min())
                    gt_min = gt_torch.min()
                    gt_max = gt_torch.max()
                    for k in range(gt_min.int().item(), gt_max.int().item() + 1):
                        gt_in = (gt_torch == k).float()
                        components, labeled_array, num_components = extract_connected_components(gt_in.detach().cpu().numpy())
                        for i, component in enumerate(components):
                            mask = torch.tensor(component.astype(np.float32))
                            if mask.sum() <= 100:
                                continue
                            yield img, (gt_in * mask).float()
        

    # def __len__(self):
    #     """return the number of samples"""
    #     return len(self.samples)

    # def __getitem__(self, idx):
    #     img_path, gt_path = self.samples[idx]
        
    #     # read tif file
        # image = tifffile.imread(img_path).transpose(2, 0, 1)  # return numpy array | convert to [c,h,w]
        # image = torch.tensor(image.astype(np.float32))
        # image = (image - image.min()) / (image.max() - image.min())

    #     # read gt file
    #     gt = torch.tensor(np.load(gt_path))
        
    #     # apply transform
        # if self.transform:
        #     image_tensor = self.transform(image_tensor)
        
    #     return image, gt
    
    def remove(self, idx):
        self.samples.pop(idx)


if __name__ == "__main__":
    dataset = CustomDataset(
        "/data2/pl/HSITask/data/HyperFree/data_compressed",
        "/data2/pl/HSITask/data/HyperFree/my_gt_version_3"
    )
    for (image, gsd), gt in dataset:
        pass
    # for i in range(len(dataset)):
    #     image, gt = dataset[i]
    #     print(gt.sum())