import json
import os
import numpy as np
import torch
import tifffile
import matplotlib.pyplot as plt
from matplotlib import colors
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse

def rsshow(I, scale=0.005):
    low, high = np.quantile(I, [scale, 1-scale])
    I[I > high] = high
    I[I < low] = low
    I = (I-low)/(high-low+1e-8)
    return I


def create_label_colormap(num_labels):
    """Create a colormap for different labels"""
    if num_labels <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.get_cmap('hsv', num_labels + 1)
    
    label_colors = np.zeros((num_labels + 1, 3))
    label_colors[0] = [0, 0, 0]  # Background is black
    for i in range(1, num_labels + 1):
        label_colors[i] = cmap(i / num_labels)[:3]
    
    return label_colors


def visualize_pair(img_path, gt_path, output_dir):
    """Visualize one image-GT pair and save to file"""
    try:
        # Load image
        image = tifffile.imread(img_path).transpose(2, 0, 1)  # [C, H, W]
        image = torch.tensor(image.astype(np.float32))
        image = (image - image.min()) / (image.max() - image.min())
        
        # Load GT
        gt = np.load(gt_path, allow_pickle=True)
        
        # Create false color image (bands 40, 20, 10)
        C, H, W = image.shape
        bands = [min(39, C-1), min(19, C-1), min(9, C-1)]
        rgb = torch.stack([image[bands[0]], image[bands[1]], image[bands[2]]], dim=2)
        rgb = rgb.cpu().numpy()
        rgb = rsshow(rgb)
        rgb = np.clip(rgb, 0, 1)


        # Create colored GT visualization
        unique_labels = np.unique(gt)
        num_labels = int(unique_labels.max())
        label_colors = create_label_colormap(num_labels)
        
        gt_colored = np.zeros((H, W, 3))
        for label in unique_labels:
            mask = (gt == label)
            gt_colored[mask] = label_colors[int(label)]
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Remove margins and spacing
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)
        
        # Plot image
        axes[0].imshow(rgb)
        axes[0].axis('off')
        
        # Plot GT
        axes[1].imshow(gt_colored)
        axes[1].axis('off')
        
        # Generate output filename
        img_basename = Path(img_path).stem
        gt_basename = Path(gt_path).stem
        output_filename = f"{img_basename}_{gt_basename}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save figure
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close(fig)
        
        return True, output_filename
        
    except Exception as e:
        return False, f"{img_path}: {str(e)}"


def process_sample(args):
    """Process a single sample (wrapper for multiprocessing)"""
    idx, sample, output_dir = args
    img_path = sample['img_path']
    gt_path = sample['gt_path']
    
    # Check if files exist
    if not os.path.exists(img_path) or not os.path.exists(gt_path):
        return idx, False, f"Files not found: {img_path} or {gt_path}"
    
    success, msg = visualize_pair(img_path, gt_path, output_dir)
    return idx, success, msg


def batch_visualize(index_file, output_dir, num_workers=8):
    """Batch visualize all samples in the index file"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load index
    with open(index_file, 'r') as f:
        index_data = json.load(f)
    
    samples = index_data['samples']
    total_samples = len(samples)
    
    print(f"Total samples: {total_samples}")
    print(f"Output directory: {output_dir}")
    print(f"Using {num_workers} workers")
    
    # Prepare arguments for multithreading
    tasks = [(i, sample, output_dir) for i, sample in enumerate(samples)]
    
    # Process with ProcessPoolExecutor
    success_count = 0
    fail_count = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_sample, task): task for task in tasks}
        
        for future in as_completed(futures):
            idx, success, msg = future.result()
            
            if success:
                success_count += 1
                if success_count % 10 == 0:
                    print(f"Progress: {success_count}/{total_samples} - Last saved: {msg}")
            else:
                fail_count += 1
                print(f"Failed [{idx}]: {msg}")
    
    print(f"\nCompleted!")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Batch visualize hyperspectral images and GTs')
    parser.add_argument('--index_file', type=str, 
                       default='/data2/pl/ImageTask/wxq/HyperSeg_lp/data/hyperfree_index_labels_my_hf.json',
                       help='Path to index JSON file')
    parser.add_argument('--output_dir', type=str,
                       default='/data2/pl/HSITask/data/HyperFree/vis_my_hf/',
                       help='Output directory for visualizations')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of worker threads')
    args = parser.parse_args()
    
    if not os.path.exists(args.index_file):
        print(f"Error: Index file not found: {args.index_file}")
        return
    
    batch_visualize(args.index_file, args.output_dir, args.num_workers)


if __name__ == '__main__':
    main()
