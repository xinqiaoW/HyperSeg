"""
数据集索引预生成脚本
用于扫描数据集目录并生成文件列表，避免训练时重复扫描百万级文件
使用多线程加速处理
"""

import os
import sys
import re
import json
import argparse
import numpy as np
import tifffile
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED

from tqdm import tqdm
import glob
import subprocess
import time

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hyperseg.utils.tools import extract_connected_components


def remove_trailing_digits(input_string):
    """移除字符串末尾的数字"""
    return re.sub(r'\d+$', '', input_string)


def process_npy_file(args):
    """
    处理单个npy文件，验证并返回样本信息
    
    Args:
        args: (gt_path, images_dir, gsd_base_dir, validate)
    
    Returns:
        tuple or None: (img_path, gt_path, gsd_path) 如果有效，否则 None
    """
    gt_path, images_dir, gsd_base_dir, validate = args
    
    try:
        gt_file = os.path.basename(gt_path)
        if not gt_file.endswith('.npy'):
            return None
        
        # 假设图像、标签基名一致，GSD 的基名比它们少 3 个字符
        name, _ = os.path.splitext(gt_file)
        
        # 优先匹配 .tif，如果不存在则尝试 .tiff
        img_path = os.path.join(images_dir, name + '.tif')
        if not os.path.exists(img_path):
            alt_img_path = os.path.join(images_dir, name + '.tiff')
            if os.path.exists(alt_img_path):
                img_path = alt_img_path
            else:
                return None
        
        # GSD: 先尝试去掉末尾 3 个字符，再尝试与标签同名（只改后缀）
        if len(name) > 3:
            gsd_basename = name[:-3]
        else:
            gsd_basename = name

        gsd_path = os.path.join(gsd_base_dir, gsd_basename + '.txt')
        if not os.path.exists(gsd_path):
            alt_gsd_path = os.path.join(gsd_base_dir, name + '.txt')
            if os.path.exists(alt_gsd_path):
                gsd_path = alt_gsd_path
            else:
                return None
        
        # 检查文件存在性
        if not os.path.exists(img_path):
            return None
        if not os.path.exists(gsd_path):
            return None
        
        # 如果需要验证数据有效性
        if validate:
            # 验证 ground truth
            gt = np.load(gt_path)
            if np.sum(gt) <= 100:
                return None
            
            # 简单验证图像（不完全加载，只检查文件可读）
            try:
                img = tifffile.imread(img_path)
                if img is None or img.size == 0:
                    return None
            except Exception:
                return None
        
        return (img_path, gt_path, gsd_path)
    
    except Exception as e:
        print(f"Error processing {gt_path}: {e}")
        return None


def process_longkou_dataset(gt_dir, validate=False):
    """
    处理 LongKou 数据集
    
    Args:
        gt_dir: ground truth 目录
        validate: 是否验证数据
    
    Returns:
        list: LongKou 样本列表
    """
    samples = []
    
    img_path = './Data/LongKou/WHU-Hi-LongKou.tif'
    gt_full_path = './Data/LongKou/WHU-Hi-LongKou_gt.tif'
    
    if not os.path.exists(img_path) or not os.path.exists(gt_full_path):
        print(f"LongKou dataset files not found")
        return samples
    
    try:
        print("Processing LongKou dataset...")
        gt = tifffile.imread(gt_full_path)
        gt_min = gt.min()
        gt_max = gt.max()
        
        for k in range(int(gt_min), int(gt_max) + 1):
            gt_in = (gt == k).astype(np.float32)
            components, labeled_array, num_components = extract_connected_components(gt_in)
            
            for component_idx, component in enumerate(components):
                if component.sum() <= 100:
                    continue
                # 存储为特殊格式: ('LongKou', img_path, gt_full_path, class_k, component_idx)
                samples.append({
                    'type': 'LongKou',
                    'img_path': img_path,
                    'gt_path': gt_full_path,
                    'class_k': int(k),
                    'component_idx': int(component_idx)
                })
        
        print(f"LongKou dataset: found {len(samples)} samples")
    
    except Exception as e:
        print(f"Error processing LongKou dataset: {e}")
    
    return samples


def iter_gt_files(gt_dir, use_system_find=True):
    """
    迭代 ground truth npy 文件，同时显示扫描进度
    """
    # 优先用系统 find（流式），避免一次性把结果全读进内存
    if use_system_find and os.name != 'nt':
        try:
            print("Using system 'find' (stream mode)...")
            proc = subprocess.Popen(
                ['find', gt_dir, '-type', 'f', '-name', '*.npy'],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1
            )
            with tqdm(desc="Scanning", unit="files") as pbar:
                for line in proc.stdout:
                    path = line.strip()
                    if not path:
                        continue
                    pbar.update(1)
                    yield path

            ret = proc.wait()
            if ret != 0:
                print(f"'find' exited with code {ret}, some files may be missing")
            return
        except Exception as e:
            print(f"System 'find' failed ({e}), falling back to Python glob...")

    # 回退到 Python 自己扫（支持子目录）
    gt_pattern = os.path.join(gt_dir, '**', '*.npy')
    for path in tqdm(glob.iglob(gt_pattern, recursive=True),
                     desc="Scanning", unit="files"):
        yield path


def prepare_dataset_index(images_dir, gt_dir, output_file,
                          gsd_base_dir="/data2/pl/HSITask/data/HyperFree/gsd",
                          data_name='HyperFree',
                          num_workers=8,
                          validate=False,
                          use_system_find=True):
    """
    生成数据集索引文件（改进版：大规模数据集下正常打印扫描 & 处理进度）
    """
    print(f"Starting dataset index preparation...")
    print(f"Images dir: {images_dir}")
    print(f"GT dir: {gt_dir}")
    print(f"GSD dir: {gsd_base_dir}")
    print(f"Workers: {num_workers}")
    print(f"Validate: {validate}")
    print(f"Data name: {data_name}")

    samples = []
    processed_files = set()

    # 临时文件（断点续传）
    temp_file = output_file + '.tmp'
    if os.path.exists(temp_file):
        try:
            print(f"\n✓ Found temporary file, resuming from previous run...")
            with open(temp_file, 'r') as f:
                temp_data = json.load(f)
                samples = temp_data.get('samples', [])
                processed_files = set(temp_data.get('processed_files', []))
            print(f"  Loaded {len(samples)} existing samples")
            print(f"  Skipping {len(processed_files)} already processed files")
        except Exception as e:
            print(f"  Warning: Failed to load temp file ({e}), starting fresh")
            samples = []
            processed_files = set()

    # 一些参数
    checkpoint_interval = 10000  # 每处理 1W 个样本存一次检查点
    max_futures_in_flight = num_workers * 4  # 限制同时挂在内存中的 future 数量

    valid_samples = []
    processed_in_this_run = 0
    submitted_in_this_run = 0
    total_scanned_files = 0

    print("\nStarting scan + processing...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor, \
            tqdm(desc="Processing", unit="files") as pbar:

        futures = set()

        def handle_finished_future(fut):
            nonlocal processed_in_this_run
            try:
                result = fut.result()
            except Exception as e:
                tqdm.write(f"Error in worker: {e}")
                return

            processed_in_this_run += 1
            pbar.update(1)

            if result is not None:
                img_path, gt_path, gsd_path = result
                valid_samples.append({
                    'type': 'npy',
                    'img_path': img_path,
                    'gt_path': gt_path,
                    'gsd_path': gsd_path
                })
                # 只记录有效样本对应的 gt，被你原来逻辑沿用
                processed_files.add(gt_path)

            # 定期保存检查点
            if processed_in_this_run % checkpoint_interval == 0:
                all_samples = samples + valid_samples
                temp_data = {
                    'samples': all_samples,
                    'processed_files': list(processed_files),
                    'checkpoint_time': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                try:
                    with open(temp_file, 'w') as f:
                        json.dump(temp_data, f)
                    tqdm.write(
                        f"✓ Checkpoint saved: {len(all_samples)} samples, "
                        f"{len(processed_files)} processed files"
                    )
                except Exception as e:
                    tqdm.write(f"⚠ Failed to save checkpoint: {e}")

        # 边扫描边提交任务，边处理结果
        for gt_path in iter_gt_files(gt_dir, use_system_find):
            total_scanned_files += 1

            # 跳过已处理文件
            if gt_path in processed_files:
                continue

            task_args = (gt_path, images_dir, gsd_base_dir, validate)
            fut = executor.submit(process_npy_file, task_args)
            futures.add(fut)
            submitted_in_this_run += 1

            # 控制挂起的 future 数量，防止一次性塞百万个 future 进内存
            if len(futures) >= max_futures_in_flight:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for d in done:
                    handle_finished_future(d)

        # 扫描结束，处理剩下的 future
        if futures:
            for fut in as_completed(futures):
                handle_finished_future(fut)

    print(f"\nScanned files: {total_scanned_files}")
    print(f"New files to process in this run: {submitted_in_this_run}")
    print(f"Valid samples in this run: {len(valid_samples)}/{submitted_in_this_run}")

    samples.extend(valid_samples)
    print(f"Total valid samples (including previous runs): {len(samples)}")

    # 如果你之后要支持 LongKou，这里保持原逻辑
    if data_name == 'LongKou':
        longkou_samples = process_longkou_dataset(gt_dir, validate)
        samples.extend(longkou_samples)

    # 保存最终 index
    print(f"\nSaving index to {output_file}...")
    output_data = {
        'images_dir': images_dir,
        'gt_dir': gt_dir,
        'gsd_base_dir': gsd_base_dir,
        'data_name': data_name,
        'total_samples': len(samples),
        'samples': samples
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    # 删除临时文件
    if os.path.exists(temp_file):
        try:
            os.remove(temp_file)
            print(f"✓ Temporary file removed")
        except Exception as e:
            print(f"⚠ Failed to remove temp file: {e}")

    print(f"\n✓ Index file created successfully!")
    print(f"Total samples: {len(samples)}")
    print(f"Output: {output_file}")

    return len(samples)
'''
python prepare_dataset_index.py \
    --images_dir "/data2/pl/HSITask/data/HyperFree/data_compressed" \
    --gt_dir "/data2/pl/HSITask/data/HyperFree/gt_compressed/ground_truth" \
    --output "../data/hyperfree_index.json" \
    --gsd_dir "/data2/pl/HSITask/data/HyperFree/gsd" \
    --data_name "HyperFree" \
    --workers 16

'''
def main():
    parser = argparse.ArgumentParser(description='Prepare dataset index for fast loading')
    parser.add_argument('--images_dir', type=str,
                        default="/data2/pl/HSITask/data/HyperFree/data_compressed",
                       help='Path to images directory')
    parser.add_argument('--gt_dir', type=str,
                        # default="/data2/pl/HSITask/data/HyperFree/labels_compressed/labels",
                        default="/data2/pl/HSITask/data/HyperFree/labels_hf_nms",
                       help='Path to ground truth directory')
    parser.add_argument('--gsd_dir', type=str, 
                       default='/data2/pl/HSITask/data/HyperFree/gsd',
                       help='Path to GSD files directory')
    parser.add_argument('--output', type=str,
                        # default="../data/hyperfree_index.json",
                        default='../data/hyperfree_index_labels_hf_nms.json',
                       help='Output index file path (JSON)')
    parser.add_argument('--data_name', type=str, default='HyperFree',
                       choices=['HyperFree'],
                       help='Dataset name')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of worker threads')
    parser.add_argument('--validate', action='store_true',
                       help='Validate data during indexing (slower but more accurate)')
    parser.add_argument('--use_system_find', action='store_true', default=True,
                       help='Use system find command for faster scanning (Linux/Unix only, default: True)')
    parser.add_argument('--no_system_find', dest='use_system_find', action='store_false',
                       help='Disable system find command, use Python glob instead')
    
    args = parser.parse_args()
    
    prepare_dataset_index(
        images_dir=args.images_dir,
        gt_dir=args.gt_dir,
        output_file=args.output,
        gsd_base_dir=args.gsd_dir,
        data_name=args.data_name,
        num_workers=args.workers,
        validate=args.validate,
        use_system_find=args.use_system_find
    )


if __name__ == '__main__':
    main()
