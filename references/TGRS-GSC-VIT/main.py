import argparse
import numpy as np
import torch.nn as nn
import torch.utils.data
from torchsummaryX import summary
from utils.dataset import load_mat_hsi, sample_gt, HSIDataset
from utils.utils import split_info_print, metrics, show_results
from utils.scheduler import load_scheduler
from models.get_model import get_model
from train import train, test
from utils.utils import Draw


def sample_fixed_per_class(gt, train_samples=5, val_samples=5, seed=None):
    """
    每类固定数量样本采样
    Args:
        gt: 原始标签图
        train_samples: 每类训练样本数
        val_samples: 每类验证样本数
        seed: 随机种子
    Returns:
        train_gt: 训练集标签图
        val_gt: 验证集标签图  
        test_gt: 测试集标签图（剩余所有样本）
    """
    if seed is not None:
        np.random.seed(seed)
    
    train_gt = np.zeros_like(gt) - 1 
    val_gt = np.zeros_like(gt) - 1
    test_gt = np.zeros_like(gt) - 1
    
    # 获取所有类别（跳过背景类0）
    classes = np.unique(gt)
    classes = classes[classes != -1]
 
    
    for c in classes:
        # 获取当前类别的所有像素位置
        positions = np.column_stack(np.where(gt == c))
        if len(positions) == 0:
            continue
            
        # 随机打乱
        np.random.shuffle(positions)
        
        # 分配训练样本
        if len(positions) >= train_samples:
            for i in range(train_samples):
                row, col = positions[i]
                train_gt[row, col] = c
                
            # 分配验证样本
            if len(positions) >= train_samples + val_samples:
                for i in range(train_samples, train_samples + val_samples):
                    row, col = positions[i]
                    val_gt[row, col] = c
                    
                # 剩余作为测试样本
                for i in range(train_samples + val_samples, len(positions)):
                    row, col = positions[i]
                    test_gt[row, col] = c
            else:
                # 如果样本不够，将剩余的全部作为验证样本
                for i in range(train_samples, len(positions)):
                    row, col = positions[i]
                    val_gt[row, col] = c
        else:
            # 如果样本少于训练样本数，全部作为训练样本
            for i in range(len(positions)):
                row, col = positions[i]
                train_gt[row, col] = c
    
    return train_gt, val_gt, test_gt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run patch-based HSI classification")
    parser.add_argument("--model", type=str, default='gscvit') # model name
    parser.add_argument("--dataset_name", type=str, default="sa") # dataset name
    parser.add_argument("--dataset_dir", type=str, default="./datasets") # dataset dir
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--patch_size", type=int, default=8) # patch_size
    parser.add_argument("--num_run", type=int, default=10)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--bs", type=int, default=128)  # bs = batch size
    parser.add_argument("--ratio", type=float, default=0.02) # ratio of training + validation sample

    opts = parser.parse_args()

    device = torch.device("cuda:{}".format(opts.device))

    # print parameters
    print("experiments will run on GPU device {}".format(opts.device))
    print("model = {}".format(opts.model))
    print("dataset = {}".format(opts.dataset_name))
    print("dataset folder = {}".format(opts.dataset_dir))
    print("patch size = {}".format(opts.patch_size))
    print("batch size = {}".format(opts.bs))
    print("total epoch = {}".format(opts.epoch))
    print("{} for training, {} for validation and {} testing".format(opts.ratio / 2, opts.ratio / 2, 1 - opts.ratio))

    # load data
    image, gt, labels = load_mat_hsi(opts.dataset_name, opts.dataset_dir)

    num_classes = len(labels)
    num_bands = image.shape[-1]

    # random seeds
    seeds = [202401, 202402, 202403, 202404, 202405, 202406, 202407, 202408, 202409, 202410]

    # empty list to storing results
    results = []

    for run in range(opts.num_run):
        np.random.seed(seeds[run])
        print("running an experiment with the {} model".format(opts.model))
        print("run {} / {}".format(run + 1, opts.num_run))

        # 使用每类固定5个样本的策略
        train_gt, val_gt, test_gt = sample_fixed_per_class(gt, train_samples=5, val_samples=5, seed=seeds[run])
        
        # 如果需要保持原有的trainval_gt结构，可以这样合并
        trainval_gt = train_gt.copy()
        trainval_gt[val_gt > 0] = val_gt[val_gt > 0]

        train_set = HSIDataset(image, train_gt, patch_size=opts.patch_size, data_aug=True)
        val_set = HSIDataset(image, val_gt, patch_size=opts.patch_size, data_aug=False)

        train_loader = torch.utils.data.DataLoader(train_set, opts.bs, drop_last=False, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_set, opts.bs, drop_last=False, shuffle=False)

        # load model and loss
        model = get_model(opts.model, opts.dataset_name, opts.patch_size)

        if run == 0:
            split_info_print(train_gt, val_gt, test_gt, labels)
            print("network information:")
            # with torch.no_grad():
            #     summary(model, torch.zeros((1, 1, num_bands, opts.patch_size, opts.patch_size)))

        model = model.to(device)
        # print(model)
        optimizer, scheduler = load_scheduler(opts.model, model)

        criterion = nn.CrossEntropyLoss()

        # where to save checkpoint model
        model_dir = "./checkpoints/" + opts.model + '/' + opts.dataset_name + '/' + str(run)

        try:
            train(model, optimizer, criterion, train_loader, val_loader, opts.epoch, model_dir, device, scheduler)
        except KeyboardInterrupt:
            print('"ctrl+c" is pused, the training is over')

        # test the model
        probabilities = test(model, model_dir, image, opts.patch_size, num_classes, device)

        prediction = np.argmax(probabilities, axis=-1)

        # computing metrics
        run_results = metrics(prediction, test_gt, n_classes=num_classes)  # only for test set
        results.append(run_results)
        show_results(run_results, label_values=labels)

        # draw the classification map
        Draw(model,image,gt,opts.patch_size,opts.dataset_name,opts.model,num_classes)

        del model, train_set, train_loader, val_set, val_loader

    if opts.num_run > 1:
        show_results(results, label_values=labels, agregated=True)