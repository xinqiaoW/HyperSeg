import glob
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Tuple, Optional
from scipy.optimize import linear_sum_assignment


class CollectData:
	def __init__(self, gt, pred_score, pred_mask=None, threshold=0.0):
		"""
		:param gt:  tensor, groundtruth tensor,shape (N, H, W)
		:param pred_score: tensor , pred_score tensor 
		:param pred_mask: tensor, pred_mask tensor
		:param data_type: string, type of data, 'nparray' or 'torch.tensor'
		"""
		self.TP = []
		self.FP = []
		self.FN = []
		self.TN = []
		self.IoU = []
		self.Precision = []
		self.Recall = []
		self.F1 = []
		self.groundtruth = gt
		self.pred_score = pred_score
		if pred_mask is not None:
			self.pred_mask = pred_mask
		else:
			self.pred_mask = (pred_score > threshold).int()
	
	def compute_key_values(self):
		"""
		Calculate the IoU score
		:return: IoU score, list
		"""
		for i in range(self.groundtruth.shape[0]):
			self.TP.append(torch.sum(self.groundtruth[i] * self.pred_mask[i]))
			self.FP.append(torch.sum((1 - self.groundtruth[i]) * self.pred_mask[i]))
			self.FN.append(torch.sum(self.groundtruth[i] * (1 - self.pred_mask[i])))
			self.TN.append(torch.sum((1 - self.groundtruth[i]) * (1 - self.pred_mask[i])))

	def compute_iou(self):
		"""
		Calculate the IoU score
		:return: IoU score, list(element isn't included when U of IoU is zero)
		"""
		if len(self.TP) == 0:
			self.compute_key_values()
		for i in range(self.groundtruth.shape[0]):
			union = (self.TP[i] + self.FP[i] + self.FN[i])
			if union == 0:
				continue
			self.IoU.append(self.TP[i] / union)
	
	def compute_precision(self):
		"""
		Calculate the precision score
		:return: precision score, list
		"""
		if len(self.TP) == 0:
			self.compute_key_values()
		for i in range(self.groundtruth.shape[0]):
			if self.TP[i] + self.FP[i] == 0:
				continue
			self.Precision.append(self.TP[i] / (self.TP[i] + self.FP[i]))
	
	def compute_recall(self):
		"""
		Calculate the recall score
		:return: recall score, list
		"""
		if len(self.TP) == 0:
			self.compute_key_values()
		for i in range(self.groundtruth.shape[0]):
			if self.TP[i] + self.FN[i] == 0:
				continue
			self.Recall.append(self.TP[i] / (self.TP[i] + self.FN[i]))

	def compute_f1(self):
		"""
		Calculate the F1 score
		:return: F1 score, list
		"""
		if len(self.TP) == 0:
			self.compute_key_values()
		for i in range(self.groundtruth.shape[0]):
			if self.TP[i] + self.FP[i] + self.FN[i] == 0:
				continue
			self.F1.append(2 * self.TP[i] / (2 * self.TP[i] + self.FP[i] + self.FN[i]))
	
	def compute_all(self):
		"""
		Calculate all the scores
		:return: IoU score, precision score, recall score, F1 score
		"""
		self.compute_key_values()
		self.compute_iou()
		self.compute_precision()
		self.compute_recall()
		self.compute_f1()


import torch
import numpy as np


class SegMetric:
    def __init__(self, gt, pred_mask, num_classes):
        """
        Initialize the segmentation metrics calculator

        Args:
            gt (torch.Tensor): Ground truth tensor of shape (H, W), where each pixel value is in range [0, num_classes], 0 - background
            pred_mask (torch.Tensor): Prediction mask tensor of shape (H, W), where each pixel value is in range [0, num_classes]
            num_classes (int): Number of classes (doesn't include background)
        """
        self.gt = gt
        self.pred_mask = pred_mask
        self.num_classes = num_classes
        self.confusion_matrix = None
        self.OA = None  # Overall Accuracy
        self.AA = None  # Average Accuracy
        self.KA = None  # Kappa Coefficient


    def compute_confusion_matrix(self):
        """
        Compute the confusion matrix for multi-class segmentation.
        Only considers pixels where GT is foreground (gt != 0).

        Returns:
            torch.Tensor: Confusion matrix of shape (num_classes + 1, num_classes + 1)
                          Row 0 will be all zeros (GT background is masked out).
                          Column 0 captures predictions of background for foreground GT pixels (misses).
        """
        # Move to CPU and flatten tensors
        gt_flat = self.gt.detach().cpu().flatten().long()
        pred_flat = self.pred_mask.detach().cpu().flatten().long()

        # Only consider pixels where GT is foreground
        valid_mask = (gt_flat != 0)
        gt_valid = gt_flat[valid_mask]
        pred_valid = pred_flat[valid_mask]

        # Compute confusion matrix using bincount (fast)
        n = self.num_classes + 1
        idxs = gt_valid * n + pred_valid
        confusion_matrix = torch.bincount(idxs, minlength=n**2).reshape(n, n).float()

        self.confusion_matrix = confusion_matrix
        return confusion_matrix

    def compute_metrics(self):
        """
        Compute Overall Accuracy (OA), Average Accuracy (AA), and Kappa Coefficient (KA).
        Only uses foreground classes (rows 1 to num_classes) for metric computation.

        Returns:
            tuple: (OA, AA, KA)
        """
        if self.confusion_matrix is None:
            self.compute_confusion_matrix()

        # Extract foreground part: rows 1 to num_classes
        # Row 0 is all zeros (GT background masked out)
        # Column 0 captures misses (pred=0 for foreground GT)

        # TP: diagonal elements for classes 1 to num_classes
        tp = torch.diag(self.confusion_matrix)[1:]

        # GT total per class: row sums for classes 1 to num_classes
        # This includes column 0 (misses), so denominator is correct
        gt_total = self.confusion_matrix.sum(dim=1)[1:]

        # Pred total per class: column sums for classes 1 to num_classes
        pred_total = self.confusion_matrix.sum(dim=0)[1:]

        # Total foreground pixels
        total_pixels = gt_total.sum()

        # Overall Accuracy (OA): correct predictions / total foreground pixels
        self.OA = tp.sum() / (total_pixels + 1e-6)

        # Average Accuracy (AA): mean of per-class recall
        recall = tp / (gt_total + 1e-6)
        self.AA = recall.mean()

        # Kappa Coefficient (KA)
        # Pe calculation uses only foreground classes (1 to num_classes)
        col_total = self.confusion_matrix.sum(dim=0)[1:]
        pe = torch.sum(gt_total * col_total) / (total_pixels ** 2 + 1e-6)
        self.KA = (self.OA - pe) / (1 - pe + 1e-6)

        return self.OA, self.AA, self.KA

    def get_metrics(self):
        """
        Get all computed metrics

        Returns:
            dict: Dictionary containing OA, AA, KA, and confusion matrix
        """
        if self.OA is None:
            self.compute_metrics()

        return {
            'OA': self.OA.item(),
            'AA': self.AA.item(),
            'KA': self.KA.item(),
            'confusion_matrix': self.confusion_matrix
        }