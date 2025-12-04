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
            pred_mask (torch.Tensor): Prediction mask tensor of shape (H, W), where each pixel value is in range [0, num_classes-1], where index {i} is {i + 1} in gt.That's because 
			we don't predict background.
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
        Compute the confusion matrix for multi-class segmentation
        
        Returns:
            torch.Tensor: Confusion matrix of shape (num_classes, num_classes)
        """
        # Initialize confusion matrix
        confusion_matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)
        
        # Compute confusion matrix
        for i in range(1, self.num_classes + 1):
            for j in range(1, self.num_classes + 1):
                confusion_matrix[i - 1, j - 1] = torch.sum(((self.gt == i) & (self.pred_mask == j)).int())
        print(confusion_matrix)
        self.confusion_matrix = confusion_matrix
        return confusion_matrix
    
    def compute_metrics(self):
        """
        Compute Overall Accuracy (OA), Average Accuracy (AA), and Kappa Coefficient (KA)
        
        Returns:
            tuple: (OA, AA, KA)
        """
        if self.confusion_matrix is None:
            self.compute_confusion_matrix()
            
        # Compute Overall Accuracy (OA)
        correct_predictions = torch.sum(torch.diag(self.confusion_matrix))
        total_pixels = torch.sum(self.confusion_matrix)
        self.OA = correct_predictions.float() / total_pixels.float()
        
        # Compute Average Accuracy (AA)
        class_accuracies = []
        for i in range(self.num_classes):
            if torch.sum(self.confusion_matrix[i, :]) > 0:
                class_acc = self.confusion_matrix[i, i].float() / torch.sum(self.confusion_matrix[i, :]).float()
                class_accuracies.append(class_acc)
                print(class_acc)
        self.AA = torch.tensor(class_accuracies).mean()
        
        # Compute Kappa Coefficient (KA)
        pe = 0.0
        for i in range(self.num_classes):
            row_sum = torch.sum(self.confusion_matrix[i, :])
            col_sum = torch.sum(self.confusion_matrix[:, i])
            pe += (row_sum * col_sum).float()
        pe = pe / (total_pixels * total_pixels).float()
        
        self.KA = (self.OA - pe) / (1 - pe)
        
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
	
    # def pair_pred_mask_index_with_gt(self):
    #     """
	# 	if pred mask's index is not same as gt's index, then change the pred mask's index to gt's index
	# 	only helpful when pred mask is much better than random mask.
	# 	"""
    #     confusion_matrix = self.compute_confusion_matrix()
    #     row_ind, col_ind = linear_sum_assignment(confusion_matrix.detach().numpy(), maximize=True)
    #     for i in range(self.num_classes):
    #         j = col_ind[i]
    #         if i != j:
    #             self.pred_mask[self.pred_mask == j] = i