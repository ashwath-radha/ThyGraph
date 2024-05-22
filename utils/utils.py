import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
import argparse
from typing import List
from sklearn.metrics import jaccard_score, roc_auc_score, average_precision_score, accuracy_score
from skimage.metrics import structural_similarity as ssim
import random
import os
import matplotlib.pyplot as plt
import math

def set_seed(args):
    #set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_opt(model: torch.nn.Module, args: argparse.ArgumentParser) -> torch.optim.Optimizer:
    print('Obtaining optimizer')

    # placeholder
    if args.opt == 'adam':
        print('Initialize adam optimizer')
        optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.wd)
    elif args.opt == 'sgd':
        print('Initialize sdg optimizer')
        optimizer = optim.SGD(model.parameters(), lr = args.lr, weight_decay = args.wd)
    elif args.opt == 'wang_sgd':
        print('Initialize wang_sdg optimizer')
        optimizer = optim.SGD(model.parameters(), lr= args.lr, weight_decay=args.wd)

    return optimizer

def compute_metrics(y_true: List[float], y_prob: List[float], thresh: float = 0.5): #-> tuple[float, float, float]:
    y_pred = (y_prob > thresh).astype(int)
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    acc = accuracy_score(y_true,y_pred)

    return acc, auroc, auprc

class EarlyStopping:
    def __init__(self, warmup: int = 10, tolerance: int = 20, metric: str = 'val_loss', verbose: bool = True) -> None:
        self.tolerance = tolerance
        self.verbose = verbose
        self.counter = 0
        self.best_metric = np.Inf if metric == 'val_loss' else 0.0
        self.early_stop = False
        self.warmup = warmup
        self.metric = metric

    def __call__(self, epoch: int, val_metric: float, model: torch.nn.Module, ckpt_path: str = 'checkpoint.pt') -> None:
        if epoch < self.warmup:
            pass
        elif self.metric == 'val_loss':
            if np.isinf(self.best_metric) or val_metric < self.best_metric:
                self.save_checkpoint(val_metric, model, ckpt_path)
                self.best_metric = val_metric
                self.counter = 0
            else:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.tolerance}')
                if self.counter >= self.tolerance:
                    self.early_stop = True
        elif self.metric in ['val_auc', 'val_acc']:
            if val_metric > self.best_metric:
                self.save_checkpoint(val_metric, model, ckpt_path)
                self.best_metric = val_metric
                self.counter = 0
            else:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.tolerance}')
                if self.counter >= self.tolerance:
                    self.early_stop = True

    def save_checkpoint(self, val_metric: float, model: torch.nn.Module, ckpt_path: str) -> None:
        if self.verbose:
            print(f'Validation {self.metric} improved from {self.best_metric:.6f} to {val_metric:.6f}. Model saved.')
        torch.save(model.state_dict(), ckpt_path)

def make_weights_for_balanced_classes(bagdataset, return_weights = False):
    all_labels = bagdataset.label_bags.label.values
    weight_per_class = [len(all_labels)/sum(all_labels==l) for l in [0,1]]    
    if return_weights:
        print('weights: ', torch.FloatTensor(weight_per_class))
        return torch.FloatTensor(weight_per_class)
    else:
        weights = [0] * len(all_labels)
        for idx in range(len(all_labels)):   
            y = all_labels[idx]    
            weights[idx] = weight_per_class[int(y)]            
          
        weights = torch.DoubleTensor(weights)                                   
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))          
        return sampler

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()

def initialize_gnn_weights(module):
    for name, param in module.named_parameters():
        if 'conv' in name and 'weight' in name:
            nn.init.xavier_normal_(param)
        elif 'conv' in name and 'bias' in name:
            nn.init.zeros_(param)

def calculate_iou(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    print('Calculating IOU')
    iou = jaccard_score(gt_mask, pred_mask, average="micro")
    return iou


def segment(imgs: torch.Tensor, teacher_masks: torch.Tensor) -> torch.Tensor:
    print('Performing segmentation')

    segmented_nodules = torch.zeros((256,256))

    return segmented_nodules


def save_heatmaps(output_path: str, pid: int, heatmap: torch.Tensor, visualization: torch.Tensor) -> None:
    print('Saving heatmap')

def save_masks(output_path: str, pid: int, masks: torch.Tensor) -> None:
    print('Saving heatmap')

# Calculate SSIM between all images given
def image_similarity(dataroot:str, img_names: str) -> np.ndarray:
    img_len = len(img_names)
    similarity_matrix = np.zeros((img_len, img_len))
    for i in range(img_len):
        img_i = np.load(os.path.join(dataroot, img_names[i]))
        for j in range(i, img_len):
            img_j = np.load(os.path.join(dataroot, img_names[j]))
            img_i, img_j = crop_images(img_i, img_j)
            i_min, i_max, j_min, j_max = np.min(img_i), np.max(img_i), np.min(img_j), np.max(img_j)
            img_i = (img_i - i_min) / (i_max - i_min)
            img_j = (img_j - j_min) / (j_max - j_min)
            assert np.max(img_i)-np.min(img_i)==1.0
            assert np.max(img_j)-np.min(img_j)==1.0
            similarity_score = ssim(img_i, img_j, data_range=1.0) # calculate SSIM, clearly define data range!
            similarity_matrix[i, j] = similarity_score
            similarity_matrix[j, i] = similarity_score
            
    return similarity_matrix

# Crop images from the center out so they can be compared
# def crop_images(img_i: np.ndarray, img_j: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
from typing import Tuple
def crop_images(img_i: np.ndarray, img_j: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    height1, width1 = img_i.shape
    height2, width2 = img_j.shape

    # Find the smaller height and width
    min_height = min(height1, height2)
    min_width = min(width1, width2)

    # Calculate the amount to be cropped from each side
    crop_top = (height1 - min_height) // 2
    crop_bottom = crop_top + min_height
    crop_left = (width1 - min_width) // 2
    crop_right = crop_left + min_width

    # Crop the images to keep the center region with the same dimensions
    pos1, pos2, pos3, pos4 = True, True, True, True
    if min_height==height1:
        img_i, pos1 = img_i[0:min_height, :], False
    if min_height==height2:
        img_j, pos3 = img_j[0:min_height, :], False
    if min_width==width1:       
        img_i, pos2 = img_i[:, 0:min_width], False
    if min_width==width2:
        img_j, pos4 = img_j[:, 0:min_width], False
    if pos1:
        img_i = img_i[crop_top:crop_bottom, :]
    if pos2:
        img_i = img_i[:, crop_left:crop_right]
    if pos3:
        img_j = img_j[crop_top:crop_bottom, :]
    if pos4:
        img_j = img_j[:, crop_left:crop_right]
    
    return img_i, img_j
