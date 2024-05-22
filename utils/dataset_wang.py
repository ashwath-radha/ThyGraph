import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
from typing import Tuple, List
import pandas as pd
import os
import numpy as np
import random
import albumentations as A
from skimage.transform import resize
import math
import sys

sys.path.append('/raid/aradhachandran/thyroid/code/classification/BE223A_thyroidProj/')
from utils.extract_labels import convert_to_string
from utils.custom_preprocessing import *

def collate_features_wang(batch, squiggly_n):
    mrn, bag_imgs, label = zip(*batch)
    bag_imgs = np.array(bag_imgs)
    bags = torch.tensor(bag_imgs)
    if len(bags.size()) == 6 and bags.size(0) == 1:
        bags = bags.squeeze(0)
    labels = torch.tensor(label)
    
    return mrn, bags, labels

class WangDataset(Dataset):
    def __init__(
        self,
        dataroot: str,
        csv_file: str,
        splitroot: str,
        split: int,
        mode: str = "train",
        stanford: bool = False,
        squiggly_n: int = 20
    ) -> None:
        print(f"Creating {mode} WangDataset for MIL pipeline")

        # load labels
        label_bags = pd.read_csv(csv_file, converters={"mrn": convert_to_string}, index_col=0)
        self.reference_bags = pd.read_csv(csv_file, converters={"mrn": convert_to_string},
                                          usecols=['mrn','img_name'])
        self.mode = mode
        label_bags = label_bags.drop_duplicates(subset="mrn", keep="first").reset_index(drop=True)
        label_bags["label"] = label_bags.cytology.map({"malignant": 1, "benign": 0})
        label_bags = label_bags.dropna(subset=["label"])

        # load split
        if stanford:
            splits = pd.read_csv(
                os.path.join(splitroot, "cv_stanford.csv"),
                converters={"holdout": convert_to_string}
            )
        else:
            splits = pd.read_csv(
                os.path.join(splitroot, f"cv{split}.csv"),
                converters={"train": convert_to_string, "test": convert_to_string}
            )
        if self.mode == "train":
            mrns = splits.train.values
        elif self.mode == "test":
            mrns = splits.test.values
        elif self.mode == "holdout":
            mrns = [str(i) for i in splits.holdout.values]
            mrns = ['0'+i if len(i) == 6 else i for i in mrns]
        self.label_bags = label_bags[label_bags.mrn.isin(mrns)].reset_index(drop=True)

        self.mrns = self.label_bags.mrn.unique()
        self.dataroot = dataroot
        
        self.ind_list, self.seed_list = np.zeros((len(self.mrns),), dtype=int), np.zeros((len(self.mrns),), dtype=int)
        self.squiggly_n = squiggly_n
        self.resize_dim = 299
        self.transforms = transforms.Compose([
            SquarePad(),
            Resize(self.resize_dim),
            GrayscaletoRGB(),
            # transforms.ToTensor()
            ])
        self.image_aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=(-30, 30), p=0.5)
            ])

    def __len__(self) -> int:
        return len(self.mrns)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, torch.Tensor]:
        mrn = self.mrns[idx]
        label = self.label_bags[self.label_bags.mrn == mrn].label.item()
        mrn_bag = self.reference_bags[self.reference_bags.mrn == mrn]['img_name'].values
        
        if self.mode=='train':
            bag_imgs = self.train_logic(mrn_bag, idx)
        if self.mode in ['test', 'holdout']:
            # bag_imgs = self.train_logic(mrn_bag, idx)
            bag_imgs = self.validation_logic(mrn_bag, idx)
        
        return mrn, bag_imgs, label

    def train_logic(self, mrn_bag, idx):
        n, shuffle_seed = len(mrn_bag), self.seed_list[idx]
        np.random.seed(shuffle_seed)
        np.random.shuffle(mrn_bag)
        
        if n > self.squiggly_n:
            bag_subset = mrn_bag[self.ind_list[idx]*self.squiggly_n:self.ind_list[idx]*self.squiggly_n + self.squiggly_n]
            if len(bag_subset) < self.squiggly_n:
                remaining_length = self.squiggly_n - len(bag_subset)
                bag_subset = np.concatenate((bag_subset, mrn_bag[:remaining_length]))
                self.ind_list[idx] = 0
                self.seed_list[idx] += 1
            else:
                self.ind_list[idx] += 1
        else:
            bag_subset = mrn_bag[:self.squiggly_n]
            bag_subset = np.tile(bag_subset, (self.squiggly_n + len(bag_subset) - 1) // len(bag_subset))[:self.squiggly_n]

        if self.ind_list[idx]*self.squiggly_n==n: # TODO MAKE SURE THIS IS RIGHT
            self.ind_list[idx] = 0
            self.seed_list[idx] += 1
        assert len(bag_subset)==self.squiggly_n
        
        bag_imgs = self.get_images(bag_subset)

        return bag_imgs

    def validation_logic(self, mrn_bag, idx):
        n, shuffle_seed = len(mrn_bag), self.seed_list[idx]
        np.random.seed(shuffle_seed)
        np.random.shuffle(mrn_bag)

        patient_all_bags = []
        if n > self.squiggly_n:
            num_patient_bags = math.ceil(len(mrn_bag)/self.squiggly_n)
            patient_all_bags = np.empty((num_patient_bags, self.squiggly_n, 3, self.resize_dim, self.resize_dim))
            for i in range(num_patient_bags):
                bag_subset = mrn_bag[i*self.squiggly_n : i*self.squiggly_n + self.squiggly_n]
                if len(bag_subset) < self.squiggly_n:
                    remaining_length = self.squiggly_n - len(bag_subset)
                    bag_subset = np.concatenate((bag_subset, mrn_bag[:remaining_length]))
                bag_imgs = self.get_images(bag_subset)
                patient_all_bags[i] = bag_imgs
        else:
            patient_all_bags = np.empty((1, self.squiggly_n, 3, self.resize_dim, self.resize_dim))
            bag_subset = mrn_bag[:self.squiggly_n]
            bag_subset = np.tile(bag_subset, (self.squiggly_n + len(bag_subset) - 1) // len(bag_subset))[:self.squiggly_n]
            bag_imgs = self.get_images(bag_subset)
            patient_all_bags[0] = bag_imgs

        return patient_all_bags #np.stack(patient_all_bags, axis=0) #np.array(patient_all_bags)
    
    def get_images(self, bag_subset):
        bag_imgs = np.empty((len(bag_subset), self.resize_dim, self.resize_dim, 3))
        for bag_idx, img_name in enumerate(bag_subset):
            img = self.norm_image(img_name)
            img = self.transforms(img)
            if self.mode=='train':
                img = self.image_aug(image=img)['image'] # not include aug in validation
            bag_imgs[bag_idx] = img #np.swapaxes(img, 0, 2)
        bag_imgs = np.swapaxes(bag_imgs, 1, 3)
        return bag_imgs

    def norm_image(self, img_name):
        img = np.load(os.path.join(self.dataroot, img_name))
        # img = resize(img, (299, 299), order=1)
        max_val = img.max()
        min_val = img.min()
        img = (img - min_val) / (max_val - min_val)

        return img