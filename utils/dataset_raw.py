# defines the objects for loading the original ultrasound images

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import pandas as pd
import numpy as np
from typing import Tuple, Union, List
import os

from utils.custom_preprocessing import *
from utils.extract_labels import convert_to_string


class ImageDataset(Dataset):
    def __init__(
        self,
        dataroot: str,
        csv_file: str,
        patient: List[str],
        resize_dims: int,
        segment: bool = False,
        segmentor: str = None
    ) -> None:
        print("Creating dataset for feature-extraction pipeline")

        # path to the images
        self.dataroot = dataroot

        # filter the dataset to only the images for the patient of interest
        df = pd.read_csv(csv_file, converters={"mrn": convert_to_string})
        df = df[df["mrn"].isin(patient)]

        # dataset comprises of the images/masks for patient(s) of interest
        self.images = df["img_name"].to_list()
        self.masks = df["mask_name"].to_list()
        self.resize_dims = resize_dims

        if "FLIP" in df.columns:
            self.flips = df["FLIP"].to_list()
            self.rotations = df["ROTATION"].to_list()
        else:
            self.flips = [False] * len(self.images)
            self.rotations = [0] * len(self.images)

        self.pid = df["mrn"].to_list()

        self.segment = segment
        self.segmentor = segmentor

    def set_transform(self, flip, rotate):
        if self.segment:
            if self.segmentor == "tnsui":
                # tnsui expects a 256x256 image with 1 channel
                self.transforms = transforms.Compose(
                    [
                        SquarePad(),
                        Resize(256),
                        transforms.ToTensor(),
                        FixedFlip(flip),
                        FixedRotation(rotate),
                        # transforms.Normalize(mean=[0.5], std=[0.5])
                    ]
                )
                self.mask_transform = transforms.Compose(
                    [
                        SquarePad(),
                        Resize(256),
                        transforms.ToTensor(),
                        FixedFlip(flip),
                        FixedRotation(rotate),
                    ]
                )
            elif segmentor == "gongseg":
                # gongseg expects a 224x224 image with 3 channels
                self.transforms = transforms.Compose(
                    [
                        SquarePad(),
                        Resize(224),
                        GrayscaletoRGB(),
                        transforms.ToTensor(),
                        FixedFlip(flip),
                        FixedRotation(rotate),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
                self.mask_transform = transforms.Compose(
                    [
                        SquarePad(),
                        Resize(224),
                        transforms.ToTensor(),
                        FixedFlip(flip),
                        FixedRotation(rotate),
                    ]
                )

        else:
            # resnet expects a 224x224 image with 3 channels
            self.transforms = transforms.Compose(
                [
                    SquarePad(),
                    Resize(self.resize_dims),
                    GrayscaletoRGB(),
                    transforms.ToTensor(),
                    FixedFlip(flip),
                    FixedRotation(rotate),
                    # transforms.Normalize(
                    #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    # ), # REMOVED THIS FROM ORIGINAL AMIA CODE, CONSIDER NORMALIZING WITH TRAIN SET ULTRASOUNDS
                ]
            )
            self.mask_transform = transforms.Compose(
                [
                    SquarePad(),
                    Resize(self.resize_dims),
                    transforms.ToTensor(),
                    FixedFlip(flip),
                    FixedRotation(rotate),
                ]
            )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, torch.Tensor, str]:
        # get the paths to the images
        image_pth = os.path.join(self.dataroot, self.images[idx])
        pid = str(self.pid[idx])
        image_name = os.path.splitext(os.path.basename(image_pth))[0]

        # load image
        image = np.load(image_pth)

        # load the mask if it exists, otherwise create a full mask
        mask_pth = self.masks[idx]
        if mask_pth is None or mask_pth != mask_pth:
            mask = np.ones(image.shape)
        else:
            mask_pth = os.path.join(self.dataroot, mask_pth)
            mask = np.load(mask_pth)

        # preprocessing
        # image, mask = full_preprocess(image, mask)

        # apply transformations if necessary (transform to tensor)
        self.set_transform(self.flips[idx], self.rotations[idx])
        image = self.transforms(image)
        mask = self.mask_transform(mask)

        # threshold mask after transformation since resizing using bilinear interpolation
        mask = mask > 0
        return pid, image, mask, image_name


# wrapper for loading all the images for patients in a given split, one patient at a time
class ImageDatasetFromSplit:
    def __init__(
        self, dataroot: str, csv_file: str, splitroot: str, split: int, mode: str
    ) -> None:
        print("Creating dataset for evaluation")

        # get the patients of interest
        split_df = pd.read_csv(os.path.join(splitroot, f"cv{split}.csv"))
        self.patient_list = split_df.dropna()[mode].to_list() # TODO change this to "holdout"

        self.csv_file = csv_file
        self.dataroot = dataroot

    def __len__(self) -> int:
        return len(self.patient_list)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, str]:
        patient = [self.patient_list[idx]]

        # create a dataset that contains all the images for a given patient
        patient_dataset = ImageDataset(self.dataroot, self.csv_file, patient)

        # load ALL the images in one batch
        patient_loader = DataLoader(patient_dataset, batch_size=len(patient_dataset))
        pid, images, _, images_name = next(iter(patient_loader))

        return pid, images, images_name
