# defines the objects for loading the original ultrasound images
import sys
import os

sys.path.insert(0, os.path.join(os.getcwd()))

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import pandas as pd
import numpy as np
from typing import Tuple, Union, List
import os
from PIL import Image
import cv2
from tqdm import tqdm
import h5py

from skimage.util.shape import view_as_windows
from sklearn.preprocessing import StandardScaler
from utils.dataset_bags import (
    format_augmented_mrn,
    filter_augmentations,
    collate_features,
)
from utils.extract_labels import convert_to_string

# main class for combining all the patches of one patient (across all frames) into one big bag


class PatchBagDataset(Dataset):
    def __init__(
        self,
        dataroot: str,
        featuresroot: str,
        csv_file: str,
        splitroot: str,
        split: int,
        mode: str = "train",
        tile_size: list = [32],
        stride_size: list = [32],
        segment: bool = False,
        max_augments: int = 3,
        include_tirads: bool = False,
        need_patch_nodule_overlap: bool = False,
        stanford: bool = False
    ) -> None:
        print(f"Creating {mode} dataset for MIL pipeline")
        
        self.stanford = stanford
        # load labels
        label_bags = pd.read_csv(csv_file, converters={"mrn": convert_to_string}, index_col=0)
        if not self.stanford:
            label_bags["mrn"] = format_augmented_mrn(label_bags.mrn)

        label_bags = label_bags.drop_duplicates(subset="mrn", keep="first").reset_index(drop=True)
        label_bags = filter_augmentations(label_bags, max_augments)
        label_bags["mrn"] = label_bags.mrn.astype(str)
        label_bags["label"] = label_bags.cytology.map({"malignant": 1, "benign": 0})
        label_bags = label_bags.dropna(subset=["label"])

        # load split
        if self.stanford:
            splits = pd.read_csv(
                os.path.join(splitroot, "cv_stanford.csv"),
                converters={"holdout": convert_to_string}
            )
        else:
            splits = pd.read_csv(
                os.path.join(splitroot, f"cv{split}.csv"),
                converters={"train": convert_to_string, "test": convert_to_string},
            )
        if mode == "train":
            mrns = splits.train.values
            self.label_bags = label_bags[label_bags.mrn.isin(mrns)].reset_index(drop=True)
        elif mode == "test":
            mrns = splits.test.values
            self.label_bags = label_bags[label_bags.mrn.isin(mrns)].reset_index(drop=True)
        elif mode == "combined":
            train_mrns = splits.train.values
            test_mrns = [i.split(".")[0] for i in splits.test.values if i]
            mrns = np.concatenate((train_mrns, test_mrns))

            self.label_bags = label_bags[
                label_bags.mrn.str.contains("|".join(mrns))
            ].reset_index(drop=True)

        elif mode == "holdout":
            mrns = splits.holdout.values
            mrns = [str(i).split(".")[0] for i in splits.holdout.values if i]
            mrns = ['0'+i if len(i) == 6 else i for i in mrns ]
            self.label_bags = label_bags[label_bags.mrn.isin(mrns)].reset_index(drop=True)

        self.mrns = self.label_bags.mrn.unique()

        self.dataroot = dataroot
        self.featuresroot = featuresroot
        self.csv_file = csv_file
        self.segment = segment
        self.tile_size = tile_size
        self.stride_size = stride_size
        if include_tirads:
            clinical_features = ['sex','age','TI-RAD']
        else:
            clinical_features = ['sex','age']
        self.clinical_features = clinical_features

        if not self.stanford:
            self.clean_clinical(label_bags, splits, clinical_features)
        self.need_patch_nodule_overlap = need_patch_nodule_overlap

    def __len__(self) -> int:
        return len(self.label_bags)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, torch.Tensor]:
        mrn = self.mrns[idx]
        label = self.label_bags[self.label_bags.mrn == mrn].label.item()
        clinical = self.label_bags[self.label_bags.mrn == mrn][self.clinical_features].values

        #clinical = self.label_bags[self.label_bags.mrn == mrn][["age", "sex",'TI-RAD']].values

        patch_features = []

        for i in range(len(self.tile_size)):
            bag_creator = PatientPatchDataset(
                self.dataroot,
                self.csv_file,
                [mrn],
                segment=self.segment,
                tile_size=self.tile_size[i],
                stride_size=self.stride_size[i],
                need_patch_nodule_overlap=self.need_patch_nodule_overlap,
                stanford=self.stanford
            )
            patch_feature = bag_creator.create_bag()
            patch_features.append(patch_feature)
        # if patch or patch_combine model
        if self.featuresroot is None: 
            patch_features = [torch.cat(list(patch_feature.values())) for patch_feature in patch_features]
            return mrn, None, patch_features, None, clinical, label
        else:
            #if frame+patch combine model
            # load h5 or pt file
            if len(patch_features) > 1:
                raise NotImplementedError('Frame + Patch model can only takes in one patch_size')
            patch_features = patch_features[0]
            bag_path = os.path.join(self.featuresroot, str(mrn) + ".h5")
            with h5py.File(bag_path, "r") as f:
                frame_features = f["features"]
                img_names = f["img_names"]
                frame_features = torch.tensor(np.array(frame_features))
                img_names = [i.decode("utf-8") for i in np.array(img_names)]

            patch_features_sorted = [
                patch_features[k] for k in img_names if k in patch_features
            ]
            patch_lens = [len(p) for p in patch_features_sorted]
            patch_features = torch.cat(patch_features_sorted)

            return mrn, frame_features, patch_features, patch_lens, clinical, label

    def clean_clinical(self, label_bags, splits, clinical_features):
        # binarize sex
        self.label_bags["sex"] = self.label_bags["sex"].map({"M": 0, "F": 1})

        # fit by train age and transform the self.label_bags age
        mrns = splits.train.values
        #train_label_bags = label_bags[label_bags.mrn.str.contains("|".join(mrns))][["age",'TI-RAD']]
        #scaler = StandardScaler().fit(train_label_bags)
        #self.label_bags[["age",'TI-RAD']] = scaler.transform(self.label_bags[["age",'TI-RAD']])
        clinical_features = clinical_features[1:]
        train_label_bags = label_bags[label_bags.mrn.str.contains("|".join(mrns))][clinical_features]
        scaler = StandardScaler().fit(train_label_bags)
        self.label_bags[clinical_features] = scaler.transform(self.label_bags[clinical_features])


# wrapper for loading all the images (WITHOUT combining them into a bag) for patients in a given split, one patient at a time
class PatchDatasetFromSplit:
    def __init__(
        self,
        dataroot: str,
        csv_file: str,
        splitroot: str,
        split: int,
        mode: str,
        tile_size: int = 32,
        stride_size: int = 32,
    ) -> None:
        print("Creating dataset for evaluation")


       # load labels
        label_bags = pd.read_csv(csv_file)#, converters={"mrn": str})
        label_bags['mrn'] = label_bags['mrn'].apply(convert_to_string)
        self.original_csv = label_bags

        label_bags = label_bags.drop_duplicates(subset="mrn", keep="first").reset_index(
            drop=True
        )
        label_bags["mrn"] = label_bags.mrn.astype(str)
        label_bags["label"] = label_bags.cytology.map({"malignant": 1, "benign": 0})
        label_bags = label_bags.dropna(subset=["label"])
        self.label_bags = label_bags

        # get the patients of interest
        split_df = pd.read_csv(
            os.path.join(splitroot, f"cv{split}.csv"),
            converters={"train": convert_to_string, "test": convert_to_string},
        )

        if mode == "train": self.patient_list = [i for i in split_df.train.values if i]
        elif mode == "test": self.patient_list = [i for i in split_df.test.values if i]
        elif mode == "holdout": self.patient_list = [i for i in split_df.holdout.values if i]

        self.csv_file = csv_file
        self.dataroot = dataroot
        self.split_df = split_df
        self.tile_size = tile_size
        self.stride_size = stride_size
        self.split = split
        self.clean_clinical()

    def __len__(self) -> int:
        return len(self.patient_list)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, str]:
        patient = [self.patient_list[idx]]
        
        image_patches = []
        image_coords = []
        images_name = []
        image_shapes = []

        for i in range(len(self.tile_size)):
            # create a dataset that contains all the images for a given patient
            patient_dataset = PatientPatchDataset(
                self.dataroot,
                self.csv_file,
                patient,
                tile_size=self.tile_size[i],
                stride_size=self.stride_size[i],
            )

            # load ALL the images for a patient, but do not combine them into one bag
            (
                image_patch,
                image_coord,
                image_name,
                image_shape,
            ) = patient_dataset.create_list()
            image_patches.append(image_patch)
            image_coords.append(image_coord)
            images_name.append(image_name)
            image_shapes.append(image_shape)

        clinical = self.label_bags[self.label_bags.mrn == self.patient_list[idx]][["age", "sex"]].values
        label = self.label_bags[self.label_bags.mrn == self.patient_list[idx]][["cytology"]].values.item()

        return (
            self.patient_list[idx],
            image_patches,
            image_coords,
            images_name,
            image_shapes,
            clinical,
            label
        )

    def clean_clinical(self):
        # binarize sex
        self.label_bags["sex"] = self.label_bags["sex"].map({"M": 0, "F": 1})

        # fit by train age and transform the self.label_bags age
        mrns = self.split_df.train.values
        #train_label_bags = label_bags[label_bags.mrn.str.contains("|".join(mrns))][["age",'TI-RAD']]
        #scaler = StandardScaler().fit(train_label_bags)
        #self.label_bags[["age",'TI-RAD']] = scaler.transform(self.label_bags[["age",'TI-RAD']])

        train_label_bags = self.label_bags[self.label_bags.mrn.str.contains("|".join(mrns))][["age"]]
        scaler = StandardScaler().fit(train_label_bags)
        self.label_bags[["age"]] = scaler.transform(self.label_bags[["age"]])

class PatientPatchDataset:
    def __init__(
        self,
        dataroot: str,
        csv_file: str,
        patient: List[str],
        segment: bool = False,
        tile_size: int = 32,
        stride_size: int = 32,
        recalculate: bool = False,
        need_patch_nodule_overlap: bool = False,
        stanford: bool=False
    ) -> None:
        # path to the images
        self.dataroot = dataroot
        self.csv_file = csv_file
        self.stanford = stanford

        # filter the dataset to only the images for the patient of interest
        df = pd.read_csv(csv_file)#, converters={"mrn": str})
        df['mrn'] = df['mrn'].apply(convert_to_string)

        if not self.stanford:
            df["mrn"] = format_augmented_mrn(df.mrn)

        df = df[df["mrn"].isin(patient)]

        # dataset comprises of the images/masks for patient(s) of interest
        self.images = df["img_name"].to_list()
        self.masks = df["mask_name"].to_list()

        if "FLIP" in df.columns:
            self.flips = df["FLIP"].to_list()
            self.rotations = df["ROTATION"].to_list()
        else:
            self.flips = [False] * len(self.images)
            self.rotations = [0] * len(self.images)

        self.pid = df["mrn"].to_list()
        self.df = df

        self.segment = segment
        self.tile_size = tile_size
        self.stride_size = stride_size
        self.recalculate = recalculate
        
        save_name = self.pid[0].split("_augmented")[0]

        # if self.stanford:
        #     self.save_dir = f"/raid/aradhachandran/thyroid/code/classification/BE223A_thyroidProj/iodata/patches_{self.tile_size}_{self.stride_size}_{self.csv_file.split('/')[-1].split('.csv')[0]}/stanford"
        #     self.save_path = (
        #         f"/raid/aradhachandran/thyroid/code/classification/BE223A_thyroidProj/iodata/patches_{self.tile_size}_{self.stride_size}_{self.csv_file.split('/')[-1].split('.csv')[0]}/stanford/{save_name}.h5"
        #     )
        #     os.makedirs(self.save_dir, exist_ok=True)
        # else:
        self.save_dir = f"/raid/aradhachandran/thyroid/code/classification/BE223A_thyroidProj/iodata/patches_{self.tile_size}_{self.stride_size}_{self.csv_file.split('/')[-1].split('.csv')[0]}"
        self.save_path = (
            f"/raid/aradhachandran/thyroid/code/classification/BE223A_thyroidProj/iodata/patches_{self.tile_size}_{self.stride_size}_{self.csv_file.split('/')[-1].split('.csv')[0]}/{save_name}.h5"
        )
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.need_patch_nodule_overlap = need_patch_nodule_overlap

    def __len__(self) -> int:
        return len(self.images)

    def save_as_h5py(
        self, coord_data: np.ndarray, img_name: str, patch_data: np.ndarray = None
    ) -> None:
        # create the file or open in append mode
        f = h5py.File(self.save_path, "a")

        # create a group for each image for a given patient
        img_group = f.create_group(f"{img_name}")

        # store tile size for future reference
        img_group.attrs["tile_size"] = self.tile_size

        # each image will have corresponding coordinates
        img_group.create_dataset("coords", data=coord_data)

        if patch_data is not None:
            img_group.create_dataset("patches", data=patch_data)

        f.close()

    def load_from_h5py(
        self, f: h5py.File, image_name: str, image: torch.Tensor
    ) -> Tuple[torch.Tensor, np.ndarray]:
        # load patches directly (I found this was slower on my local machine...)
        # if image_name == 'something':
        #     print('keys: ', list(f.keys()))
        if "patches" in f[image_name]:
            coords = f[image_name]["coords"][:]
            patches = f[image_name]["patches"][:]
            print('coords: ', coords)
            return torch.tensor(patches), coords

        # load coordinates and extract the patches
        else:
            patches = []
            coords = f[image_name]["coords"][:]
            tile_size = f[image_name].attrs["tile_size"]

            for coord in coords:
                img_patch = image[
                    coord[0] : coord[0] + tile_size, coord[1] : coord[1] + tile_size
                ]

                # img_patch = transforms.functional.resize(img_patch.unsqueeze(0), [self.tile_size, self.tile_size])
                # img_patch = img_patch.squeeze(0)

                patches.append(img_patch)

            return torch.stack(patches).unsqueeze(1), coords

    def get_h5py_reader(self) -> h5py.File:
        # return an open h5py file if it exists
        if os.path.isfile(self.save_path):
            f = h5py.File(self.save_path, "r")
        else:
            f = None
        return f

    def create_bag(self) -> torch.Tensor:
        all_image_patches = {}

        # look for an existing file with coordinates
        f = self.get_h5py_reader()

        # iterate over all images
        for idx in range(self.__len__()):
            # get patches for each image
            image_patches, _, image_name, _ = self.get_image(idx, f=f)

            # combine images into a bag
            if image_patches.shape[0] > 0:
                all_image_patches[image_name] = image_patches
                # all_image_patches.append(image_patches)

        if f is not None:
            f.close()
        # return torch.cat(all_image_patches)
        return all_image_patches

    def create_list(
        self,
    ) -> Tuple[List[torch.Tensor], List[np.ndarray], List[str], List[np.ndarray]]:
        all_image_patches = []
        all_coords = []
        all_names = []
        all_shapes = []

        # look for an existing file with coordinates
        f = self.get_h5py_reader()

        # iterate over all images
        for idx in tqdm(range(self.__len__()), total=self.__len__()):
            # get patches for each image
            image_patches, image_coords, image_name, orig_shape = self.get_image(
                idx, f=f
            )

            # append
            if image_patches.shape[0] > 0:
                all_image_patches.append(image_patches)
                all_coords.append(image_coords)
                all_names.append(image_name)
                all_shapes.append(orig_shape)

        if f is not None:
            f.close()

        return all_image_patches, all_coords, all_names, all_shapes


    def get_image(
        self, idx: int, f=None
    ) -> Tuple[torch.Tensor, np.ndarray, str, np.ndarray]:
        # get the paths to the images
        image_pth = os.path.join(self.dataroot, self.images[idx])
        image_name = os.path.splitext(os.path.basename(image_pth))[0]

        # load image
        image = np.load(image_pth)
        image = torch.from_numpy(image).float()

        flip = self.flips[idx]
        angle = self.rotations[idx]
        if flip:
            image = TF.hflip(image)
        image = TF.rotate(image.unsqueeze(0), angle).squeeze()

        orig_shape = np.array(image.shape)

        # Normalize the image
        img_mean = image.mean()
        img_std = image.std()
        image = (image - img_mean) / img_std

        # if cooordinates already exist, or if we do not want to recalculate
        # print('this is F: ', image_name in list(f.keys()))
        if f is not None and not self.recalculate and image_name in list(f.keys()): # I added the last conditional
            image_patches, image_coords = self.load_from_h5py(f, image_name, image)
        else:
            # recalculate coordinates/patches
            print("new patches for: ", image_pth)
            image_patches, image_coords = self.__getitem__(idx, image)

            # if there is no file with coordinates, save one
            if f is None:
                self.save_as_h5py(image_coords, image_name, patch_data=None)

        return image_patches, image_coords, image_name, orig_shape

    def __getitem__(
        self, idx: int, image: torch.Tensor
    ) -> Tuple[torch.Tensor, np.ndarray]:
        # load the mask if it exists, otherwise create a full mask
        mask_pth = self.masks[idx]
        if mask_pth is None or mask_pth != mask_pth:
            mask = np.ones(image.shape)
        else:
            mask_pth = os.path.join(self.dataroot, mask_pth)
            mask = np.load(mask_pth)

        # get region(s) to patch
        if self.segment:
            # get nodule bounding boxes if using segmentation mask
            # HAVE NOT TESTED THIS YET

            bboxs = []
            contours, _ = cv2.findContours(
                mask.copy(), 1, 1
            )  # not copying here will throw an error
            for contour in contours:
                rect = cv2.minAreaRect(
                    contour
                )  # basically you can feed this rect into your classifier
                (x, y), (w, h), a = rect
                bbox.append((x, y, w, h))
        else:
            # patch the entire image (default)
            width = image.shape[1]
            height = image.shape[0]
            bboxs = [(0, 0, width, height)]

        # adjust shape so that we patch the center of the image, and crop as evenly as we can around the edges
        for i, bbox in enumerate(bboxs):
            left, extra_width, top, extra_height = 0, 0, 0, 0

            # we want (bbox[2]-self.tile_size) % self.stride_size to be an integer, so remove any extra
            if (bbox[2] - self.tile_size) % self.stride_size > 0:
                extra_width = (bbox[2] - self.tile_size) % self.stride_size
                left = extra_width // 2

            # we want (bbox[3]-self.tile_size) % self.stride_size to be an integer, so remove any extra
            if (bbox[3] - self.tile_size) % self.stride_size > 0:
                extra_height = (bbox[3] - self.tile_size) % self.stride_size
                top = extra_height // 2

            bboxs[i] = (
                bbox[0] + left,
                bbox[1] + top,
                width - extra_width,
                height - extra_height,
            )

        good_patches_img = []
        for i, bbox in enumerate(bboxs):
            # all the top-left coordinates of the patches - should fit perfectly into the bbox ROI
            x_coords = list(
                range(
                    bbox[0],
                    bbox[0] + bbox[2] - self.tile_size + self.stride_size,
                    self.stride_size,
                )
            )
            y_coords = list(
                range(
                    bbox[1],
                    bbox[1] + bbox[3] - self.tile_size + self.stride_size,
                    self.stride_size,
                )
            )

            X, Y = np.meshgrid(x_coords, y_coords)
            coords = np.vstack([Y.ravel(), X.ravel()]).T

            for coord in coords:
                img_patch = image[
                    coord[0] : coord[0] + self.tile_size,
                    coord[1] : coord[1] + self.tile_size,
                ]

                # only append img_patch if mask_patch contains the nodule
                if self.need_patch_nodule_overlap:
                    pred_patch = mask[
                        coord[0] : coord[0] + self.tile_size,
                        coord[1] : coord[1] + self.tile_size,
                    ]
                    # If the image patch's nodule prediction is 'nodule-less' then don't use this patch
                    if np.sum(pred_patch)==0:
                        # print('nodule-less patch :(')
                        continue

                good_patches_img.append(img_patch)
        
        if len(good_patches_img)==0:
            print('CAUSED HERE by this image: ', mask_pth)
        return torch.stack(good_patches_img).unsqueeze(1), coords


def reconstruct_from_patches(
    patches: torch.Tensor,
    coords: torch.Tensor,
    save_name: str,
    tile_size: int = 128,
    shape: torch.Tensor = None,
) -> None:
    patches = patches.squeeze()
    coords = coords.squeeze()
    shape = shape.squeeze()

    patches = patches.numpy()
    coords = coords.numpy()
    shape = shape.numpy()

    if shape is None:
        max_width = np.max(coords[:, 1])
        max_height = np.max(coords[:, 0])
        counts = np.zeros([max_height + tile_size, max_width + tile_size])
        reconstructed_image = np.zeros([max_height + tile_size, max_width + tile_size])
    else:
        counts = np.zeros(shape)
        reconstructed_image = np.zeros(shape)

    for i in range(patches.shape[0]):
        col = coords[i][1]
        row = coords[i][0]
        img = patches[i]
        img[:, 0] = 255
        img[0, :] = 255
        reconstructed_image[row : row + tile_size, col : col + tile_size] = patches[i]
        counts[row : row + tile_size, col : col + tile_size] += 1

    corrected_image = np.divide(reconstructed_image, counts)
    im = Image.fromarray(corrected_image.astype(np.uint8))
    im.save(save_name + ".png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="script for testing dataloaders")

    parser.add_argument(
        "--dataroot",
        type=str,
        default="C:/Users/jeffe/Downloads/class_data",
        help="path to input file with all images",
    )
    parser.add_argument(
        "--splitroot",
        type=str,
        default="./iodata/splits/noIndet",
        help="path to splits",
    )
    parser.add_argument(
        "--csv_file_path",
        type=str,
        default="./iodata/bags/label_noIndet.csv",
        help="path to input file of all images",
    )
    parser.add_argument(
        "--show_images",
        action="store_true",
        help="show images produced by testing functions",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=64,
        help="show images produced by testing functions",
    )
    parser.add_argument(
        "--stride_size",
        type=int,
        default=64,
        help="show images produced by testing functions",
    )

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seg = PatchBagDataset(
        dataroot=args.dataroot,
        featuresroot=None,
        csv_file=args.csv_file_path,
        splitroot=args.splitroot,
        split=0,
        mode="combined",
        stride_size=args.stride_size,
        tile_size=args.tile_size,
    )
    print("Number of mrns: ", len(seg))
    loader = DataLoader(
        seg, batch_size=1, shuffle=True, collate_fn=collate_features, num_workers=4
    )

    for batch_idx, (pid, _, bag, _, _, label) in tqdm(enumerate(loader)):
        bag = bag.to(device)

        print("mrn: ", pid[0])
        print("bag shape: ", bag.shape)
        print("label: ", int(label[0]))

        # break

    # seg = PatchDatasetFromSplit(
    #     args.dataroot,
    #     args.csv_file_path,
    #     args.splitroot,
    #     split=0,
    #     stride_size=args.stride_size,
    #     tile_size=args.tile_size,
    # )
    # loader = DataLoader(seg, batch_size=1, shuffle=False)
    # for batch_idx, (mrn, images, image_coords, images_name, image_shapes) in enumerate(
    #     loader
    # ):
    #     print("batch: ", batch_idx)
    #     print("mrn: ", mrn)
    #     print("number of images: ", len(images))

    #     for i in range(len(images)):
    #         print(images[i].shape)
    #         print(image_coords[i].shape)
    #         reconstruct_from_patches(
    #             images[i],
    #             image_coords[i],
    #             save_name=f"temp/{images_name[i][0]}",
    #             tile_size=args.tile_size,
    #             shape=image_shapes[i],
    #         )

    #     break
