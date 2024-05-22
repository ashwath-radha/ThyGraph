import torchvision.transforms
from torch.utils.data import Dataset
import torch
from typing import Tuple, List
import pandas as pd
import os
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils.extract_labels import convert_to_string

def format_augmented_mrn(mrns: List[str]) -> List[str]:
    """
    Renaming of MRN from label*.csv Make the numbers have 7 digits each (not sure why we do this lol)
    """

    fixed_mrns = []
    for i in mrns:
        if "augmented" in i:
            if len(i.split("_augmented")[0]) == 7:
                fixed_mrns.append(i)
            else:
                fixed_mrns.append("0" + i)
        else:
            if len(i) == 7:
                fixed_mrns.append(i)
            else:
                fixed_mrns.append("0" + i)
    return fixed_mrns

def filter_augmentations(label_bags: pd.DataFrame, max_augments: int) -> pd.DataFrame:
    """
    label_augmented.csv has 3 augmentations per patient. this is allow us to only use up to a certain number
    """

    use = []
    for i in label_bags.mrn:
        if "augmented" in i:
            if int(i.split("_augmented")[1]) > max_augments - 1:
                use.append(False)
            else:
                use.append(True)
        else:
            use.append(True)

    return label_bags[use]

def collate_features(batch):
    mrn = np.array([item[0] for item in batch])
    frame_features = batch[0][1]
    if frame_features is None:
        frame_features = torch.tensor([0])
    patch_features = batch[0][2]    
    if patch_features is None:
        patch_features = torch.tensor([0])
    else:
        if len(patch_features) == 1:
            patch_features = patch_features[0]
        elif len(patch_features) == 2:
            patch_features =  patch_features
        else:
            raise NotImplementedError("the number of patch size should be no more than 2")

    patch_lens = batch[0][3]  # [item[3] for item in batch]
    clinical = torch.tensor(batch[0][4])
    label = torch.LongTensor([batch[0][5]])
    return [mrn, frame_features, patch_features, patch_lens, clinical, label]

class BagDataset(Dataset):
    # This is the case where we already extracted features
    def __init__(
        self,
        dataroot: str,
        csv_file: str,
        splitroot: str,
        split: int,
        mode: str = "train",
        max_augments: int = 3,
        include_tirads: bool = False,
        stanford: bool = False
    ) -> None:
        print(f"Creating {mode} dataset for MIL pipeline")

        # load labels
        label_bags = pd.read_csv(csv_file, converters={"mrn": convert_to_string}, index_col=0)
        self.reference_bags = pd.read_csv(csv_file, converters={"mrn": convert_to_string}, usecols=['mrn','file_name'])

        if not stanford:
            label_bags["mrn"] = format_augmented_mrn(label_bags.mrn)

        label_bags = label_bags.drop_duplicates(subset="mrn", keep="first").reset_index(
            drop=True
        )

        label_bags = filter_augmentations(label_bags, max_augments)

        label_bags["mrn"] = label_bags.mrn.astype(str)
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
        if mode == "train":
            mrns = splits.train.values
            self.label_bags = label_bags[label_bags.mrn.isin(mrns)].reset_index(drop=True)
        elif mode == "test":
            mrns = splits.test.values
            self.label_bags = label_bags[label_bags.mrn.isin(mrns)].reset_index(drop=True)
        elif mode == "holdout":
            mrns = [str(i).split(".")[0] for i in splits.holdout.values if i]
            mrns = ['0'+i if len(i) == 6 else i for i in mrns ]
            # only get original for val
            self.label_bags = label_bags[label_bags.mrn.isin(mrns)].reset_index(drop=True)

        self.mrns = self.label_bags.mrn.unique()
        self.dataroot = dataroot
        if include_tirads:
            clinical_features = ['sex','age','TI-RAD']
        else:
            clinical_features = ['sex','age']
        self.clinical_features = clinical_features

        if not stanford:
            self.clean_clinical(label_bags, splits, clinical_features)

    def __len__(self) -> int:
        return len(self.mrns)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, torch.Tensor]:
        mrn = self.mrns[idx]
        file_names_to_use = self.reference_bags[self.reference_bags['mrn']==mrn]['file_name'].values
        # print('file_name_to_use: ', file_names_to_use)
        bag_path = os.path.join(self.dataroot, str(mrn) + ".h5")
        label = self.label_bags[self.label_bags.mrn == mrn].label.item()
        clinical = self.label_bags[self.label_bags.mrn == mrn][self.clinical_features].values

        # load h5 or pt file
        with h5py.File(bag_path, "r") as f:
            mrn_image_names = np.array(f['img_names'], dtype=str)
            # print(mrn_image_names)
            bag_features = f["features"]
            bag_features = torch.tensor(np.array(bag_features))
            # print(bag_features[0], bag_features[3], bag_features[5])
            matching_indices = np.where(np.isin(mrn_image_names, file_names_to_use))[0]
            bag_features = bag_features[matching_indices, :]

        return mrn, bag_features, None, None, clinical, label

    def clean_clinical(self, label_bags, splits, clinical_features):
        # binarize sex
        self.label_bags["sex"] = self.label_bags["sex"].map({"M": 0, "F": 1})

        # fit by train age and transform the self.label_bags age
        mrns = splits.train.values
        #train_label_bags = label_bags[label_bags.mrn.str.contains("|".join(mrns))][["age",'TI-RAD']]
        #scaler = StandardScaler().fit(train_label_bags)
        #self.label_bags[["age",'TI-RAD']] = scaler.transform(self.label_bags[["age",'TI-RAD']])
        clinical_features = clinical_features[1:]
        train_label_bags = label_bags[label_bags.mrn.isin(mrns)][clinical_features]
        scaler = StandardScaler().fit(train_label_bags)
        self.label_bags[clinical_features] = scaler.transform(self.label_bags[clinical_features])

if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description="script for testing dataloaders")

    parser.add_argument(
        "--dataroot",
        type=str,
        default="./iodata/bags/files_augment_layer3",
        help="path to input file with all images",
    )
    parser.add_argument(
        "--splitroot", type=str, default="./iodata/splits", help="path to splits"
    )
    parser.add_argument(
        "--csv_file_path",
        type=str,
        default="./iodata/bags/label_augmented.csv",
        help="path to input file of all images",
    )
    parser.add_argument(
        "--show_images",
        action="store_true",
        help="show images produced by testing functions",
    )

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seg = BagDataset(
        dataroot=args.dataroot,
        csv_file=args.csv_file_path,
        splitroot=args.splitroot,
        split=0,
        mode="test",
    )
    print("Number of mrns: ", len(seg))
    loader = DataLoader(seg, batch_size=1, shuffle=False)

    for batch_idx, (pid, bag, label) in tqdm(enumerate(loader)):
        bag = bag.to(device)

        print("mrn: ", pid[0])
        print("bag shape: ", bag.shape)
        print("label: ", int(label[0]))

        if batch_idx == 10:
            break
