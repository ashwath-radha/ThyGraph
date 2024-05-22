# defines the objects for loading the original ultrasound images

import argparse
import os
from typing import Tuple, Union, List
from tqdm import tqdm
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
import pandas as pd


def argument_parser() -> argparse.ArgumentParser:
    print("Running argument parser")

    parser = argparse.ArgumentParser(description="Feature extraction script")

    # data options
    parser.add_argument(
        "--output",
        type=str,
        default="./iodata/bags/",
        help="path to output directory for saving bags",
    )
    parser.add_argument(
        "--csv_file", type=str, default="./label.csv", help="full path to csv file"
    )
    parser.add_argument(
        "--num_augments",
        type=int,
        default=3,
        help="number of augmentations to perform per patient",
    )

    args = parser.parse_args()
    return args


def get_augmented_version(
    df: pd.DataFrame, patient: str, augment_index: int = 0, reference=None
):
    if reference is not None:
        reference = pd.read_csv(reference)

    # We're creating another set of rows
    df_copy = df.copy()
    df_copy = df_copy[df_copy["mrn"] == patient].reset_index(drop=True)

    # Iterate through each image
    for i, row in df_copy.iterrows():
        augmented_mrn = row["mrn"] + f"_augmented{augment_index}"
        df_copy.at[i, "mrn"] = augmented_mrn

        existing = reference[
            (reference["mrn"] == augmented_mrn)
            & (reference["img_name"] == row["img_name"])
        ]
        assert len(existing) < 2

        if len(existing) > 0:
            df_copy.at[i, "FLIP"] = existing["FLIP"].values[0]
            df_copy.at[i, "ROTATION"] = existing["ROTATION"].values[0]
        else:
            print(patient, row["img_name"])
            if random.random() > 0.5:
                df_copy.at[i, "FLIP"] = True

            df_copy.at[i, "ROTATION"] = random.randrange(-15, 16)

    return df_copy


def main(args: argparse.ArgumentParser) -> None:
    print("Running augmentation script")

    df = pd.read_csv(os.path.join(args.output, args.csv_file), converters={"mrn": str})

    # Add augmentations
    df["FLIP"] = False
    df["ROTATION"] = 0

    # obtain a list of unique MRNs
    patients = df["mrn"].unique()

    # augment all patients
    for patient in tqdm(patients):
        for i in range(args.num_augments):
            augmented = get_augmented_version(df, patient, augment_index=i)
            df = pd.concat([df, augmented]).reset_index(drop=True)

    df = df.drop("Unnamed: 0", axis=1)

    df.to_csv(
        os.path.join(
            args.output, os.path.splitext(args.csv_file)[0] + "_augmented.csv"
        ),
        index=False,
    )


if __name__ == "__main__":
    args = argument_parser()
    main(args)
