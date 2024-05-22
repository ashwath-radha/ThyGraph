# Driver script for feature extraction. Use ThyNet ensemble proxy to extract features for each US frame for each patient.
# Options to choose different pretrained weights
# Options to perform segmentation before extracting features

import argparse
import os
from typing import List
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image

from models.pre_trained_segmentation import *
from models.resnet import *
from utils.dataset_raw import *
from utils.dataset_bags import *
from utils.custom_preprocessing import *
from utils.extract_labels import convert_to_string

def argument_parser() -> argparse.ArgumentParser:
    print("Running argument parser")

    parser = argparse.ArgumentParser(description="Feature extraction script")

    # data options
    parser.add_argument(
        "--output",
        type=str,
        default="/raid/aradhachandran/thyroid/code/classification/BE223A_thyroidProj/iodata/bags/files_miccai_cv3",
        help="path to output directory for saving bags",
    )
    parser.add_argument(
        "--dataroot",
        type=str,
        default="/radraid/aradhachandran/thyroid/data/deepstorage_data/gcn_us",
        help="path to input file with all images",
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        default="/raid/aradhachandran/thyroid/code/classification/BE223A_thyroidProj/iodata/bags/label_miccai_cv3_noIndet.csv",
        help="full path to csv file",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="which gpu you want to use",
    )

    # segmentor options
    parser.add_argument(
        "--segment",
        action="store_true",
        help="whether or not to perform segmentation befor feature extraction",
    )
    parser.add_argument(
        "--segmentor",
        type=str,
        choices=["gongseg", "tnsui"],
        default="gongseg",
        help="which pretrained segmentation model to use",
    )

    # feature extractor options
    parser.add_argument(
        "--radimagenet",
        action="store_true",
        help="use resnet pretrained on radimagenet",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="batch size for data loader"
    )

    args = parser.parse_args()
    return args


# creates the models, dataloaders, and loops over all patients to extract the features for each patient
def extract(args: argparse.ArgumentParser, patients: List[str], resize_dims: int) -> None:
    print("Extracting features")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # initialize segmentation models
    if args.segment:
        print('segmenting, stop everything!!!')
        segmentor = get_segmentor(mode=args.segmentor)

    # initialize resnet model
    if args.radimagenet:
        print('radimagenet, stop everything!!!')
        extractor = ResNetRadImageNet()
    else:
        print('using resnet!')
        extractor = ResNetPretrained()
    extractor.to(device)
    extractor.eval()

    # itereate through all patients
    for patient in tqdm(patients):
        # dataset obtains all of the frames for a patient
        # print(patient)
        dataset = ImageDataset(
            args.dataroot,
            args.csv_file,
            [patient],
            resize_dims = resize_dims,
            segment=args.segment,
            segmentor=args.segmentor
        )
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
        )

        # store the feature vector and original image file name for each frame
        features = []
        img_names = []

        with torch.no_grad():
            # for each patient, iterate through all frames
            for batch_idx, (pid, imgs, _, paths) in tqdm(
                enumerate(loader), leave=False
            ):
                imgs = imgs.to(device, dtype=torch.float)

                # extract features
                batch_features = extractor(imgs)

                # store features and image_names
                features.append(batch_features)
                img_names = img_names + list(paths)

            # convert to nxd np array where n is the number of frames for a patient
            features = np.concatenate(features, axis=0)

            # save features
            save_features(args.output, str(patient), features, img_names)

def patient_list(csv_file: str, filter: bool = False, output: str = None) -> List[str]:
    print("Getting Patients")
    os.makedirs(output, exist_ok=True)

    # get unique patients by looking for unique entries in the mrn column
    df = pd.read_csv(csv_file, converters={"mrn": convert_to_string})
    patients = df["mrn"].unique()
    # patients = format_augmented_mrn(patients)

    if filter and output is not None:
        existing_patients = []
        for file in os.listdir(output):
            if file.endswith("h5"):
                existing_patients.append(os.path.splitext(file)[0])

        new_patients = set(patients).difference(set(existing_patients))
        patients = sorted(list(new_patients))

    return patients


def save_features(
    output: str, patient: str, features: np.ndarray, img_names: List[str]
) -> None:
    print(f"Saving features of a patient {patient}")

    # save a single h5 file for each patient
    patient = patient.split("-")[0]
    # if "augmented" in patient:
    #     if len(patient.split("_augmented")[0]) == 7:
    #         pass
    #     else:
    #         patient = "0" + patient
    # else:
    #     if len(i) == 7:
    #         pass
    #     else:
    #         patient = "0" + patient

    output_pth = os.path.join(output, patient + ".h5")

    # create a writable h5 file
    file = h5py.File(output_pth, "w")

    # save the nx2048 feature matrix where n is the number of frames
    file.create_dataset("features", data=features, dtype=features.dtype)

    # save a n-dimensional array to store the original image corresponding to each feature vector
    img_names = np.array(img_names, dtype=object)
    string_dt = h5py.special_dtype(vlen=str)
    file.create_dataset("img_names", data=img_names, dtype=string_dt)

def main(args: argparse.ArgumentParser) -> None:
    print("Running feature extraction script")

    # obtain a list of unique MRNs
    patients = patient_list(args.csv_file, filter=True, output=args.output)
    print(f"Feature extraction for {len(patients)} patients...")

    # extract features for all patients
    extract(args, patients, resize_dims=256)

if __name__ == "__main__":
    args = argument_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    main(args)
