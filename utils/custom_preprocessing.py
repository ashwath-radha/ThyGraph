# defines the functions for preprocessing the ultrasound images

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pydicom
from tqdm import tqdm
from skimage.exposure import rescale_intensity
import skimage
from typing import Union
import torch
import torchvision.transforms.functional as TF


class GrayscaletoRGB:
    # Duplicate a single channel grayscale image across RBG channels

    def __call__(
        self, img: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        # use np repeat if np array
        if isinstance(img, np.ndarray):
            return np.repeat(img[:, :, np.newaxis], 3, axis=2)

        # use torch repeat_interleave if tensor
        elif isinstance(img, torch.Tensor):
            # if in batch format, then repeat along the channel dimension which is dim=1
            if len(img.shape) == 4:
                return torch.repeat_interleave(img, 3, dim=1)
            # if in image format, then repeat along the channel dimension which is dim=0
            else:
                return torch.repeat_interleave(img, 3, dim=0)

class SquarePad:
    # Equal padding on left/right and top/bottom to create a square image

    def __call__(self, img: np.ndarray) -> np.ndarray:
        # original image shape
        w, h = img.shape

        # maximum dimension of all the images
        scale = np.max([568, 759])  # 368, 494

        # equal padding on left/right, top/bottom
        hp = int((scale - w) / 2)
        vp = int((scale - h) / 2)

        # pad with zeros
        # print(hp, vp)
        if vp < 0:
            vp = 0
        if hp < 0:
            hp = 0
        padding = ((hp, hp), (vp, vp))
        return np.pad(img, padding, mode="constant", constant_values=0)


class Resize:
    # Resize an np array
    def __init__(self, size):
        if isinstance(size, int):
            self._size = (size, size)
        else:
            self._size = size

    def __call__(self, img: np.ndarray) -> np.ndarray:
        resize_image = skimage.transform.resize(img, self._size)
        # the resize will return a float64 array
        return resize_image


class FixedFlip:
    """Flip image"""

    def __init__(self, flip: bool):
        self.flip = flip

    def __call__(self, x):
        if self.flip:
            return TF.hflip(x)
        else:
            return x


class FixedRotation:
    """Rotate by the given angle."""

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return TF.rotate(x, self.angle)


################
#  Below is from Ashwath
################


def full_preprocess(img, mask, crop_dims=(100, -100, 125, -140)):
    cropped_img = img[crop_dims[0] : crop_dims[1], crop_dims[2] : crop_dims[3]]
    cropped_mask = mask[crop_dims[0] : crop_dims[1], crop_dims[2] : crop_dims[3]]
    cropped_img, cropped_mask = autocrop(cropped_img, cropped_mask)
    ret_img, ret_mask = findAndRemoveP(
        cropped_img, cropped_mask
    )  # sometimes have incorrectly placed squares
    p2, p98 = np.percentile(ret_img, (2, 98))
    rescaled = rescale_intensity(
        ret_img, in_range=(p2, p98)
    )  # rescale_intensity(us_img)
    ret_img = whitening(rescaled)
    return ret_img, ret_mask  # cropped_img, cropped_mask


def autocrop(image, mask, threshold=0, ret_coords=False):
    """Crops any edges below or equal to threshold

    Returns cropped image.

    """
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2

    # print(flatImage.shape)
    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    # print(rows.shape, rows.size)
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0] : cols[-1] + 1, rows[0] : rows[-1] + 1]
        mask = mask[cols[0] : cols[-1] + 1, rows[0] : rows[-1] + 1]
    else:
        image = image[:1, :1]

    if ret_coords:
        if rows.size != 0:
            # print(cols[0], cols[-1] + 1, rows[0], rows[-1] + 1)
            return cols[0], cols[-1] + 1, rows[0], rows[-1] + 1
        else:
            return 0, flatImage.shape[1], 0, flatImage.shape[0]
    else:
        return image, mask


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    return err


# this code has been edited. previously the values in range() were hardcoded, but some images in the dataset were smaller than those hardcoded values
def findAndRemoveP(
    img, mask, p_path="./iodata/feature_extractor/theP.npy", threshold=1000
):
    """
    The P is the ultrasound probe orientation marker. You typically see it at the
    top left of the image. This is just a function that detects and blacks out
    that marker.
    """
    theP = np.load(p_path)

    best_i, best_j, min_mse = 0, 0, 10000000
    for i in range(img.shape[0] - 25):
        for j in range(img.shape[1] - 25):
            mse_calc = mse(theP, img[i : i + 25, j : j + 25])
            if mse_calc < min_mse:
                min_mse = mse_calc
                best_i = i
                best_j = j
    if min_mse < threshold:
        img[best_i : best_i + 25, best_j : best_j + 25] = 0
        mask[best_i : best_i + 25, best_j : best_j + 25] = 0
    return img, mask


def whitening(image):
    """Whitening. Normalises image to zero mean and unit variance."""

    image = image.astype(np.float32)

    mean = np.mean(image)
    std = np.std(image)

    if std > 0:
        ret = (image - mean) / std
    else:
        ret = image * 0.0
    return ret
