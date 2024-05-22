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
# from torchvision.models.resnet import ResNet50_Weights
import torchvision.models as models
import torchvision.transforms as tv_transforms
import pandas as pd
import pickle
from itertools import product, chain
from scipy.spatial.distance import cdist
import sys
sys.path.append('/raid/aradhachandran/thyroid/code/classification/BE223A_thyroidProj')
from utils.custom_preprocessing import *
from utils.extract_labels import convert_to_string

# Calculate Euclidean distance
def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Calculate SSIM between all images given
def image_point_distance(file_to_avg_coord_loc:dict, img_names: str) -> np.ndarray:
    img_len = len(img_names)
    distance_matrix = np.zeros((img_len, img_len))
    for i in range(img_len):
        img_i = file_to_avg_coord_loc[img_names[i]]
        for j in range(i, img_len):
            img_j = file_to_avg_coord_loc[img_names[j]]
            distance = euclidean_distance(img_i, img_j)
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
#     distance_matrix = (distance_matrix - np.min(distance_matrix)) / (np.max(distance_matrix) - np.min(distance_matrix))
    return distance_matrix

# Generate ResNet50 feature extractor for patches for graph
def get_patch_feature_extractor():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # extractor = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    extractor = models.resnet50(pretrained=True)
    extractor = torch.nn.Sequential(*list(extractor.children())[:-1])
    extractor.to(device)
    extractor.eval()
    
    transforms = tv_transforms.Compose(
        [GrayscaletoRGB(),
         tv_transforms.ToTensor(),
    #      transforms.Normalize(
    #          mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    #      ),
        ]
    )
    
    return extractor, transforms, device

def extract_patches(image, patch_size=256):
    """
    Divide an 2D image into 9 patches forming a 3x3 grid.

    Parameters:
        image (numpy.ndarray): Input 2D image.

    Returns:
        list: List of 9 patches.
    """
    # Normalize the image
    img_mean = image.mean()
    img_std = image.std()
    image = (image - img_mean) / img_std

    # Get dimensions of the input image
    rows, cols = image.shape

    # Initialize list to store patches
    patches, patch_coordinates = [], []

    # Calculate step size for creating the 3x3 grid
    step_row = max(1, rows // 3)
    step_col = max(1, cols // 3)

    # Loop through the 3x3 grid
    for i in range(3):
        for j in range(3):
            # Calculate center point for the current grid entry
            center_row = min((i + 0.5) * step_row, rows - 1)
            center_col = min((j + 0.5) * step_col, cols - 1)

            # Calculate starting row and column indices for the patch
            start_row = max(0, int(center_row - patch_size / 2))
            start_col = max(0, int(center_col - patch_size / 2))

            # Calculate ending row and column indices for the patch
            end_row = min(rows, start_row + patch_size)
            end_col = min(cols, start_col + patch_size)

            # Check if patch extends outside the input array and adjust coordinates
            if end_row - start_row < patch_size:
                start_row = max(0, end_row - patch_size)
            if end_col - start_col < patch_size:
                start_col = max(0, end_col - patch_size)

            # Extract the patch from the array
            patch = image[start_row:end_row, start_col:end_col]

            # Append the patch to the list
            patches.append(patch)
            patch_coordinates.append(((start_row,end_row), (start_col,end_col)))

    return patches, patch_coordinates

# Generate coordinates/locations for current cohort and return cohort information
def generate_current_cohort(csv_file, dataroot, add_coords=True):
    current_cohort = csv_file.copy() #pd.read_csv(csv_file, index_col=0)
    current_cohort['mrn'] = current_cohort['mrn'].apply(convert_to_string)
    current_cohort['mrn_acc'] = current_cohort['img_name'].str[:16]
    current_cohort['img_name'] = current_cohort['img_name'].str.replace('.npy','').str[17:]
    # Load in pickle file mapping MRN-Acc to Data Pull/Image Study
    image_study_mrn_accs_dict = {}
    with open('/raid/aradhachandran/thyroid/code/classification/labels/image_study_mrn_accs_012424.pkl', 'rb') as f:
        image_study_mrn_accs = pickle.load(f)
    for thing in image_study_mrn_accs:
        m, a = thing.split('/')[-2:]
        image_study_mrn_accs_dict[os.path.join(m, a)] = thing
    current_cohort['mrn_acc'] = current_cohort['mrn_acc'].map(image_study_mrn_accs_dict)
    # print(current_cohort.shape, current_cohort['mrn_acc'].nunique())

    img_to_location = setup_img_to_location()
    # Remove cohort members without a processed extracted location
    current_cohort['full_path'] = current_cohort['mrn_acc'] + '/' + current_cohort['img_name']
    current_cohort['dgx_path'] = current_cohort['full_path'].apply(
        lambda x: os.path.join(dataroot, '/'.join(x.split('/')[3:])+'.npy'))
    current_cohort['extracted_loc'] = current_cohort['full_path'].map(img_to_location)
    current_cohort = current_cohort[~current_cohort['extracted_loc'].isna()]
    # drop instances with RIGHT and LEFT in extracted_loc
    current_cohort = current_cohort[current_cohort['extracted_loc'].apply(lambda x: not all(item in x for item in ['RIGHT', 'LEFT']))]
    # drop instances with TRANS and LONG in extracted_loc
    current_cohort = current_cohort[current_cohort['extracted_loc'].apply(lambda x: not all(item in x for item in ['TRANS', 'LONG']))]
    # drop instances with LATERAL and MEDIAL in extracted_loc
    current_cohort = current_cohort[current_cohort['extracted_loc'].apply(lambda x: not all(item in x for item in ['LATERAL', 'MEDIAL']))]
    # drop instances with SUPERIOR and INFERIOR in extracted_loc
    current_cohort = current_cohort[current_cohort['extracted_loc'].apply(lambda x: not all(item in x for item in ['SUPERIOR', 'INFERIOR']))]
    # drop instances where a LEVEL is in the extracted_loc list
    current_cohort = current_cohort[~current_cohort['extracted_loc'].apply(lambda x: any('LEVEL' in entry for entry in x))]
    # remove illogical location permutations
    current_cohort = current_cohort[current_cohort['extracted_loc'].apply(lambda x: not all(item in x for item in ['SUPERIOR', 'LATERAL']))]
    current_cohort = current_cohort[current_cohort['extracted_loc'].apply(lambda x: not all(item in x for item in ['SUPERIOR', 'MEDIAL']))]
    current_cohort = current_cohort[current_cohort['extracted_loc'].apply(lambda x: not all(item in x for item in ['SUPERIOR', 'MID']))]
    current_cohort = current_cohort[current_cohort['extracted_loc'].apply(lambda x: not all(item in x for item in ['INFERIOR', 'LATERAL']))]
    current_cohort = current_cohort[current_cohort['extracted_loc'].apply(lambda x: not all(item in x for item in ['INFERIOR', 'MEDIAL']))]
    current_cohort = current_cohort[current_cohort['extracted_loc'].apply(lambda x: not all(item in x for item in ['INFERIOR', 'MID']))]
    current_cohort = current_cohort[current_cohort['extracted_loc'].apply(lambda x: not all(item in x for item in ['LATERAL', 'MID']))]
    current_cohort = current_cohort[current_cohort['extracted_loc'].apply(lambda x: not all(item in x for item in ['MEDIAL', 'MID']))]
    current_cohort = current_cohort[current_cohort['extracted_loc'].apply(lambda x: not all(item in x for item in ['TRANS', 'LATERAL']))]
    current_cohort = current_cohort[current_cohort['extracted_loc'].apply(lambda x: not all(item in x for item in ['TRANS', 'MEDIAL']))]
    current_cohort = current_cohort[current_cohort['extracted_loc'].apply(lambda x: not all(item in x for item in ['LONG', 'SUPERIOR']))]
    current_cohort = current_cohort[current_cohort['extracted_loc'].apply(lambda x: not all(item in x for item in ['LONG', 'INFERIOR']))]

    # # remove locations with length 1
    # current_cohort = current_cohort[current_cohort['extracted_loc'].apply(lambda x: len(x)>1 or any('ISTHMUS' in entry for entry in x))]
    
    if add_coords:    
        full_path_to_grid = setup_full_path_to_grid(current_cohort)
        current_cohort['grid_coords_plane'] = current_cohort['full_path'].map(full_path_to_grid)
    
    return current_cohort

# Filter and process image to location information
def setup_img_to_location():
    with open('/raid/aradhachandran/thyroid/code/classification/labels/img_to_location.pkl', 'rb') as f:
        img_to_location = pickle.load(f)
    
    correction_dict = {
        'Thyroid': 'THYROID', 'LVI': 'LEVEL I', 'L7': 'LEVEL VII', 'L6': 'LEVEL VI',
        'L1': 'LEVEL I', 'INF': 'INFERIOR', 'SUP': 'SUPERIOR', 'LAT': 'LATERAL',
        'UPPER':'SUPERIOR', 'LOWER':'INFERIOR', 'SAG': 'LONG'
    }

    correction_tuples = {
        tuple(['LVL', 'I']): ['LEVEL I'],
        tuple(['LV', 'I']): ['LEVEL I'],
        tuple(['LEV', '7']): ['LEVEL VII'],
    }

    for key, inner_list in img_to_location.items():
        img_to_location[key] = [correction_dict.get(item, item) for item in inner_list]

    for key, inner_list in img_to_location.items():
        img_to_location[key] = correction_tuples.get(tuple(inner_list), inner_list)

    for key, inner_list in img_to_location.items():
        img_to_location[key] = [item for item in inner_list if item not in {'THYROID','LOBE','LEV', 'LEVEL', 'REPORT', 'REPORT2','C8-5'}]

    for key, inner_list in img_to_location.items():
        img_to_location[key] = list(set(inner_list))

    img_to_location = {key: inner_list for key, inner_list in img_to_location.items() if any(inner_list)}

    # resort each value in img_to_location based on preset_order
    preset_order = ['RIGHT','LEFT','ISTHMUS', 'LEVEL I', 'LEVEL II', 'LEVEL III', 'LEVEL IV', 'LEVEL V', 'LEVEL VI', 'LEVEL VII',
                    'LONG','TRANS',
                    'LATERAL','MEDIAL',
                    'SUPERIOR','INFERIOR',
                    'MID']
    for key, inner_list in img_to_location.items():
        img_to_location[key] = sorted(inner_list, key=lambda x: preset_order.index(x))

    return img_to_location


def setup_full_path_to_grid(current_cohort):
    full_path_to_grid = {}
    total_mrns = current_cohort['mrn'].nunique()
    for _, mrn_info in tqdm(current_cohort.groupby('mrn')[['full_path','extracted_loc']], total=total_mrns):
        mrn_info = mrn_info.sort_values(by='extracted_loc', key=lambda x: x.apply(len), ascending=False)
        for full_path, extracted_loc in mrn_info[['full_path', 'extracted_loc']].values:
            extracted_loc_copy = extracted_loc.copy()
            if 'ISTHMUS' in extracted_loc:
                # coords, plane = [(5, i) for i in range(1,4)], 'L' # COMMENTED 022224
                coords, plane = [(5, i) for i in range(0,3)], 'L'
            elif 'RIGHT' in extracted_loc:
                extracted_loc_copy.remove("RIGHT")
                coords, plane = get_coords_logic(extracted_loc_copy, side='RIGHT')
            elif 'LEFT' in extracted_loc:
                extracted_loc_copy.remove("LEFT")
                coords, plane = get_coords_logic(extracted_loc_copy, side='LEFT')
            else:
                coords, plane = get_coords_logic(extracted_loc_copy)
            full_path_to_grid[full_path] = (coords, plane)
    return full_path_to_grid

POSITIONS = {'LONG': ['LATERAL', 'MID', 'MEDIAL'],
             'TRANS': ['SUPERIOR', 'MID', 'INFERIOR']}
ALL_POSITIONS = set([item for sublist in POSITIONS.values() for item in sublist])
SIDES = ['RIGHT', 'LEFT']
PLANES = ['LONG', 'TRANS']

SIDE_AXIS = {'LEFT':0, 'RIGHT': 1}

X_AXIS = {'RIGHT LATERAL':1, 'RIGHT MID':2, 'RIGHT MEDIAL':3, 
          'LEFT MEDIAL':7, 'LEFT MID':8, 'LEFT LATERAL':9}
Y_AXIS = {'INFERIOR':1, 'MID':2, 'SUPERIOR':3}

def get_length2_loc_logic(extracted_loc, side):
    plane = set(PLANES).intersection(set(extracted_loc))
    assert len(plane)==1
    plane = plane.pop()
    
    plane_positions = set(POSITIONS[plane])
    position = plane_positions.intersection(set(extracted_loc))
    assert len(position)==1
    position = position.pop()
    
    if plane=='LONG':
        return [(X_AXIS[f'{side} {position}'], i) for i in range(1,4)], 'L'
    if plane=='TRANS':
        if side=='LEFT':
            return [(i, Y_AXIS[position]) for i in range(7,10)], 'T'
        if side=='RIGHT':
            return [(i, Y_AXIS[position]) for i in range(1,4)], 'T'

def get_length1_loc_logic(extracted_loc, side):
    plane = set(PLANES).intersection(set(extracted_loc))
    try:
        plane = plane.pop()
        if side=='LEFT' and plane=='LONG':
            return list(product(range(7,10), range(1,4))), 'L'
        if side=='RIGHT' and plane=='LONG':        
            return list(product(range(1,4), range(1,4))), 'L'
        if side=='LEFT' and plane=='TRANS':
            return sorted(list(product(range(7,10), range(1,4))), key=lambda x: (x[1], x[0])) , 'T'
        if side=='RIGHT' and plane=='TRANS':
            return sorted(list(product(range(1,4), range(1,4))), key=lambda x: (x[1], x[0])) , 'T'

    except KeyError:
        try:
            position = ALL_POSITIONS.intersection(set(extracted_loc))
            position = position.pop()
            try:
                return [(X_AXIS[f'{side} {position}'], i) for i in range(1,4)], 'L'
            except KeyError:
                if side=='LEFT':        
                    # return [(i, Y_AXIS[position]) for i in range(7,10)], 'T'
                    return sorted([(i, Y_AXIS[position]) for i in range(7,10)], key=lambda x: (x[1], x[0])) , 'T'
                if side=='RIGHT':        
                    # return [(i, Y_AXIS[position]) for i in range(1,4)], 'T'
                    return sorted([(i, Y_AXIS[position]) for i in range(1,4)], key=lambda x: (x[1], x[0])) , 'T'

        except KeyError:            
            return 

def get_length0_loc_logic(extracted_loc, side):
    if side=='RIGHT':
        return list(product(range(1,4), range(1,4))), 'TL'
    elif side=='LEFT':
        return list(product(range(7,10), range(1,4))), 'TL'
    else: # side is None
        item = extracted_loc[0]
        if item=='TRANS':
            if 'SUPERIOR' in extracted_loc: y_range = [3]
            elif 'MID' in extracted_loc: y_range = [2]
            elif 'INFERIOR' in extracted_loc: y_range = [1]
            else: y_range = range(1,4)
            # return list(product(chain(range(1,4), range(7,10)), y_range)), 'T'
            return sorted(list(product(chain(range(1,4), range(7,10)), y_range)), key=lambda x: (x[1], x[0])) , 'T'

        elif item=='LONG':
            if 'LATERAL' in extracted_loc: x_range = [1, 9]
            elif 'MID' in extracted_loc: x_range = [2, 8]
            elif 'MEDIAL' in extracted_loc: x_range = [3, 7]
            else: x_range = chain(range(1,4), range(7,10))
            return list(product(x_range, range(1,4))), 'L'
        elif item=='MID':
            first = [(i, 2) for i in chain(range(1,4), range(7,10))]
            second = list(product([2,8], range(1,4)))
            return first+second, 'TL'
        else:
            try:
                # return [(i, Y_AXIS[item]) for i in chain(range(1,4), range(7,10))], 'T'
                return sorted([(i, Y_AXIS[item]) for i in chain(range(1,4), range(7,10))], key=lambda x: (x[1], x[0])) , 'T' # SHOULD VERIFY

            except KeyError:
                x_indeces = [X_AXIS[keys] for keys in X_AXIS if item in keys]
                return list(product(x_indeces, range(1,4))), 'L'

def get_coords_logic(extracted_loc, side=None):
    if len(extracted_loc)==2 and side:
        coords, plane = get_length2_loc_logic(extracted_loc, side)
    if len(extracted_loc)==1 and side:
        coords, plane = get_length1_loc_logic(extracted_loc, side)
    if len(extracted_loc)==0 or not side:
        coords, plane = get_length0_loc_logic(extracted_loc, side)
    return coords, plane
