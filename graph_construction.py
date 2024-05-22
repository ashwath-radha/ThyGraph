import argparse
from typing import List
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd 
import os
from pandas import Series
import glob
import pickle
import datetime
from tqdm import tqdm
import h5py
import networkx as nx
import json
import sys
from collections import defaultdict

sys.path.append('/raid/aradhachandran/thyroid/code/classification/BE223A_thyroidProj')
from utils.extract_labels import *
from utils.utils import image_similarity, crop_images
from utils.graph_utils import *

def argument_parser() -> argparse.ArgumentParser:
    print('Running argument parser')

    parser = argparse.ArgumentParser(description='Feature extraction script')
    
    parser.add_argument(
        "--input_features",
        type=str,
        default="/raid/aradhachandran/thyroid/code/classification/BE223A_thyroidProj/iodata/bags/files_miccai_cv3",
        help="path to input directory for loading features"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/raid/aradhachandran/thyroid/code/classification/BE223A_thyroidProj/iodata/bags/gcn_files_miccai_cv3",
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
        "--edge_method",
        type=str,
        default="feature_sim",
        choices=["feature_sim", "image_ssim", "image_loc", "patch_loc"],
        help="pick a graph construction method"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="9",
        help="which gpu you want to use",
    )

    args = parser.parse_args()
    return args

def construct_graphs(args: argparse.ArgumentParser) -> None:
    # Iterate through patients
    patients = [h5file.replace('.h5','') for h5file in os.listdir(args.input_features)]
    output_path = args.output + f'_{args.edge_method}'
    os.makedirs(output_path, exist_ok=True)

    # Load csv with labels for each image
    label_csv = pd.read_csv(args.csv_file, index_col=0, converters={"mrn": convert_to_string})
    if args.edge_method=='patch_loc': current_cohort = generate_current_cohort(label_csv, args.dataroot)

    for patient_id in tqdm(patients, total=len(patients)):
        if patient_id not in label_csv['mrn'].unique():
            print(f'{patient_id} not in label_csv file...SKIPPING')
            continue
        print('MRN: ', patient_id)
        if args.edge_method=='feature_sim':
            make_feature_similarity_graph(patient_id, output_path, args.input_features, label_csv)
        if args.edge_method=='image_ssim':
            make_image_similarity_graph(patient_id, output_path, args.input_features, label_csv, args.dataroot)
        if args.edge_method=='image_loc':
            with open('/raid/aradhachandran/thyroid/code/classification/labels/file_to_avg_coord_loc.pkl', 'rb') as f:
                file_to_avg_coord_loc = pickle.load(f)
            make_image_location_graph(patient_id, output_path, args.input_features, label_csv, file_to_avg_coord_loc)
        if args.edge_method=='patch_loc':
            extractor, transforms, device = get_patch_feature_extractor()
            # current_cohort = generate_current_cohort(label_csv, args.dataroot)
            revised_make_patch_location_graph(patient_id, output_path, args.dataroot, label_csv, current_cohort, extractor, transforms, device, save_graph=True)

def revised_make_patch_location_graph(patient_id: str, output_path: str, dataroot: str, label_csv: pd.DataFrame, current_cohort: pd.DataFrame, extractor, transforms, device, save_graph=False):
    G = nx.Graph()
    patch_to_coords = defaultdict(list)
    patch_name_to_node_index, node_index = {}, 0
    patient_bag = label_csv[label_csv['mrn']==patient_id].img_name.values
    patch_name_to_features = {}
    
    for img_path in tqdm(patient_bag, total=len(patient_bag)):
        dgx_path = os.path.join(dataroot, img_path)
        assert os.path.exists(dgx_path)
        try:
            assert dgx_path in current_cohort.dgx_path.values
        except AssertionError:
            print(f"{dgx_path} is not in current_cohort dgx_path column, continue")
            continue

        image = np.load(dgx_path)
        patches, patch_coordinates = extract_patches(image, patch_size=224)
        for i, patch in enumerate(patches, start=1):
            patch_name = dgx_path+f'_patch{i}'
            with torch.no_grad():
                input_image = transforms(patch).unsqueeze(0)
                features = extractor(input_image.to(device, dtype=torch.float))
                feature_str = json.dumps(features.cpu().numpy().tolist())
                patch_name_to_features[patch_name] = feature_str
        
        coords, plane = current_cohort[current_cohort.dgx_path==dgx_path].grid_coords_plane.values[0]
        grouped_tuples = [coords[i:i+3] for i in range(0, len(coords), 3)]
        if plane=='T':
            for gt in grouped_tuples:
                patch_num = 1
                for z_dim in [1, 0, -1]:
                    for lil_coord in gt:
                        lil_coord = lil_coord + (z_dim,)
                        patch_to_coords[dgx_path+f'_patch{patch_num}'].append(lil_coord)
                        patch_num += 1
        elif plane=='L' or plane=='TL':
            for gt in grouped_tuples:
                patch_num = 1
                for z_dim in [1, 0, -1]:
                    for lil_coord in reversed(gt):
                        lil_coord = lil_coord + (z_dim,)
                        patch_to_coords[dgx_path+f'_patch{patch_num}'].append(lil_coord)
                        patch_num += 1

    modified_patch_to_coords = {}
    for file_name, coordinates in patch_to_coords.items():
        if len(coordinates) > 0:
            for i, coord in enumerate(coordinates, start=1):
                new_key = f"{file_name}_coord{i}"
                modified_patch_to_coords[new_key] = [coord]
    
    all_modified_patch_names = list(modified_patch_to_coords.keys())
    for patch_coord_name in all_modified_patch_names:
        patch_name = re.sub(r'_coord\d+', '', patch_coord_name)
        G.add_node(node_index, features=patch_name_to_features[patch_name], name=patch_coord_name)
        patch_name_to_node_index[patch_coord_name] = node_index
        node_index+=1
    coordinates_list = np.array(list(modified_patch_to_coords.values())).squeeze()
    distance_matrix = cdist(coordinates_list, coordinates_list)
    assert len(all_modified_patch_names)==len(coordinates_list)==len(distance_matrix)
    
    threshold = 2.2 #1.9
    for i in range(len(coordinates_list)):
        for j in range(i, len(coordinates_list)):
            node_distance = distance_matrix[i, j]
            if node_distance <= threshold:
                patch_i, patch_j = all_modified_patch_names[i], all_modified_patch_names[j]
                node_i, node_j = patch_name_to_node_index[patch_i], patch_name_to_node_index[patch_j]
                if G.has_edge(node_i, node_j):
                    print(f"Edge between nodes {node_i} and {node_j} exists and is being replaced...")
                G.add_edge(node_i, node_j, weight=threshold-node_distance)  # Add an edge between nodes i and j
    print('Graph constructed')
    print(len(G.nodes), len(G.edges))
    # save Graph as a file
    if save_graph:
        nx.write_graphml(G, os.path.join(output_path,f'{patient_id}.graphml'))
        print('Graph saved')
    else:
        return G, distance_matrix, modified_patch_to_coords

# def make_patch_location_graph(patient_id: str, output_path: str, dataroot: str, label_csv: pd.DataFrame, current_cohort: pd.DataFrame, extractor, transforms, device):
#     G = nx.Graph()
#     patch_to_coords = defaultdict(list)
#     patch_name_to_node_index, node_index = {}, 0
#     patient_bag = label_csv[label_csv['mrn']==patient_id].img_name.values
#     for img_path in tqdm(patient_bag, total=len(patient_bag)):
#         dgx_path = os.path.join(dataroot, img_path)
#         assert os.path.exists(dgx_path)
#         try:
#             assert dgx_path in current_cohort.dgx_path.values
#         except AssertionError:
#             print(f"{dgx_path} is not in current_cohort dgx_path column, continue")
#             continue

#         image = np.load(dgx_path)
#         patches, patch_coordinates = extract_patches(image, patch_size=224)

#         for i, patch in enumerate(patches, start=1):
#             patch_name = dgx_path+f'_patch{i}'
#             with torch.no_grad():
#                 input_image = transforms(patch).unsqueeze(0)
#                 features = extractor(input_image.to(device, dtype=torch.float))
#                 feature_str = json.dumps(features.cpu().numpy().tolist())
#             G.add_node(node_index, features=feature_str, name=patch_name)
#             patch_name_to_node_index[patch_name] = node_index
#             node_index+=1
#         coords, plane = current_cohort[current_cohort.dgx_path==dgx_path].grid_coords_plane.values[0]
#         grouped_tuples = [coords[i:i+3] for i in range(0, len(coords), 3)]
#         if plane=='T':
#             for gt in grouped_tuples:
#                 patch_num = 1
#                 for z_dim in [1, 0, -1]:
#                     for lil_coord in gt:
#                         lil_coord = lil_coord + (z_dim,)
#                         patch_to_coords[dgx_path+f'_patch{patch_num}'].append(lil_coord)
#                         patch_num += 1
#         elif plane=='L':
#             for gt in grouped_tuples:
#                 patch_num = 1
#                 for z_dim in [1, 0, -1]:
#                     for lil_coord in reversed(gt):
#                         lil_coord = lil_coord + (z_dim,)
#                         patch_to_coords[dgx_path+f'_patch{patch_num}'].append(lil_coord)
#                         patch_num += 1
#         else: # assuming L orientation for images with TL plane, TODO make a model that can classify an image to T or L plane for more accurate representation in graph
#             for gt in grouped_tuples:
#                 patch_num = 1
#                 for z_dim in [1, 0, -1]:
#                     for lil_coord in reversed(gt):
#                         lil_coord = lil_coord + (z_dim,)
#                         patch_to_coords[dgx_path+f'_patch{patch_num}'].append(lil_coord)
#                         patch_num += 1
#     # OLD APPROACH
#     # sorted_tuples = sorted(coords, key=lambda x: (x[0], x[1]))
#     # grouped_tuples = [sorted_tuples[i:i+3] for i in range(0, len(sorted_tuples), 3)]
#     # for gt in grouped_tuples:
#     #         patch_num = 1
#     #         for z_dim in [1, 0, -1]:
#     #             for lil_coord in reversed(gt):
#     #                 lil_coord = lil_coord + (z_dim,)
#     #                 patch_to_coords[dgx_path+f'_patch{patch_num}'].append(lil_coord)
#     #                 patch_num += 1

#     modified_patch_to_coords = {}
#     for file_name, coordinates in patch_to_coords.items():
#         if len(coordinates) > 0:
#             for i, coord in enumerate(coordinates, start=1):
#                 new_key = f"{file_name}_coord{i}"
#                 modified_patch_to_coords[new_key] = [coord]

#     all_modified_patch_names = list(modified_patch_to_coords.keys())
#     coordinates_list = np.array(list(modified_patch_to_coords.values())).squeeze()
#     distance_matrix = cdist(coordinates_list, coordinates_list)
#     # norm_distance_matrix = distance_matrix/np.max(distance_matrix)                
#     assert len(all_modified_patch_names)==len(coordinates_list)==len(distance_matrix)
#     threshold = 1.9 #2.2
#     for i in range(len(coordinates_list)):
#         for j in range(i, len(coordinates_list)):
#             node_distance = distance_matrix[i, j]
#             if node_distance <= threshold:
#                 patch_i, patch_j = all_modified_patch_names[i], all_modified_patch_names[j]
#                 patch_i, patch_j = re.sub(r'_coord\d+', '', patch_i), re.sub(r'_coord\d+', '', patch_j)
#                 node_i, node_j = patch_name_to_node_index[patch_i], patch_name_to_node_index[patch_j]
#                 if G.has_edge(node_i, node_j):
#                     print(f"Edge between nodes {node_i} and {node_j} exists and is being replaced...")
#                 G.add_edge(node_i, node_j, weight=threshold-node_distance)  # Add an edge between nodes i and j
#     print('Graph constructed')
#     print(len(G.nodes), len(G.edges))
#     return G, distance_matrix, modified_patch_to_coords
#     # # save Graph as a file
#     # nx.write_graphml(G, os.path.join(output_path,f'{patient_id}.graphml'))
#     # print('Graph saved')

def make_image_location_graph(patient_id: str, output_path: str, input_features: str, label_csv: pd.DataFrame, file_to_avg_coord_loc: dict) -> None:
    with h5py.File(os.path.join(input_features, f'{patient_id}.h5'), 'r') as file:
        # Load features for the patient
        features = file['features'][:]
        img_names = file['img_names'][:]

        # Remap img_names to img_filepaths (full filepath)
        img_filepaths = []
        for fp in img_names:
            fp = fp.decode('utf-8')
            full_fp = label_csv[(label_csv['mrn']==patient_id) & (label_csv['img_name'].str.contains(fp))]['img_name'].values[0]
            img_filepaths.append(full_fp)
        assert len(img_names)==len(img_filepaths)
        img_filepaths = [x for x in img_filepaths if x in file_to_avg_coord_loc]

        # Create a graph for the patient
        G = nx.Graph()
        for i in range(len(img_filepaths)):
            feature_str = json.dumps(features[i].tolist())  # Convert the NumPy array to a JSON string
            G.add_node(i, features=feature_str, name=img_filepaths[i])  # Add nodes with feature strings

        # Calculate coordinate distance and add edges based on the threshold
        distance_matrix = image_point_distance(file_to_avg_coord_loc, img_filepaths)
        threshold = 2.2 # 2.5
        for i in range(len(img_filepaths)):
            for j in range(i + 1, len(img_filepaths)):
                node_distance = distance_matrix[i, j]
                if node_distance < threshold:
                    G.add_edge(i, j, weight=threshold-node_distance)  # Add an edge between nodes i and j
        print('Graph constructed')
        print(len(G.nodes), len(G.edges))

        # save Graph as a file
        nx.write_graphml(G, os.path.join(output_path, f'{patient_id}.graphml'))
        print('Graph saved')

def make_image_similarity_graph(patient_id: str, output_path: str, input_features: str, label_csv: pd.DataFrame, dataroot: str) -> None:
    with h5py.File(os.path.join(input_features, f'{patient_id}.h5'), 'r') as file:
        # Load features for the patient
        features = file['features'][:]
        img_names = file['img_names'][:]

        # Remap img_names to img_filepaths (full filepath)
        img_filepaths = []
        for fp in img_names:
            fp = fp.decode('utf-8')
            full_fp = label_csv[(label_csv['mrn']==patient_id) & (label_csv['img_name'].str.contains(fp))]['img_name'].values[0]
            img_filepaths.append(full_fp)
        assert len(img_names)==len(img_filepaths)
        
        # Create a graph for the patient (This is a simplified example; modify as per your data)
        G = nx.Graph()
        # print('Adding nodes...')
        for i in range(len(img_filepaths)):
            feature_str = json.dumps(features[i].tolist())  # Convert the NumPy array to a JSON string
            G.add_node(i, features=feature_str, name=img_filepaths[i])  # Add nodes with feature strings
        
        # Calculate image similarity scores and add edges based on the threshold
        similarity_matrix = image_similarity(dataroot, img_filepaths)
        max_value = np.max(similarity_matrix[similarity_matrix != 1]) # Calculate max SSIM to normalize other SSIMs
        threshold = np.median((similarity_matrix[similarity_matrix != 1])/max_value) # Median normalized SSIM
        # print(f"Adding edges based on threshold ({threshold})...")
        for i in range(len(img_filepaths)):
            for j in range(i + 1, len(img_filepaths)):
                if similarity_matrix[i, j]/max_value >= threshold:
                    G.add_edge(i, j)  # Add an edge between nodes i and j
        print('Graph constructed')

        # save Graph as a file
        nx.write_graphml(G, os.path.join(output_path,f'{patient_id}.graphml'))
        print('Graph saved')

def make_feature_similarity_graph(patient_id: str, output_path: str, input_features: str, label_csv: pd.DataFrame) -> None:
    # Load h5py file for each patient
    with h5py.File(os.path.join(args.input_features,f'{patient_id}.h5'), 'r') as file:
        # Load features for the patient
        features = file['features'][:]
        img_names = file['img_names'][:]

        # Remap img_names to img_filepaths (full filepath)
        img_filepaths = []
        for fp in img_names:
            fp = fp.decode('utf-8')
            full_fp = label_csv[(label_csv['mrn']==patient_id) & (label_csv['img_name'].str.contains(fp))]['img_name'].values[0]
            img_filepaths.append(full_fp)
        assert len(img_names)==len(img_filepaths)
        
        # Create a graph for the patient (This is a simplified example; modify as per your data)
        G = nx.Graph()
        for i in range(len(features)):
            feature_str = json.dumps(features[i].tolist())  # Convert the NumPy array to a JSON string
            G.add_node(i, features=feature_str, name=str(img_filepaths[i]))  # Add nodes with feature strings
            # print(f'Added node {i}')
        
        # Calculate similarity scores and add edges based on the threshold
        similarity_matrix = cosine_similarity(features)
        threshold = 0.98
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                if similarity_matrix[i, j] > threshold:
                    G.add_edge(i, j)  # Add an edge between nodes i and j
                    # print(f'Added edge ({i}, {j})')
        print('Graph constructed')

        # save Graph as a file
        nx.write_graphml(G, os.path.join(output_path,f'{patient_id}.graphml'))
        print('Graph saved')

def main(args: argparse.ArgumentParser) -> None:
    print("Constructing and saving graphs")
    construct_graphs(args)

if __name__=='__main__':
    args = argument_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    main(args)

