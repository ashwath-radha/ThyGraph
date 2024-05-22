import torch_geometric
from torch_geometric.data import Dataset, Data
import torchvision.transforms
import torch
from typing import Tuple, List
import pandas as pd
import os
import h5py
import numpy as np
import ast
import networkx as nx
from sklearn.preprocessing import StandardScaler
from utils.extract_labels import convert_to_string
import random
from torch_geometric.data import Batch

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

def collate_features_graph(batch):
    mrn, data, label = zip(*batch)
    
    # Merge node features (x) and edge information (edge_index)
    x = torch.cat([item.x for item in data], dim=0)
    edge_index = torch.cat([item.edge_index for item in data], dim=1)
    data = Data(x=x, edge_index=edge_index)
    
    # Stack labels along a new dimension
    label = torch.stack(label, dim=0)
    
    return [list(mrn), data, label]

class GraphDataset(Dataset):
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
    ):
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
                converters={"holdout": str}
            )
        else:
            splits = pd.read_csv(
                os.path.join(splitroot, f"cv{split}.csv"),
                converters={"train": str, "test": str},
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
        
        self.pyg_data_dir = os.path.join(self.dataroot, 'pyg_data')
        if not os.path.exists(self.pyg_data_dir):
            os.makedirs(self.pyg_data_dir)
            print(f"Directory '{self.pyg_data_dir}' created successfully.")
        else:
            print(f"Directory '{self.pyg_data_dir}' already exists.")

    def get(self):
        pass
    def len(self):
        pass
    def __len__(self):
        return len(self.mrns)

    def __getitem__(self, idx):
        mrn = self.mrns[idx]
        path = os.path.join(self.dataroot, f"{mrn}.graphml")
        path_pt = os.path.join(self.pyg_data_dir, f"{mrn}.pt")

        if os.path.exists(path_pt):
            data = torch.load(path_pt)
        else:
            G = nx.read_graphml(path)

            # Preprocess graph and convert to PyG Data object
            adjacency_matrix = nx.to_numpy_array(G)
            node_features = [ast.literal_eval(G.nodes[i]['features']) for i in G.nodes()]
            x = torch.tensor(node_features, dtype=torch.float)#.unsqueeze(0)
            edge_list = np.array(adjacency_matrix.nonzero())
            edge_index = torch.tensor(edge_list, dtype=torch.long)
            data = Data(x=x, edge_index=edge_index)
            torch.save(data, path_pt)

        label = self.label_bags[self.label_bags.mrn == mrn].label.item()
        clinical = self.label_bags[self.label_bags.mrn == mrn][self.clinical_features].values

        return mrn, data, label #clinical, label
    
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


