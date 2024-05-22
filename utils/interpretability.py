import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Tuple
import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils.dataset_raw import ImageDatasetFromSplit
from scipy import stats
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

sys.path.append('/raid/aradhachandran/thyroid/code/classification/BE223A_thyroidProj')
from models.mil import MIL, AMIL, PAMIL, PMIL
sys.path.append('/raid/aradhachandran/thyroid/code/classification/BE223A_thyroidProj')
from models.gcn import *
sys.path.append('/raid/aradhachandran/thyroid/code/classification/BE223A_thyroidProj')
from models.baselines import *
from models.framepatch_model import FramePatchParallel
from models.patchpatch_model import PatchPatchParallel
from models.resnet import resnet50custom
from models.semi_supervised import mean_teacher

# # GCN + SAGPooling
# class SAG_GCN(torch.nn.Module):
#     def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
#         super(SAG_GCN, self).__init__()
#         self.pooling_ratio = 0.75
#         self.dropout_ratio = 0.5
#         self.gnn = GraphConv # GraphConv, GATConv, GCNConv
        
#         self.conv1 = GCNConv(input_dim, hidden_dim)
#         self.pool1 = SAGPooling(hidden_dim, ratio=self.pooling_ratio, GNN=self.gnn)
#         self.conv2 = GCNConv(hidden_dim, hidden_dim)
#         self.pool2 = SAGPooling(hidden_dim, ratio=self.pooling_ratio, GNN=self.gnn)

#         self.lin1 = torch.nn.Linear(hidden_dim*2, hidden_dim)
#         self.lin2 = torch.nn.Linear(hidden_dim, output_dim)
#         self.lin3 = torch.nn.Linear(output_dim, 2)

#         # Initialize weights
#         initialize_gnn_weights(self)
#         initialize_weights(self)

#     def forward(self, input_data: torch.Tensor, device: str='cuda', return_attention: bool = False, return_features: bool = False):
#         x = input_data['frame_features']['x'].squeeze()
#         edge_index = input_data['frame_features']['edge_index']
#         batch = input_data['frame_features']['batch']

#         x = F.relu(self.conv1(x, edge_index))
#         x, edge_index, _, batch, _, A = self.pool1(x, edge_index, None, batch)
#         x1 = torch.cat([global_max_pool(x.cpu(), batch.cpu()).to(device), global_mean_pool(x, batch)], dim=1)

#         x = F.relu(self.conv2(x, edge_index))
#         x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
#         x2 = torch.cat([global_max_pool(x.cpu(), batch.cpu()).to(device), global_mean_pool(x, batch)], dim=1)
        
#         x = x1 + x2

#         x = F.relu(self.lin1(x))
#         x = F.dropout(x, p=self.dropout_ratio, training=self.training)
#         x = F.relu(self.lin2(x))
#         logits = self.lin3(x)
#         y_prob = F.softmax(logits, dim=-1)
#         if return_attention:
#             print(y_prob[:,1])
#             return A

#         return logits, y_prob

def init_seg_model(args: argparse.ArgumentParser) -> torch.nn.Module:
    print('Creating segmentation model')

    #for example
    model = mean_teacher()
    #load parameter
    model.eval()

    return model

def init_cls_model(args: argparse.ArgumentParser, split: int) -> torch.nn.Module:
    #initialize models
    if args.model_type == 'mil':
      #no longer used
        if args.input_type == 'patch':
            cls_model = PMIL(tile_size = args.tile_size[0])
        elif args.input_type == 'frame':
            cls_model = MIL() #or MIL
    elif args.model_type == 'amil':
        if args.input_type == 'patch':
            cls_model = PAMIL(tile_size = args.tile_size[0], use_clinical = args.use_clinical)
        elif args.input_type == 'combine_patch':
            cls_model = PatchPatchParallel(tile_size = args.tile_size, use_clinical = args.use_clinical)
        elif args.input_type == 'combine':
            cls_model = FramePatchParallel(tile_size = args.tile_size[0], use_clinical = args.use_clinical)
        elif args.input_type == 'frame':
            cls_model = AMIL(use_clinical = args.use_clinical) #or MIL
    elif args.input_type == 'graph':
        if args.model_type == 'gcn':
            if args.edge_method == 'patch_loc':
                cls_model = GCN(input_dim=2048, hidden_dim=256, output_dim=128) # input is ResNet50 features
            else:
                cls_model = GCN(input_dim=6016, hidden_dim=256, output_dim=128) # input is full frame ThyNet features
        elif args.model_type == 'gat':
            if args.edge_method == 'patch_loc':
                cls_model = GAT(input_dim=2048, hidden_dim=256, output_dim=128) # input is ResNet50 features
            else:
                cls_model = GAT(input_dim=6016, hidden_dim=256, output_dim=128) # input is full frame ThyNet features
        elif args.model_type == 'sag_gcn':
            if args.edge_method == 'patch_loc':
                # cls_model = SAG_GCN(input_dim=2048, hidden_dim=256, output_dim=128) # input is ResNet50 features
                cls_model = SAG_GCN(
                    input_dim=2048, hidden_dim=args.hidden_dim, output_dim=128,
                    pooling_ratio=args.pooling_ratio, n_layers=args.n_layers, gnn=args.gnn)
            else:
                cls_model = SAG_GCN(input_dim=6016, hidden_dim=256, output_dim=128) # input is full frame ThyNet features
        elif args.model_type == 'sag_gat':
            cls_model = SAG_GAT(input_dim=2048, hidden_dim=256, output_dim=128)
    elif args.model_type=='wang':
        cls_model = WangModel()
    else:
        raise NotImplementedError

    print(f'Load pre-trained model')
    best_model_weights = torch.load(os.path.join(args.result_dir, f"split_{split}_ckpt.pt"))
    cls_model.load_state_dict(best_model_weights)

    #load parameter
    cls_model.eval()

    return cls_model

class Ensemble(nn.Module):
    def __init__(self, resnet: nn.Module, mil: nn.Module) -> None:
        super(Ensemble, self).__init__()
        self.resnet = resnet
        self.mil = mil
        
    def forward(self, x: torch.Tensor, return_attention: bool =False) -> torch.Tensor:
        features = self.resnet(x)
        #import pdb;pdb.set_trace()
        logits,y_prob, A  = self.mil(features)
        if return_attention:
            return A
        return logits

def init_gradcam_model(args: argparse.ArgumentParser, split: int) -> nn.Module:
    mil = init_cls_model(args,split)
    resnet = resnet50custom()
    model = Ensemble(resnet,mil)
    model.eval()
    print(model)
    return model

def AxGradCam(model: torch.nn.Module, val_loader: int,args: argparse.ArgumentParser, split: int) -> None:
    print('Running GradCam')

    for batch_idx, (mrn, imgs, imgs_name) in enumerate(tqdm(val_loader)):
        imgs = imgs.squeeze()
        mrn = mrn[0]
        mrn_heatmap_dir = os.path.join(args.eval_dir, f'split{split}_heatmap', str(mrn))
        os.makedirs(mrn_heatmap_dir,exist_ok=True)

        #gradcam
        target_layers = [model.resnet.layer4[-1]]
        targets = [ClassifierOutputTarget(1)]
        with GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=False) as cam:
            grayscale_cam = cam(input_tensor=imgs, targets = targets)

        #attention:
        with torch.no_grad():
            A = model(imgs,return_attention =True).detach().numpy().squeeze()
            A_norm =  (A-min(A))/(max(A)-min(A))
            if np.all(A_norm ==np.nan):
                import pdb;pdb.set_trace()
        with open(os.path.join(mrn_heatmap_dir,'attention_scores.npy'), 'wb') as f:
            np.save(f, A_norm)
        
        for i in range(imgs.shape[0]):
            img = imgs[i].detach().numpy()
            img_norm = (img - np.min(img)) / (np.max(img)-np.min(img))
            img_norm_t = np.transpose(img_norm, (1,2,0))

            if args.save_raw:
                cam_input = grayscale_cam
            else:
                #import pdb;pdb.set_trace()
                cam_input = grayscale_cam * A_norm.reshape(-1,1,1) #gradcam * attention
                cam_input = (cam_input - np.min(cam_input)) / (np.max(cam_input)-np.min(cam_input))
                

            overlayed_heatmap = show_cam_on_image(img_norm_t, cam_input[i], use_rgb=True)
            overlayed_heatmap = Image.fromarray(overlayed_heatmap)
            overlayed_heatmap.save(os.path.join(mrn_heatmap_dir,f'{imgs_name[i]}.jpg'))
        break

def get_img_heat_mask(csv_file, dataroot, mrn, images_name, sub_A, image_coords, tile_size):
    imgs = []
    heatmaps = []
    masks = []
    img_names = []
    for ind in range(len(sub_A)):
        
        #find the path for the img
        all_imgs_ind = csv_file[csv_file.mrn == mrn[0]].img_name.values
        img_ind_path = all_imgs_ind[[images_name[ind][0] in img_ind for img_ind in all_imgs_ind]]
        img_names.append(images_name[ind][0])
        img_ind_path = os.path.join(dataroot,img_ind_path.item())
        
        img = np.load(img_ind_path)
        imgs.append(img)
        
        #coords and scoors for this specific img
        scores = sub_A[ind]
        coords = image_coords[ind].numpy().squeeze()[:,[1,0]]

        # Create a heatmap
        heatmap = np.zeros_like(img, dtype=float)
        for i, coord in enumerate(coords):
            x, y = coord
            heatmap[y:y+tile_size, x:x+tile_size] += scores[i]

        heatmap = gaussian_filter(heatmap, sigma=10)
        heatmaps.append(heatmap)
        #print(heatmap.min(),heatmap.max())
        
        #if mask exists then append to masks
        # if os.path.isfile(img_ind_path[:-3]+'mask.npy'):
        #     mask = np.load(img_ind_path[:-3]+'mask.npy')
        #     masks.append(mask)
        # else:
        masks.append(None)

    heatmap_min = np.min([np.min(i) for i in heatmaps])
    heatmap_max = np.max([np.max(i) for i in heatmaps])
    heatmaps = [(h - heatmap_min) / (heatmap_max - heatmap_min) for h in heatmaps]
    return zip(imgs,heatmaps,masks, img_names)

def plot_img_heat_mask(zipped_imgs, mrn_heatmap_dir,tile_size):
    cmap = mcolors.LinearSegmentedColormap.from_list("", [(1, 0, 0, 0), (1, 0, 0, 1)])

    for img, heatmap, mask, img_name in zipped_imgs:
        print(img.shape)
        # img = (img - np.min(img)) / (np.max(img) - np.min(img))
        fig, ax = plt.subplots(figsize = (5,5))
        img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_GRAY2BGR)
        if mask is not None:
            contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, (255, 255, 0), thickness=2)
        ax.imshow(img)
        print('heatmap: ', np.min(heatmap), np.max(heatmap))
        ax.imshow(heatmap, cmap=cmap, alpha=0.5, vmin = 0, vmax = 1, interpolation='bicubic')

        #remove axis
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])


        plt.savefig(os.path.join(mrn_heatmap_dir,f'{img_name}_{tile_size}.jpg'))
        plt.close()

def plot_patch_heatmap(args, cls_model, val_dataset):
    val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False, num_workers = 4
        )
    csv_file = val_dataset.original_csv

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cls_model.to(device)
    if args.num is None:
        args.num = len(val_dataset)

    num = 0
    with torch.no_grad():
        for batch_idx, (mrn, image_patches, image_coords, images_name, image_shapes,clinical , label) in enumerate(val_loader):
            
            print(str(mrn[0]))
            mrn_heatmap_dir = os.path.join(args.eval_dir, f'split{val_dataset.split}_heatmap', label[0] , str(mrn[0]))

            #check if we have created heatmap for this patient
            if os.path.exists(mrn_heatmap_dir):
                if len(os.listdir(mrn_heatmap_dir)) > 0:
                    print('heatmap generated for', str(mrn[0]),'--pass')
                    continue
                else:
                    os.makedirs(mrn_heatmap_dir, exist_ok = True)
            else:
                os.makedirs(mrn_heatmap_dir)

            #combine patch features
            if len(args.tile_size) == 2:
                patch_features = []
                for i_tile in image_patches:
                    feature_tile_i = []
                    for p in i_tile:
                        feature_tile_i.append(p.squeeze(0))
                    patch_features.append(torch.cat(feature_tile_i).to(device))                
            else:
                patch_features =  torch.cat([p.squeeze(0) for p in image_patches[0]]).to(device)
            
            input_data = {'patch_features':patch_features, 'clinical':clinical.squeeze(0).to(device) }
            As = cls_model(input_data, return_attention = True)

            # standardize patch and patch combine
            if len(args.tile_size) == 1:
                As = [As]

            #loop through all As (patch combine and patch)
            for ith_A, A in enumerate(As):
                A = A.detach().cpu()

                # Compute the percentiles
                ranks = stats.rankdata(A)
                percentiles = ranks / len(A) * 100
                
                #find the ind of each new img
                patch_lens = [len(p.squeeze(0)) for p in image_patches[ith_A]]
                cum_patch_lens = np.cumsum(patch_lens)
                
                #how to calculate attention
                #divide attentions for each img
                if args.attention_type == 'percentile':
                    norm_percentiles =  (percentiles - percentiles.min()) / (percentiles.max() - percentiles.min())
                    sub_A = [norm_percentiles[i:j] for i, j in zip([0] + cum_patch_lens.tolist(), cum_patch_lens.tolist() + [None])][:-1]
                elif args.attention_type == 'raw':
                    normalized_A = (A - A.min()) / (A.max() - A.min()).numpy()
                    sub_A = [normalized_A[i:j] for i, j in zip([0] + cum_patch_lens.tolist(), cum_patch_lens.tolist() + [None])][:-1]

                #get images
                zipped_imgs = get_img_heat_mask(csv_file, args.raw_dataroot, mrn, images_name[ith_A], sub_A, image_coords[ith_A],args.tile_size[ith_A])
                
                #plot heatmap 
                plot_img_heat_mask(zipped_imgs, mrn_heatmap_dir, args.tile_size[ith_A])

            num+=1
            if num == args.num:
                break

def get_attention(cls_model, val_loader):
    with torch.no_grad():
        for batch_idx, (mrn, frame_features, patch_features, patch_lens, clinical, target) in enumerate(val_loader):
            frame_features = frame_features.to(device)
            if isinstance(patch_features,list):
                patch_features = [patch_feature.to(device) for patch_feature in patch_features]
            else:
                patch_features = patch_features.to(device)
            clinical = clinical.to(device)
            target = torch.tensor([target])
            target = target.to(device)
            input_data = {'frame_features':frame_features, 'patch_features':patch_features,
            'clinical':clinical, 'patch_lens':patch_lens}
            A = cls_model(input_data, return_attention = True)
    
    return A

