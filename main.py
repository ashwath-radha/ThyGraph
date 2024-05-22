import argparse
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyG_DataLoader

import pandas as pd
import os
import sys
sys.path.append('/raid/aradhachandran/thyroid/code/classification/BE223A_thyroidProj')
from models.mil import *
from models.semi_supervised import *
from utils.dataset_bags import *
from utils.dataset_patches import *
from utils.dataset_graphs import *
from utils.dataset_wang import *
from utils.train import *
from utils.utils import *

def argument_parser() -> argparse.ArgumentParser:
    print("Running argument parser")
    parser = argparse.ArgumentParser(description="Training script")

    parser.add_argument(
        "--output", type=str, default="/raid/aradhachandran/thyroid/code/classification/BE223A_thyroidProj/results", help="path to output directory"
    )
    parser.add_argument(
        "--exp", type=str, default="temp", help="name of the experiment"
    )
    parser.add_argument(
        "--dataroot",
        type=str,
        default="/radraid/aradhachandran/thyroid/data/deepstorage_data/gcn_us",
        help="path to data",
    )
    parser.add_argument(
      '--featuresroot', 
      type=str, 
      default='/raid/aradhachandran/thyroid/code/classification/BE223A_thyroidProj/iodata/bags/files_miccai_cv3',
      help='path to h5 features')

    parser.add_argument(
        "--csv_file",
        type=str,
        default="/raid/aradhachandran/thyroid/code/classification/BE223A_thyroidProj/iodata/bags/label_miccai_cv3_noIndet.csv",
        help="full path to csv file",
    )
    parser.add_argument(
        "--splitroot", type=str, default="/raid/aradhachandran/thyroid/code/classification/BE223A_thyroidProj/iodata/splits/miccai_cv3", help="path to splits"
    )
    parser.add_argument("--log", action="store_true", help="tensorboard log")
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="which gpu you want to use",
    )
    # wang model parametesr
    parser.add_argument(
        "--squiggly_n", 
        type=int, 
        default=None, 
        help="standard bag size for wang model"
    )
    # train parameters
    parser.add_argument(
        "--need_patch_nodule_overlap",
        action="store_true",
        help="whether patches need to have nodule overlap",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="amil",
        choices=["mil", "amil","gcn","gat","sag_gcn","sag_gat","wang"],
        help="model type",
    )
    parser.add_argument(
        "--input_type",
        type=str,
        default="amil",
        choices=["frame", "patch", 'combine', 'combine_patch',"graph","wang_frame"],
        help="input type will determine the model that is initiated",
    )
    parser.add_argument(
        "--edge_method",
        type=str,
        default="None",
        choices=["feature_sim" , "image_ssim", "image_loc", "patch_loc"],
        help="edge_method picks specific graphs constructed"
    )
    parser.add_argument(
        "--semi_supervised",
        action="store_true",
        help="whether you want to train semi-supervised model",
    )
    parser.add_argument(
        "--use_clinical", 
        action="store_true", 
        help="whether to use clinical features (sex and age)"
    )
    parser.add_argument(
        "--include_tirads", 
        action="store_true", 
        help="whether to use tirads in addition to the clinical features"
    )
    parser.add_argument(
        "--tile_size", 
        nargs='+',
        type=int, 
        default=64, 
        help="dimension of each tile"
    )
    parser.add_argument(
        "--stride_size",
        nargs='+',
        type=int,
        default=64,
        help="the number of pixels in between each tile",
    )
    parser.add_argument(
        "--n_splits", type=int, default=5, help="number of splits for cross-validation"
    )
    parser.add_argument(
        "--opt", type=str, choices=["adam", "sgd", "wang_sgd"], default="adam", help="optimizer"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="optimizer - learning rate"
    )
    parser.add_argument(
        "--wd", type=float, default=0, help="optimizer - weight decay (L2)"
    )
    parser.add_argument(
        "--weighted",
        action="store_true",
        help="whether to weight by train labels (weighted sampling)",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=["ce", "wce", "focal", "wang_loss"],
        default = 'wce',
        help="loss function to use. ce - cross entropy loss; wce - weighted cross entropy loss; focal - focal loss",
    )
    parser.add_argument(
        "--max_augments",
        type=int,
        default=0,
        help="highest number of augmentations to take from csv_file if they exist",
    )
    parser.add_argument(
        "--seed", type=int, default=224, help="seed for reproducibility"
    )

    parser.add_argument(
        "--epochs", type=int, default=200, help="number of epochs for training"
    )

    # bash script - SAG_GCN optimizing
    parser.add_argument('--hidden_dim', type=int, help='Hidden dimension')
    parser.add_argument('--pooling_ratio', nargs='+', type=str, help='Pooling ratio (list)')
    parser.add_argument('--n_layers', type=int, help='Number of layers')
    parser.add_argument('--gnn', type=str, help='GNN type')

    args = parser.parse_args()
    return args

def main(args: argparse.ArgumentParser) -> None:
    if not args.exp.startswith('wang'):
        assert args.splitroot.split('/')[-1] in args.exp
        assert args.splitroot.split('/')[-1] in args.csv_file

    print("Running train script")
    set_seed(args)
    # create result directory for the experiment
    args.result_dir = os.path.join(args.output, args.exp)
    os.makedirs(args.result_dir, exist_ok=True)

    accs = []
    aurocs = []
    auprcs = []
    ious = []

    for i in range(args.n_splits):
        print(f"\nFold {i} - {args.exp}")
        # semi_supervised pipeline
        if args.semi_supervised:
            pass
            """
            train_dataset = SegmentationBagDataset(args.dataroot, args.splitroot, i, mode='train')
            val_dataset = SegmentationBagDataset(args.dataroot, args.splitroot, i, mode='test')

            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

            mean_iou, val_auc = train_semisupervised(args, train_loader, val_loader)

            ious.append(mean_iou)
            aucs.append(val_auc)
            """
        else:
            if args.input_type == 'patch' or args.input_type == 'combine_patch':
                train_dataset = PatchBagDataset(
                    args.dataroot,
                    None,
                    args.csv_file,
                    args.splitroot,
                    i,
                    tile_size=args.tile_size,
                    stride_size=args.stride_size,
                    mode="train",
                    max_augments=args.max_augments,
                    include_tirads = args.include_tirads,
                    need_patch_nodule_overlap = args.need_patch_nodule_overlap
                )
                val_dataset = PatchBagDataset(
                    args.dataroot,
                    None,
                    args.csv_file,
                    args.splitroot,
                    i,
                    tile_size=args.tile_size,
                    stride_size=args.stride_size,
                    mode="test",
                    max_augments=args.max_augments,
                    include_tirads = args.include_tirads,
                    need_patch_nodule_overlap = args.need_patch_nodule_overlap
                )
            elif args.input_type == 'combine':
                train_dataset = PatchBagDataset(
                    args.dataroot, 
                    args.featuresroot, 
                    args.csv_file, 
                    args.splitroot, 
                    i, 
                    tile_size=args.tile_size, 
                    stride_size=args.stride_size, 
                    mode='train',
                    max_augments=args.max_augments,
                    include_tirads = args.include_tirads,
                    need_patch_nodule_overlap = args.need_patch_nodule_overlap
                )
                val_dataset = PatchBagDataset(
                    args.dataroot, 
                    args.featuresroot, 
                    args.csv_file, 
                    args.splitroot,
                    i, 
                    tile_size=args.tile_size, 
                    stride_size=args.stride_size, 
                    mode='test',
                    max_augments=args.max_augments,
                    include_tirads = args.include_tirads,
                    need_patch_nodule_overlap = args.need_patch_nodule_overlap
                )
            elif args.input_type == 'frame':
                train_dataset = BagDataset(
                    args.featuresroot,
                    args.csv_file,
                    args.splitroot,
                    i,
                    mode="train",
                    max_augments=args.max_augments,
                    include_tirads = args.include_tirads
                )
                val_dataset = BagDataset(
                    args.featuresroot,
                    args.csv_file,
                    args.splitroot,
                    i,
                    mode="test",
                    max_augments=args.max_augments,
                    include_tirads = args.include_tirads
                )
            elif args.input_type == 'graph':
                train_dataset = GraphDataset(
                    args.featuresroot.replace('/files_','/gcn_files_') + f'_{args.edge_method}',
                    args.csv_file,
                    args.splitroot,
                    i,
                    mode="train",
                    max_augments=args.max_augments,
                    include_tirads = args.include_tirads
                )
                val_dataset = GraphDataset(
                    args.featuresroot.replace('/files_','/gcn_files_') + f'_{args.edge_method}',
                    args.csv_file,
                    args.splitroot,
                    i,
                    mode="test",
                    max_augments=args.max_augments,
                    include_tirads = args.include_tirads
                )
            elif args.input_type == 'wang_frame':
                train_dataset = WangDataset(
                    args.dataroot,
                    args.csv_file,
                    args.splitroot,
                    split=i,
                    mode="train",
                    squiggly_n=args.squiggly_n
                )
                val_dataset = WangDataset(
                    args.dataroot,
                    args.csv_file,
                    args.splitroot,
                    split=i,
                    mode="test",
                    squiggly_n=args.squiggly_n
                )

            class_weights = make_weights_for_balanced_classes(
                train_dataset, return_weights=True
            )
            set_seed(args)
            if args.weighted:
                sampler = make_weights_for_balanced_classes(train_dataset)
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=1,
                    collate_fn=collate_features,
                    sampler=sampler, num_workers = 4
                )
            else:
                if args.input_type=='graph':
                    train_loader = PyG_DataLoader(
                        train_dataset,
                        batch_size=8,
                        shuffle=True,
                        collate_fn=collate_features_graph,
                        num_workers = 16, 
                        pin_memory=True
                    )
                elif args.input_type=='wang_frame':
                    train_loader = DataLoader(
                        train_dataset,
                        batch_size=2, # from paper
                        shuffle=True,
                        pin_memory=True,
                        num_workers = 8,
                        collate_fn=lambda batch: collate_features_wang(batch, args.squiggly_n)
                    )
                else:
                    train_loader = DataLoader(
                        train_dataset,
                        batch_size=1,
                        shuffle=True,
                        collate_fn=collate_features, 
                        num_workers = 8,
                        pin_memory=True
                    )
            set_seed(args)
            if args.input_type=='graph':
                val_loader = PyG_DataLoader(
                    val_dataset, batch_size=1, shuffle=False, collate_fn=collate_features_graph, num_workers = 8, pin_memory=True
                )
            elif args.input_type=='wang_frame':
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=1,
                    shuffle=True,
                    pin_memory=True,
                    num_workers=8,
                    collate_fn=lambda batch: collate_features_wang(batch, args.squiggly_n)
                )
            else:
                val_loader = DataLoader(
                    val_dataset, batch_size=1, shuffle=False, collate_fn=collate_features, num_workers = 8, pin_memory=True
                )
            
            if args.input_type=='graph':
                val_acc, val_auroc, val_auprc = train_graph(
                    args, i, train_loader, val_loader, class_weights
                )
            elif args.input_type=='wang_frame':
                val_acc, val_auroc, val_auprc = train_wang(
                    args, i, train_loader, val_loader, class_weights
                )
            else:
                val_acc, val_auroc, val_auprc = train(
                    args, i, train_loader, val_loader, class_weights
                )
            accs.append(val_acc)
            aurocs.append(val_auroc)
            auprcs.append(val_auprc)

    # save summary result
    summary = pd.DataFrame(
        {
            "folds": range(args.n_splits),
            "Accuracy": accs,
            "AUROC": aurocs,
            "AUPRC": auprcs,
        }
    )
    summary_file_path = os.path.join(args.result_dir, "summary.csv")
    summary.to_csv(summary_file_path, index=False)
    print(f"\nTraining ends. Performance result saved to {summary_file_path}")

    # save arguments for evaluation
    args_file_path = os.path.join(args.result_dir, "args.txt")
    with open(args_file_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)
    print(f"Arugments saved to {args_file_path}")
    print("End script.")

if __name__ == "__main__":
    args = argument_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    main(args)
