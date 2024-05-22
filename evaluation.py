import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import os
from sklearn.metrics import roc_auc_score
import argparse
import json

from models.mil import *
from models.semi_supervised import *
from utils.dataset_raw import *
from utils.dataset_bags import *
from utils.dataset_patches import *
from utils.dataset_graphs import *
from utils.interpretability import *
from utils.utils import * 
from utils.train import *


def argument_parser() -> argparse.ArgumentParser:
    print('Running argument parser')

    parser = argparse.ArgumentParser(description='Evaluation script')
    # experiment
    parser.add_argument('--raw_dataroot', type=str, default='/radraid/aradhachandran/thyroid/data/deepstorage_data/gcn_us', help='path to directory with all images')
    parser.add_argument('--output', type=str, default='/raid/aradhachandran/thyroid/code/classification/BE223A_thyroidProj/results', help='path to output directory')
    parser.add_argument('--exp', type=str, default='temp', help='name of the experiment')
    parser.add_argument('--eval', type=str, default='eval', help='name of the evaluation')
    parser.add_argument('--mode', type=str, default='holdout', help='name of the patient set to use') # could also be "test"
    parser.add_argument( "--csv_file",type=str,default="/raid/aradhachandran/thyroid/code/classification/BE223A_thyroidProj/iodata/bags/label_noIndet.csv",help="full path to csv file")
    # validation
    parser.add_argument('--stanford', action='store_true', help='to run evaluation on stanford EV dataset')
    parser.add_argument('--heatmap', action='store_true', help='return only attention and heatmaps without \
                                                                    predictions and probabilties during validation')
    parser.add_argument( "--num",type=int,default= None , help="number of patients")
	
    parser.add_argument('--save_raw', action='store_true', help='save raw saliency map (without multiplying attention scores)')
    parser.add_argument('--attention_type', type=str, default='percentile', choices = ['percentile','attention'],help='attention score - raw or percentile')
    parser.add_argument('--stride_div', type=int, default=1, help='stride divided by certain number to generate better heatmap')
    parser.add_argument(
        "--gpu",
        type=str,
        default="8",
        help="which gpu you want to use",
    )
    args = parser.parse_args()
    args_original = vars(args)
    with open(os.path.join(args.output,args.exp,'args.txt'), 'r') as f:
        args.__dict__ = json.load(f)

    #update validation parameters
    args.stanford = args_original['stanford']
    if args.stanford:
        args.eval_dir = os.path.join(args_original['output'],args_original['exp'],'stanford')
        args.splitroot = args.splitroot.replace('_'.join(args_original['exp'].split('_')[2:]), 'stanford')
        args.featuresroot='/raid/aradhachandran/thyroid/code/classification/BE223A_thyroidProj/iodata/bags/files_stanford'
        args.dataroot='/radraid/aradhachandran/thyroid/data/deepstorage_data/Stanford_AIMI_data/processed/'
    else:
        args.eval_dir = os.path.join(args_original['output'],args_original['exp'],args_original['eval'])
    args.heatmap = args_original['heatmap']
    if args.heatmap:
        args.stride_size = int(args.stride_size[0]/args_original['stride_div'])
    args.save_raw = args_original['save_raw']
    args.attention_type = args_original['attention_type']
    args.raw_dataroot = args_original['raw_dataroot']
    args.csv_file = args_original['csv_file']
    args.num = args_original['num']
    args.mode = args_original['mode']
    args.result_dir = os.path.join(args_original['output'],args_original['exp'])
    if not isinstance(args.tile_size,list):
        args.tile_size = [args.tile_size]
    if not isinstance(args.stride_size,list):
        args.stride_size = [args.stride_size]
    # print(args)
    return args

def main(args: argparse.ArgumentParser) -> None:

    print('Running evaluation script')
    os.makedirs(args.eval_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    summary_file_path = os.path.join(args.eval_dir, 'summary.csv')

    accs = []
    aurocs = []
    auprcs = []
    #ious = []
    for i in range(args.n_splits):
        print(f'\nFold {i}')
        if args.heatmap:
            if args.input_type == 'frame':
                print(f'-------Creating heatmap for split {i}--------')
                val_dataset = ImageDatasetFromSplit(args.raw_dataroot, args.csv_file, args.splitroot, i, args.mode)

                gradcam_model = init_gradcam_model(args, i)
                AxGradCam(gradcam_model, val_dataset,args, i)
            elif args.input_type == 'patch' or args.input_type == 'combine_patch':
                print(f'-------Creating heatmap for split {i}--------')
                val_dataset = PatchDatasetFromSplit(
                    args.dataroot,
                    args.csv_file,
                    args.splitroot,
                    i,
                    args.mode,
                    tile_size=args.tile_size,
                    stride_size=args.stride_size,
                )
                cls_model = init_cls_model(args, i)
                A = plot_patch_heatmap(args, cls_model, val_dataset)

        else:
            # if os.path.isfile(summary_file_path):
            #     raise NotImplementedError(f"{summary_file_path} exists")

            if args.input_type == 'patch' or args.input_type == 'combine_patch':
                val_dataset = PatchBagDataset(
                    args.dataroot,
                    None,
                    args.csv_file,
                    args.splitroot,
                    i,
                    tile_size=args.tile_size,
                    stride_size=args.stride_size,
                    mode=args.mode,
                    max_augments=args.max_augments,
                    stanford=args.stanford,
                    need_patch_nodule_overlap = args.need_patch_nodule_overlap
                )
            elif args.input_type == 'combine':
                val_dataset = PatchBagDataset(
                    args.dataroot, 
                    args.featuresroot, 
                    args.csv_file, 
                    args.splitroot,
                    i, 
                    tile_size=args.tile_size, 
                    stride_size=args.stride_size, 
                    mode=args.mode,
                    max_augments=args.max_augments,
                    need_patch_nodule_overlap = args.need_patch_nodule_overlap
                )
            # mil pipeline
            elif args.input_type == 'frame':
                print(args)
                val_dataset = BagDataset(
                    args.featuresroot,
                    args.csv_file,
                    args.splitroot,
                    i,
                    mode=args.mode,
                    max_augments=args.max_augments,
                    stanford=args.stanford
                )
            elif args.input_type == 'graph':
                val_dataset = GraphDataset(
                    args.featuresroot.replace('/files_','/gcn_files_') + f'_{args.edge_method}',
                    args.csv_file,
                    args.splitroot,
                    i,
                    mode=args.mode,
                    max_augments=args.max_augments,
                    include_tirads = args.include_tirads
                )
            set_seed(args)
            val_loader = DataLoader(
                val_dataset, batch_size=1, shuffle=False, collate_fn=collate_features, num_workers = 4
            )

            cls_model = init_cls_model(args, i)
            cls_model.to(device)
            print(cls_model)

            if args.input_type=='graph':
                val_acc, val_auroc, val_auprc = eval_graph(args, i, 'final', val_loader, cls_model, save_dict=True, evaluation = True, class_weights = torch.tensor([1,1]).float())
            else:
                val_acc, val_auroc, val_auprc = eval(args, i, 'final', val_loader, cls_model, save_dict=True, evaluation = True, class_weights = torch.tensor([1,1]).float())

            accs.append(val_acc)
            aurocs.append(val_auroc)
            auprcs.append(val_auprc)

    # save metrics for each splits
    if not args.heatmap:
        summary = pd.DataFrame({'folds':range(args.n_splits), 
            'Accuracy': accs,
            'AUROC': aurocs, 'AUPRC': auprcs})
        summary.to_csv(summary_file_path, index = False)
        print(f'\nEvaluation ends. Performance result saved to {summary_file_path}')

    #save arguments for evaluation
    args_file_path = os.path.join(args.eval_dir,'args.txt')
    with open(args_file_path,'w') as f:
        json.dump(args.__dict__, f, indent =2)
    print(f'Arugments saved to {args_file_path}')
    print('End script.')

if __name__ == '__main__':
    args = argument_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    main(args)