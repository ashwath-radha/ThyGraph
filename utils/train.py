import argparse
from typing import Tuple
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
from torch.optim.lr_scheduler import StepLR
import datetime
import time
import sys

sys.path.append('/raid/aradhachandran/thyroid/code/classification/BE223A_thyroidProj/')
from utils.loss_function import *
from utils.utils import *
from utils.interpretability import *
from models.mil import MIL, AMIL, PAMIL, PMIL
from models.gcn import GCN, GAT
from models.baselines import WangModel
from models.framepatch_model import FramePatchParallel
from models.patchpatch_model import PatchPatchParallel

#from models.resnet import *
import os
from tensorboardX import SummaryWriter

def train(args: argparse.ArgumentParser, split, train_loader: torch.utils.data.DataLoader, val_loader:torch.utils.data.DataLoader, class_weights: torch.Tensor) ->  Tuple[float, float, float]:
    if args.log:
        log_dir = os.path.join(args.result_dir, f'split_{split}_log')
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_count = torch.cuda.device_count()
    assert device_count>=1

    #initialize models
    if args.model_type == 'mil':
      #no longer used
        if args.input_type == 'patch':
            cls_model = PMIL(tile_size = args.tile_size[0])
        elif args.input_type == 'frame':
            cls_model = MIL() #or MIL
    elif args.model_type == 'amil':
        if args.input_type == 'patch':
            cls_model = PAMIL(tile_size = args.tile_size[0], use_clinical = args.use_clinical, include_tirads = args.include_tirads)
        elif args.input_type == 'combine_patch':
            cls_model = PatchPatchParallel(tile_size = args.tile_size, use_clinical = args.use_clinical, include_tirads = args.include_tirads)
        elif args.input_type == 'combine':
            cls_model = FramePatchParallel(tile_size = args.tile_size[0], use_clinical = args.use_clinical, include_tirads = args.include_tirads)
        elif args.input_type == 'frame':
            cls_model = AMIL(use_clinical = args.use_clinical, include_tirads = args.include_tirads) #or MIL
    else:
        raise NotImplementedError
    print(cls_model)
    cls_model.to(device)
    if device == 'cuda' and device_count > 1:
        cls_model = DataParallel(cls_model)  # Wrap the model with DataParallel

    #initialize optimizer
    optimizer = get_opt(cls_model, args)

    #initialize loss functions
    if args.loss == 'ce':
        cls_loss_fun = nn.CrossEntropyLoss()
    elif args.loss == 'wce':
        class_weights = class_weights/torch.sum(class_weights)
        cls_loss_fun = nn.CrossEntropyLoss(weight = class_weights.to(device))
    elif args.loss == 'focal':
        class_weights = class_weights/torch.sum(class_weights)
        cls_loss_fun = FocalLoss(gamma = 0.75 , alpha = class_weights)

    ##initialize earlystopping
    early_stopping = EarlyStopping(warmup = 10, tolerance = 25, metric='val_auc')

    for epoch in range(args.epochs):
        print(f'\nTraining epoch {epoch}......')

        train_loss = 0
        labels = []
        y_probs = []

        cls_model.train()
        for batch_idx, (mrn, frame_features, patch_features, patch_lens, clinical, target) in enumerate(train_loader):
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
            logits,y_prob, A = cls_model(input_data)
            cls_loss = cls_loss_fun(logits,target)
            cls_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            train_loss += cls_loss

            if (batch_idx + 1) % 40 == 0:
                print(f'batch {batch_idx}; mrn {mrn.item()}; label {target.detach().cpu().numpy().item()}; loss {cls_loss: .4f}')
            if (batch_idx + 1) % 150 == 0:                
                print('Attention: ', A.flatten().unique()[:10])

            #save for metrics computation
            labels.append(target.detach().cpu().numpy().item())
            y_probs.append(y_prob[0,1].item())

        train_loss = train_loss / len(train_loader) #size of loaer
        train_acc, train_auroc, train_auprc = compute_metrics(np.array(labels),np.array(y_probs),thresh = 0.5)
        print(f'Epoch {epoch} [train]: loss{train_loss: .4f}; accuracy{train_acc: .4f}; AUROC{train_auroc: .4f}; AUPRC{train_auprc: .4f}\n')
        
        if writer is not None:
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/acc', train_acc, epoch)
            writer.add_scalar('train/auroc', train_auroc, epoch)
            writer.add_scalar('train/auprc', train_auprc, epoch)

        early_stop = eval(args, split, epoch, val_loader, cls_model, early_stopping, writer, class_weights = class_weights)
        if early_stop:
            break
    
    #load the weights of the best model
    best_model_weights = torch.load(os.path.join(args.result_dir, f"split_{split}_ckpt.pt"))
    cls_model.load_state_dict(best_model_weights)
    val_acc, val_auroc, val_auprc = eval(args, split, 'final', val_loader, cls_model, save_dict=True, class_weights = class_weights)
    if writer:
        writer.close()
    return val_acc, val_auroc, val_auprc

def eval(args: argparse.ArgumentParser, split: int, epoch: int, val_loader: torch.utils.data.DataLoader, cls_model: torch.nn.Module, 
    early_stopping: EarlyStopping = None, writer: SummaryWriter = None, save_dict: bool = False, evaluation: bool = False, 
    class_weights: torch.Tensor = None) -> Tuple[float, float, float]:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cls_model.eval()
    if args.loss == 'ce':
        cls_loss_fun = nn.CrossEntropyLoss()
    elif args.loss == 'wce':
        cls_loss_fun = nn.CrossEntropyLoss(weight = class_weights.to(device))
    elif args.loss == 'focal':
        cls_loss_fun = FocalLoss(gamma = 0.75 , alpha = class_weights)

    val_loss = 0.0
    mrns = []
    labels = []
    y_probs = []

    with torch.no_grad():
        for batch_idx, (mrn, frame_features, patch_features, patch_lens, clinical, target) in enumerate(val_loader):
            # features = features.squeeze()
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
            logits,y_prob, A = cls_model(input_data)

            if (batch_idx + 1) % 50 == 0:                
                print('Attention: ', A.flatten().unique()[:10])
            
            cls_loss = cls_loss_fun(logits,target)
            val_loss += cls_loss
            mrns.append(mrn.item())
            labels.append(target.detach().cpu().numpy().item())
            y_probs.append(y_prob[0,1].item())

    val_loss = val_loss / len(val_loader) 
    val_acc, val_auroc, val_auprc = compute_metrics(np.array(labels),np.array(y_probs),thresh = 0.5)
    print(f'Epoch {epoch} [val]: loss{val_loss: .4f}; accuracy{val_acc: .4f}; AUROC{val_auroc: .4f}; AUPRC{val_auprc: .4f}')

    if save_dict:
        split_dict = pd.DataFrame({'mrn':mrns, 'label':labels, 'prob': y_probs})
        if evaluation:
            split_dict.to_csv(os.path.join(args.eval_dir,f'split_{split}_result.csv'), index = False)
        else:            
            split_dict.to_csv(os.path.join(args.result_dir,f'split_{split}_result.csv'), index = False)

    if writer is not None:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/acc', val_acc, epoch)
        writer.add_scalar('val/auroc', val_auroc, epoch)
        writer.add_scalar('val/auprc', val_auprc, epoch)

    if early_stopping is not None:
        if early_stopping.metric=='val_loss':
            early_stopping(epoch, val_loss.cpu(), cls_model, ckpt_path = os.path.join(args.result_dir, f"split_{split}_ckpt.pt"))
        elif early_stopping.metric=='val_auc':
            early_stopping(epoch, val_auroc, cls_model, ckpt_path = os.path.join(args.result_dir, f"split_{split}_ckpt.pt"))
        elif early_stopping.metric=='val_acc':
            early_stopping(epoch, val_acc, cls_model, ckpt_path = os.path.join(args.result_dir, f"split_{split}_ckpt.pt"))

        if early_stopping.early_stop:
            print('Early Stopped!')
            return True
        else:
            return False
    else:
        return val_acc, val_auroc, val_auprc

def train_graph(args: argparse.ArgumentParser, split, train_loader: torch.utils.data.DataLoader, val_loader:torch.utils.data.DataLoader, class_weights: torch.Tensor) ->  Tuple[float, float, float]:
    if args.log:
        log_dir = os.path.join(args.result_dir, f'split_{split}_log')
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_count = torch.cuda.device_count()
    print(f"Number of GPUs available: {device_count}")
    assert device_count>=1

    # initialize models
    if args.input_type == 'graph':
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
                # cls_model = SAG_GCN(input_dim=2048, hidden_dim=256, output_dim=128)
                cls_model = SAG_GCN(
                    input_dim=2048, hidden_dim=args.hidden_dim, output_dim=128,
                    pooling_ratio=args.pooling_ratio, n_layers=args.n_layers, gnn=args.gnn) # input is ResNet50 features
            else:
                cls_model = SAG_GCN(input_dim=6016, hidden_dim=256, output_dim=128) # input is full frame ThyNet features
        elif args.model_type == 'sag_gat':
            cls_model = SAG_GAT(input_dim=2048, hidden_dim=256, output_dim=128)
    else:
        raise NotImplementedError
    print(cls_model)
    cls_model.to(device)
    if device == torch.device("cuda") and device_count > 1:
        cls_model = DataParallel(cls_model)  # Wrap the model with DataParallel

    #initialize optimizer
    optimizer = get_opt(cls_model, args)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.98)

    #initialize loss functions
    if args.loss == 'ce':
        cls_loss_fun = nn.CrossEntropyLoss()
    elif args.loss == 'wce':
        class_weights = class_weights/torch.sum(class_weights)
        cls_loss_fun = nn.CrossEntropyLoss(weight = class_weights.to(device))
    elif args.loss == 'focal':
        class_weights = class_weights/torch.sum(class_weights)
        cls_loss_fun = FocalLoss(gamma = 0.75 , alpha = class_weights)

    ##initialize earlystopping
    early_stopping = EarlyStopping(warmup = 10, tolerance = 25, metric='val_auc')
    
    for epoch in range(args.epochs):
        print(f'\nTraining epoch {epoch}......')

        train_loss = 0
        labels = []
        y_probs = []

        cls_model.train()
        for batch_idx, (mrn, frame_features, target) in enumerate(train_loader):
            frame_features = frame_features.to(device)
            target = target.to(device)
            input_data = {'frame_features':frame_features}#, 'clinical':clinical}

            logits,y_prob = cls_model(input_data)
            cls_loss = cls_loss_fun(logits,target)
            cls_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            train_loss += cls_loss

            if (batch_idx + 1) % 20 == 0:
                print(f'batch {batch_idx}; mrn {mrn}; label {target.detach().cpu().numpy()}; loss {cls_loss: .4f}')

            #save for metrics computation
            labels.extend(target.detach().cpu().numpy())
            y_probs.extend(y_prob[:,1].detach().cpu().numpy())
            
        train_loss = train_loss / len(train_loader) #size of loader
        train_acc, train_auroc, train_auprc = compute_metrics(np.array(labels),np.array(y_probs),thresh = 0.5)
        print(f'Epoch {epoch} [train]: loss{train_loss: .4f}; accuracy{train_acc: .4f}; AUROC{train_auroc: .4f}; AUPRC{train_auprc: .4f}\n')
        
        if writer is not None:
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/acc', train_acc, epoch)
            writer.add_scalar('train/auroc', train_auroc, epoch)
            writer.add_scalar('train/auprc', train_auprc, epoch)

        if epoch >= 9:
            scheduler.step()

        early_stop = eval_graph(args, split, epoch, val_loader, cls_model, early_stopping, writer, class_weights = class_weights)
        if early_stop:
            break

    #load the weights of the best model
    best_model_weights = torch.load(os.path.join(args.result_dir, f"split_{split}_ckpt.pt"))
    cls_model.load_state_dict(best_model_weights)
    val_acc, val_auroc, val_auprc = eval_graph(args, split, 'final', val_loader, cls_model, save_dict=True, class_weights = class_weights)
    if writer:
        writer.close()
    return val_acc, val_auroc, val_auprc

def eval_graph(args: argparse.ArgumentParser, split: int, epoch: int, val_loader: torch.utils.data.DataLoader, cls_model: torch.nn.Module, 
    early_stopping: EarlyStopping = None, writer: SummaryWriter = None, save_dict: bool = False, evaluation: bool = False, 
    class_weights: torch.Tensor = None) -> Tuple[float, float, float]:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.loss == 'ce':
        cls_loss_fun = nn.CrossEntropyLoss()
    elif args.loss == 'wce':
        cls_loss_fun = nn.CrossEntropyLoss(weight = class_weights.to(device))
    elif args.loss == 'focal':
        cls_loss_fun = FocalLoss(gamma = 0.75 , alpha = class_weights)

    val_loss = 0.0
    mrns = []
    labels = []
    y_probs = []
    
    cls_model.eval()
    with torch.no_grad():
        for batch_idx, (mrn, frame_features, target) in enumerate(val_loader):
            frame_features = frame_features.to(device)
            target = target.to(device)
            input_data = {'frame_features':frame_features}
            logits,y_prob = cls_model(input_data)
            cls_loss = cls_loss_fun(logits,target)
            val_loss += cls_loss

            #save for metrics computation
            mrns.extend(mrn)
            labels.extend(target.detach().cpu().numpy())
            y_probs.extend(y_prob[:,1].detach().cpu().numpy())

    val_loss = val_loss / len(val_loader)
    val_acc, val_auroc, val_auprc = compute_metrics(np.array(labels),np.array(y_probs),thresh = 0.5)
    print(f'Epoch {epoch} [val]: loss{val_loss: .4f}; accuracy{val_acc: .4f}; AUROC{val_auroc: .4f}; AUPRC{val_auprc: .4f}')

    if save_dict:
        split_dict = pd.DataFrame({'mrn':mrns, 'label':labels, 'prob': y_probs})
        if evaluation:
            split_dict.to_csv(os.path.join(args.eval_dir,f'split_{split}_result.csv'), index = False)
        else:            
            split_dict.to_csv(os.path.join(args.result_dir,f'split_{split}_result.csv'), index = False)

    if writer is not None:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/acc', val_acc, epoch)
        writer.add_scalar('val/auroc', val_auroc, epoch)
        writer.add_scalar('val/auprc', val_auprc, epoch)

    if early_stopping is not None:
        if early_stopping.metric=='val_loss':
            early_stopping(epoch, val_loss.cpu(), cls_model, ckpt_path = os.path.join(args.result_dir, f"split_{split}_ckpt.pt"))
        elif early_stopping.metric=='val_auc':
            early_stopping(epoch, val_auroc, cls_model, ckpt_path = os.path.join(args.result_dir, f"split_{split}_ckpt.pt"))

        if early_stopping.early_stop:
            print('Early Stopped!')
            return True
        else:
            return False
    else:
        return val_acc, val_auroc, val_auprc

def train_wang(args: argparse.ArgumentParser, split, train_loader: torch.utils.data.DataLoader, 
               val_loader:torch.utils.data.DataLoader, class_weights: torch.Tensor) ->  Tuple[float, float, float]:
    if args.log:
        log_dir = os.path.join(args.result_dir, f'split_{split}_log')
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_count = torch.cuda.device_count()
    assert device_count>=1

    #initialize models
    if args.model_type == 'wang':
        if args.input_type == 'wang_frame':
            cls_model = WangModel()
    else:
        raise NotImplementedError
    print(cls_model)
    cls_model.to(device)
    if device == 'cuda' and device_count > 1:
        cls_model = DataParallel(cls_model)  # Wrap the model with DataParallel

    #initialize optimizer
    optimizer = get_opt(cls_model, args)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.98)

    #initialize loss functions
    if args.loss == 'wang_loss':
        cls_loss_fun = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(warmup = 0, tolerance = 100, metric='val_auc')
    
    for epoch in range(args.epochs):
        print(f'\nTraining epoch {epoch}......')

        train_loss = 0
        labels = []
        y_probs = []

        cls_model.train()
        for batch_idx, (mrn, bag_imgs, target) in enumerate(train_loader):
            bag_imgs = bag_imgs.to(torch.float32).to(device)
            target = torch.tensor(target)
            target = target.to(device)
            
            logits,y_prob,A = cls_model(bag_imgs, 'train')
            cls_loss = cls_loss_fun(logits,target)
            cls_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            train_loss += cls_loss

            if (batch_idx + 1) % 40 == 0:
                print(f'batch {batch_idx}; mrn {mrn}; label {target.detach().cpu().numpy()}; loss {cls_loss: .4f}')
            # if (batch_idx + 1) % 150 == 0:                
            #     print('Attention: ', A.flatten().unique()[:10])

            #save for metrics computation
            labels.extend(target.detach().cpu().numpy())
            y_probs.extend(y_prob[:,1].detach().cpu().numpy())

            # if batch_idx==5:
            #     break

        train_loss = train_loss / len(train_loader) #size of loader
        train_acc, train_auroc, train_auprc = compute_metrics(np.array(labels),np.array(y_probs),thresh = 0.5)
        print(f'Epoch {epoch} [train]: loss{train_loss: .4f}; accuracy{train_acc: .4f}; AUROC{train_auroc: .4f}; AUPRC{train_auprc: .4f}\n')
        
        if writer is not None:
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/acc', train_acc, epoch)
            writer.add_scalar('train/auroc', train_auroc, epoch)
            writer.add_scalar('train/auprc', train_auprc, epoch)
        
        if epoch >= 9:
            scheduler.step()
    
        early_stop = eval_wang(args, split, epoch, val_loader, cls_model, early_stopping, writer, class_weights = class_weights)
        if early_stop:
            break
    
    #load the weights of the best model
    best_model_weights = torch.load(os.path.join(args.result_dir, f"split_{split}_ckpt.pt"))
    cls_model.load_state_dict(best_model_weights)
    val_acc, val_auroc, val_auprc = eval_wang(args, split, 'final', val_loader, cls_model, save_dict=True, class_weights = class_weights)
    if writer:
        writer.close()
    return val_acc, val_auroc, val_auprc

def eval_wang(args: argparse.ArgumentParser, split: int, epoch: int, val_loader: torch.utils.data.DataLoader, cls_model: torch.nn.Module, 
    early_stopping: EarlyStopping = None, writer: SummaryWriter = None, save_dict: bool = False, evaluation: bool = False, 
    class_weights: torch.Tensor = None) -> Tuple[float, float, float]:
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cls_model.eval()
    if args.loss == 'wang_loss':
        cls_loss_fun = nn.CrossEntropyLoss()

    val_loss = 0.0
    mrns = []
    labels = []
    y_probs = []

    with torch.no_grad():
        for batch_idx, (mrn, bag_imgs, target) in enumerate(val_loader):
            bag_imgs = bag_imgs.to(torch.float32).to(device)
            target = torch.tensor(target)
            target = target.to(device)
            logits,y_prob,A = cls_model(bag_imgs, 'test')
        
            # if (batch_idx + 1) % 50 == 0:                
            #     print('Attention: ', A.flatten().unique()[:10])
            
            cls_loss = cls_loss_fun(logits,target.repeat(len(logits)))
            val_loss += cls_loss
            mrns.extend(mrn)
            
            labels.extend(target.detach().cpu().numpy())#.item())
            y_probs.append(y_prob[:,1].detach().cpu().numpy().max())

            # if batch_idx==10:
            #     break

    val_loss = val_loss / len(val_loader) 
    val_acc, val_auroc, val_auprc = compute_metrics(np.array(labels),np.array(y_probs),thresh = 0.5)
    print(f'Epoch {epoch} [val]: loss{val_loss: .4f}; accuracy{val_acc: .4f}; AUROC{val_auroc: .4f}; AUPRC{val_auprc: .4f}')

    if save_dict:
        split_dict = pd.DataFrame({'mrn':mrns, 'label':labels, 'prob': y_probs})
        if evaluation:
            split_dict.to_csv(os.path.join(args.eval_dir,f'split_{split}_result.csv'), index = False)
        else:            
            split_dict.to_csv(os.path.join(args.result_dir,f'split_{split}_result.csv'), index = False)

    if writer is not None:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/acc', val_acc, epoch)
        writer.add_scalar('val/auroc', val_auroc, epoch)
        writer.add_scalar('val/auprc', val_auprc, epoch)

    if early_stopping is not None:
        if early_stopping.metric=='val_loss':
            early_stopping(epoch, val_loss.cpu(), cls_model, ckpt_path = os.path.join(args.result_dir, f"split_{split}_ckpt.pt"))
        if early_stopping.metric=='val_auc':
            early_stopping(epoch, val_auroc, cls_model, ckpt_path = os.path.join(args.result_dir, f"split_{split}_ckpt.pt"))
        if early_stopping.metric=='val_acc':
            early_stopping(epoch, val_acc, cls_model, ckpt_path = os.path.join(args.result_dir, f"split_{split}_ckpt.pt"))

        if epoch==99:
            print(f'Reached epoch {epoch}, ending training!')
            return True
        else:
            return False
    else:
        return val_acc, val_auroc, val_auprc

    return val_acc, val_auroc, val_auprc