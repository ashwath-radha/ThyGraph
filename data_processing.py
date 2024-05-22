import argparse
from typing import List
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import pandas as pd 
import os
from pandas import Series
import glob
import pickle
import datetime
from tqdm import tqdm
import sys
sys.path.append('/raid/aradhachandran/thyroid/code/classification/BE223A_thyroidProj/utils')
from extract_labels import *

def argument_parser() -> argparse.ArgumentParser:
    print('Running argument parser')

    parser = argparse.ArgumentParser(description='Feature extraction script')
    
    parser.add_argument('--input_csv', type=str, default='/raid/aradhachandran/thyroid/code/classification/labels/allmrns_rad_cyt_022224.csv', help='')
    parser.add_argument('--output_csv', type=str, default='/raid/aradhachandran/thyroid/code/classification/BE223A_thyroidProj/iodata/bags/label_miccai_cv3.csv', help='')
    parser.add_argument('--split_dir', type=str, default='/raid/aradhachandran/thyroid/code/classification/BE223A_thyroidProj/iodata/splits/miccai_cv3')
    parser.add_argument('--n_splits', type=int, default=5) # set to 0 for train-val-test split
    parser.add_argument('--data_path',type=str,default='/radraid/aradhachandran/thyroid/data/deepstorage_data/gcn_us')
    args = parser.parse_args()
    return args

# extract cytology for mrns present in directory of data
# returns a df with columns MRN, cytology label, sex, age, tirads, accession number
def extract_labels(input_csv: str):
    labels = getLabels(input_csv=input_csv, reports=None)
    return labels

def extract_imagePaths(datapath: str, labels_df: pd.DataFrame, topk: int, mask_ext='.mask.npy'):
    # go through mrns and get images and masks
    mrns = os.listdir(datapath)
    mrn_imagepaths = []
    skipped = 0
    stop = 0
    for mrn in tqdm(mrns, total=len(mrns)):
        if not mrn.startswith('.') and mrn in labels_df['MRN'].values:
            mrnpath = os.path.join(datapath,mrn)

            # for subdir in subdirs:
            mrn_usaccs = labels_df[labels_df['MRN']==mrn]['us acc'].values
            assert len(mrn_usaccs)==1
            subdir = mrn_usaccs[0]
            if topk:
                with open(f'/raid/aradhachandran/thyroid/code/classification/labels/mrn_files_ad_output_top{topk}.pkl', 'rb') as f:
                    mrn_files_ad_output_topk = pickle.load(f)
                im_files = mrn_files_ad_output_topk[f'{mrn}/{subdir}']
                im_files = [file.replace('raid', 'radraid') for file in im_files]
            else:
                im_files = os.listdir(os.path.join(mrnpath,subdir))
            images, masks = [], []
            for im in im_files:
                if im.endswith('patches'):# or im.endswith('.pred.npy'):
                    continue
                if mask_ext=='None' and im.endswith('.pred.npy'):
                    continue
                if im.endswith(mask_ext): masks += [im]
                else: images += [im]
            
            # double check that same amount of masks as images
            if mask_ext!='None':
                if len(masks) != len(images): raise Exception('number of images and masks do not align, check mrn ', mrn, len(masks),len(images))
            else: assert len(masks)==0

            imagepath = os.path.join(mrn,subdir)
            # catch if no masks
            if len(masks) == 0:
                for im in images:
                    mrn_imagepaths += [[mrn,os.path.join(imagepath,im),'']]
            else:
                # get image name 
                names = [mask.strip(mask_ext) for mask in masks]

                # create df with mrn and image, mask paths
                for im in names:
                    imageName = im+'.npy'
                    maskName = im+mask_ext

                    # # only add if something within border_cut pixels of edges
                    border_cut = 32
                    if np.sum(np.load(datapath+'/'+os.path.join(imagepath,maskName))[border_cut:-border_cut, border_cut:-border_cut])!=0:
                        mrn_imagepaths += [[mrn,os.path.join(imagepath,imageName),os.path.join(imagepath,maskName)]]
    
    # convert to dataframe for later merging
    mrn_imagepaths = pd.DataFrame(mrn_imagepaths,columns=['MRN','img_name','mask_name'])
    mrn_imagepaths['MRN'] = mrn_imagepaths['MRN'].apply(convert_to_string)
    print(skipped)
    return mrn_imagepaths

# Map the bethesda results to cytology label of benign/malignant
# remove any samples with bethesda < 2
# group indeterminate samples into the malignant class
# Reasoning is that these generally go for surgery since cannot rule out benign
# set drop_indet to true to keep only samples with that were labels as benign (bethesda 2) or malignant (bethesda 6)
def map_cytology(data: pd.DataFrame, drop_indet = False):
    # drop samples with cytology (bethesda score) < 2
    data = data[data['cytology'] > 1]

    if drop_indet:
        data = data.loc[(data['cytology'] == 2) | (data['cytology'].isin([5,6]))] # drops indet
        cyto_map = {2: 'benign', 5: 'malignant', 6: 'malignant'} # maps bethesda 2,5,6 to benign, malignant, resp.
    else:
        cyto_map = {2: 'benign', 3: 'malignant', 4: 'malignant', 5: 'malignant', 6: 'malignant'}

    data = data.replace({'cytology': cyto_map})
    return data

# create cross-validation splits and save to files
def create_splits(data: pd.DataFrame, input_csv: str, split_dir: str, n_splits: int, indet_dropped = False) -> None:
    data = data.drop_duplicates(ignore_index=True)

    print("DataFrame shape before splitting...", data.shape)
    X = data['mrn'].tolist()
    y = data['cytology'].tolist()
    skf = StratifiedKFold(n_splits=n_splits, random_state=1, shuffle=True)
    skf.get_n_splits(X, y)
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        #print(data.shape)
        X_train = [str(X[i]) for i in train_index]
        X_test = [str(X[i]) for i in test_index]

        split = pd.DataFrame()
        split['train'] = X_train
        split = pd.concat([split,pd.Series(X_test)], ignore_index=True, axis=1)
        split.columns = ['train','test']

        # put into respective folder based on if indeterminate samples were included
        # if indet_dropped: folder = '/noIndet/'
        # else: folder = '/withIndet/'
        # split.to_csv(split_dir + folder + 'cv'+str(fold)+'.csv', index=False)
        split.to_csv(split_dir + '/cv'+str(fold)+'.csv', index=False)

# create train/test split and save to file
def create_traintest(data: pd.DataFrame, bethesda_labels: pd.DataFrame, input_csv: str, split_dir: str, indet_dropped = False) -> None:
    ben_bethesda, mal_bethesda = [2], [5,6]
    bethesda_labels = bethesda_labels[bethesda_labels['label']!=-1]
    bethesda_labels = bethesda_labels[bethesda_labels['label'].isin(ben_bethesda+mal_bethesda)]
    bethesda_labels['final_label'] = 0
    bethesda_labels.loc[bethesda_labels['label'].isin(mal_bethesda), 'final_label'] = 1

    allmrns_rad_cyt = pd.read_csv(input_csv, index_col=0,converters={'MRN':convert_to_string, 'us acc':str, 'us date':pd.to_datetime, 'hist date':pd.to_datetime})
    bethesda_labels = bethesda_labels.merge(allmrns_rad_cyt[['MRN','us date', 'hist date']], how='left', on='MRN')
    # today = bethesda_labels['hist date'].max().date() #datetime.date.today()
    # bethesda_labels['hist_date_difference'] = (today - bethesda_labels['hist date'].dt.date).dt.days
    bethesda_labels = bethesda_labels[bethesda_labels['MRN'].isin(data['mrn'].values)]
    # pats_ofInterest = bethesda_labels[bethesda_labels['match_prevalence']==True]['MRN'].values
    # labels = bethesda_labels[bethesda_labels['match_prevalence']==True]['final_label'].values

    def split_dataset(patient_ids, labels, test_size=0.2, validation_size=0.2, random_state=None):
        X_train, X_test, y_train, y_test = train_test_split(patient_ids, labels, test_size=test_size, stratify=labels, random_state=random_state)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size, stratify=y_train, random_state=random_state)
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    # # split whole dataset
    # X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(bethesda_labels['MRN'].values, 
    #                                                                bethesda_labels['final_label'].values, 
    #                                                                test_size=0.2, validation_size=0.1, random_state=1)

    split = pd.DataFrame()
    split['train'] = X_train
    split = pd.concat([split,pd.Series(X_val)], ignore_index=True, axis=1)
    split = pd.concat([split,pd.Series(X_test)], ignore_index=True, axis=1)
    split.columns = ['train','test','holdout']

    print(f"Train Cohort: {len(X_train)} ({sum(y_train)}), Test Cohort: {len(X_val)} ({sum(y_val)}), Holdout Cohort: {len(X_test)} ({sum(y_test)})")
    print(f"for {split_dir.split('/')[-1]}...")
    split.to_csv(split_dir + '/cv0.csv', index=False)

    return

# check that a given MRN-accession study has >0 images after processing
def check_images(row, data_path):
    mrn, acc = row['MRN'], row['us acc']
    feats = os.listdir(os.path.join(data_path, mrn, acc))
    feats = [file for file in feats if not file.endswith('.pred.npy') and not file.endswith('patches')]
    return len(feats)!=0

# ensure a patient has <= n ultrasounds in study
def filter_study_size(row, min_n, max_n, data_path):
    mrn, acc = row['MRN'], row['us acc']
    feats = os.listdir(os.path.join(data_path, mrn, acc))#f'/radraid/aradhachandran/thyroid/data/deepstorage_data/gcn_us/{mrn}/{acc}/')
    feats = [file for file in feats if not file.endswith('.pred.npy') and not file.endswith('patches')]
    return len(feats)>min_n and len(feats)<=max_n # bump to 100?

def get_images_with_ad_score(row):
    mrn, acc = row['MRN'], row['us acc']
    try:
        file = mrn_files_ad_output_topk[mrn+'/'+acc]
    except KeyError:
        file = []
    return len(file)>0

def check_for_pred(row, mask_ext):
    mrn, acc = row['MRN'], row['us acc']
    feats = os.listdir(f'/radraid/aradhachandran/thyroid/data/deepstorage_data/gcn_us/{mrn}/{acc}/')
    preds = [f for f in feats if f.endswith(mask_ext)] # load all nodule prediction .npy files
    # ensure >= 1 of the predictions is non-empty
    for p in preds:
        pred = np.load(f'/radraid/aradhachandran/thyroid/data/deepstorage_data/gcn_us/{mrn}/{acc}/{p}')
        if np.sum(pred)!=0:
            return True
    return False

def filter_by_dicom_metadata(df, filter_str):
    with open(f'/raid/aradhachandran/thyroid/code/classification/labels/file_to_machine_022524.pkl', 'rb') as f:
        file_to_machine = pickle.load(f)
    with open(f'/raid/aradhachandran/thyroid/code/classification/labels/file_to_imageclass_022524.pkl', 'rb') as f:
        file_to_imageclass = pickle.load(f)

    df['file_name'] = df['img_name'].str.replace('.npy','').str[17:]
    df['machine'] = df['file_name'].map(file_to_machine)
    df['imageclass'] = df['file_name'].map(file_to_imageclass)
    assert df['machine'].notna().all(), "There are NaN entries in the 'machine' column."
    df = df[df['machine'].str.lower().str.contains(filter_str)]

    return df

def filter_image_frequency(df, column, min_freq, max_freq):
    column_counts = df[column].value_counts()
    filtered_df = df[df[column].isin(column_counts[column_counts >= min_freq].index)]
    filtered_df = filtered_df[filtered_df[column].isin(column_counts[column_counts <= max_freq].index)]
    return filtered_df

def main(args: argparse.ArgumentParser) -> None:
    print('Extracting labels...')
    labels = extract_labels(args.input_csv)
    print("Extracted Labels: ", labels.shape)
    
    labels = labels[labels.apply(check_images, args=(args.data_path,), axis=1)] # ensure a patient has >0 ultrasounds in study
    labels = labels[labels.apply(filter_study_size, args=(0,80,args.data_path,), axis=1)] # ensure a patient has <= n ultrasounds in study
    print("Filtering Labels by Study Size: ", labels.shape)

    topk = None #10
    if topk:
        with open(f'/data/aradhachandran/thyroid/classification/labels/mrn_files_ad_output_top{topk}.pkl', 'rb') as f:
            mrn_files_ad_output_topk = pickle.load(f)
        labels = labels[labels.apply(get_images_with_ad_score, axis=1)]
        print("Picking top 10 AD outputs: ", labels.shape)
    
    # check that it has nodule predictions
    mask_ext = 'None' #.pred.npy '.mask.npy'
    if mask_ext=='.pred.npy':
        labels = labels[labels.apply(check_for_pred, args=(mask_ext,), axis=1)] # incldue MRN-acc if it has at least one pred
        print("Filtering out studies without nodule predictions: ", labels.shape)

    labels = labels.loc[labels['label'].isin([2,5,6])]
    print("Filter for only B-II, B-V, B-VI cases: ", labels.shape)

    print('Extracting image paths...')
    imagePaths = extract_imagePaths(args.data_path, labels, topk=topk, mask_ext=mask_ext) # TODO could update this function to only select single frame candidate
    print("Image paths: ", imagePaths.shape)

    # merge labels and image/mask paths, modify col names and drop accession number
    data = pd.merge(labels,imagePaths,on=['MRN'],how='right')
    data['sex'], data['age'] = np.nan, np.nan
    data = data[['MRN','sex','age','label','TIRADS','img_name','mask_name']]
    data = data.rename(columns={'MRN': 'mrn', 'label': 'cytology', 'TIRADS': 'TI-RAD'})
    print(f"Full Dataset: {data.shape}, MRN count: {data['mrn'].nunique()}")

    print('Mapping cytology...')
    # create and save dataframes based on grouping of cytology
    data_noIndet = map_cytology(data, drop_indet = True)

    # changing cytology to actually be the malignancy date halfway mark
    only_malignants = False
    if only_malignants:
        # data_noIndet['cytology'] = 'benign'
        # data_noIndet.loc[data_noIndet['date_difference']<=1107, 'cytology'] = 'malignant'
        bethesda_labels = pd.read_csv('/raid/aradhachandran/thyroid/code/classification/labels/bethesda_labels.csv', index_col=0)
        bethesda_labels['MRN'] = bethesda_labels['MRN'].apply(convert_to_string)
        match_prev_pats = bethesda_labels[bethesda_labels['match_prevalence']==True]['MRN'].values
        data_noIndet['cytology'] = 'benign'
        data_noIndet.loc[data_noIndet['mrn'].isin(match_prev_pats), 'cytology'] = 'malignant'

    data_noIndet = filter_by_dicom_metadata(data_noIndet, filter_str='philips')
    print(f"Filtering for Philips Machines: {data_noIndet.shape}, MRN count: {data_noIndet['mrn'].nunique()}")

    data_noIndet = filter_image_frequency(data_noIndet, column='mrn', min_freq=5, max_freq=50)
    print(f"Filtering for Image Frequency: {data_noIndet.shape}, MRN count: {data_noIndet['mrn'].nunique()}")
    
    # Pick MRNs with specific us_date
    allmrns_rad_cyt = pd.read_csv(args.input_csv, index_col=0,converters={'MRN':convert_to_string, 'us acc':str, 'us date':pd.to_datetime, 'hist date':pd.to_datetime})
    allmrns_rad_cyt = allmrns_rad_cyt.rename(columns={'MRN':'mrn'})
    data_noIndet = data_noIndet.merge(allmrns_rad_cyt[['mrn','us date']], how='left', on='mrn')
    data_noIndet = data_noIndet[data_noIndet['us date'].dt.year.isin(list(range(2020,2022)))]
    print(f"Full Processed Dataset: {data_noIndet.shape}, MRN count: {data_noIndet['mrn'].nunique()}")

    out = args.output_csv.split('.csv')[0]
    data_noIndet.to_csv(out+'_noIndet.csv')

    os.makedirs(args.split_dir, exist_ok=True)
    if args.n_splits:
        # subset mrn and cytology from data df for creating splits
        data_noIndet_splits = data_noIndet[['mrn','cytology']]
        create_splits(data_noIndet_splits, args.input_csv, args.split_dir, args.n_splits, indet_dropped=True)
    else:
        data_noIndet_traintest = data_noIndet[['mrn','cytology']]
        create_traintest(data_noIndet_traintest, labels, args.input_csv, args.split_dir, indet_dropped=True)

if __name__=='__main__':
    args = argument_parser()
    main(args)
