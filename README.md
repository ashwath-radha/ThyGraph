# ThyGraph: A Graph-Based Approach for Thyroid Nodule Diagnosis from Ultrasound Studies

Improved thyroid nodule risk stratification from ultrasound (US) can mitigate overdiagnosis and unnecessary biopsies. Previous studies often train deep learning models using manually selected single US frames; these approaches deviate from clinical practice where physicians utilize multiple image views for diagnosis. This paper introduces ThyGraph, a novel graph-based approach that improves feature aggregation and correlates anatomically proximate images, by leveraging spatial information to model US image studies as patient-level graphs.  In the proposed graph paradigm, ThyGraph can effectively aggregate information across views of a nodule and take advantage of inter-image dependencies to improve nodule risk stratification, leading to better patient triaging and reducing reliance on biopsies.

## Directory Structure
```
.
├── main.py                                  # Train and evaluate models
├── feature_extraction.py                    # Extract features with pre-trained ThyNet Ensemble
├── evaluation.py                            # Evaluate on test set
├── data_processing.py                       # Extract labels from radiology/cytology reports
├── graph_construction.py                    # Construct different kinds of graphs 
├── utils                                    # Folder containing functions 
│   ├── utils.py                             # Functions used in train.py
│   ├── train.py                             # Train and evaluate functions
│   ├── interpretability.py                  # Interpretability visualization functions
│   ├── loss_function.py                     # Loss function used in train.py
│   ├── dataset_raw.py                       # Dataloader for feature extraction
│   |── dataset_bags.py                      # Dataloader for full frame images
│   |── dataset_patches.py                   # Dataloader for image patches
│   |── dataset_graphs.py                    # Dataloader for graphs
│   |── dataset_wang.py                      # Dataloader for Wang model baseline
│   |── graph_utils.py                       # Functions used in graph_construction.py
|   └── custom_preprocessing.py              # Preprocessing US images
├── models                                   # Models                  
│   ├── mil.py                               # MIL and AMIL models
│   ├── gcn.py                               # GCN models
│   ├── resnet.py                            # ResNet50
│   ├── semi_supervised.py                   # Semi-supervised learning model
│   └── pre_trained_segmentation.py          # Pre-trained thyroid nodule segmentation model
├── iodata                                   # Folder containing raw data, features, labels, and splits
│   ├── splits                               # Folder for splits
│   │   ├── splits_1.csv
│   │   └── ...
│   ├── feature_extractor                    # Folder to store pre-trained weights
│   │   ├── RadImageNet-ResNet50_notop.h5               
│   │   └── ...
│   │── bags                                 # Folder containing labels and features for each patient
│   │   ├── label_dummy.csv                  # Output of data_preprocessing.py
│   │   └── files                            # Folder with extracted features in .h5
│   │       ├── <mrn>.h5        
│   │       └── ... 
│   ├── patches_<configuration>              # Contains pre-saved patches per MRN in .h5
│   │       ├── <mrn>.h5        
│   │       └── ... 
├── testing                                  # Testing functions in utils and models folder             
│   ├── test_utils.py                        # testing utils/utils.py, utils/interpretability.py
│   ├── test_dataloader.py                   # testing utils/dataset_raw.py, utils/dataset_bags.py
│   └── test_model.py                        # testing models
├── results                                  # Output from training and evaluation
│   ├── <experiment_name> 
│   │   ├──summary.csv                       # Accuracy, AUROC, AUCPR per fold
│   │   ├──args.txt                          # Training arguments
│   │   ├──split_0_ckpt.pt                   # Weights for the fold
│   │   ├──split_0_result.csv                # Prediction for each patient in the fold
│   │   ├──split_0_log                       # Tensorboard log file
│   │   └── ...
│   └── ...
└── ...
```

## Prerequisite

Create a virtual environment and install necessary dependencies by:

```
conda env create -f environment.yml

pip install -r requirements.txt

```

## Instructions for Running

The core files in this codebase are: main.py, feature_extraction.py, evaluation.py, data_processing.py and graph_construction.py. The instructions for running each component are included below. Core files utilize a variety of arguments that modulate input for each script; further guidance of configuring these arguments and running scripts are outlined below.

### 1. Data preprocessing - data_processing.py

The purpose of data_processing.py is to establish the final cohort for experiments. It is possible to create a train/validation/test split as well as a stratified 5-fold cross validation split.

Required arguments:
+ input_csv -- path to the csv file which contains patients’ information
+ output_csv - path to the csv file which stores the output label and path for each image file
+ split_dir - path to splits directory
+ n_splits - number of splits (if 0 it will create a train/validation/test split, if >0 it will create that many CV folds)
+ data_path -- path to the raw data

### 2. Feature extraction - feature_extraction.py

The purpose of feature_extraction.py is to extract deep features from ultrasound images using various model architectures.

Required arguments:
+ output - path to the output directory
+ dataroot - path to the raw data (class_data in box drive)
+ csv_file - path to the csv file which contains patients’ information
+ gpu - Which gpu to use

### 3. Graph construction - graph_construction.py

The purpose of graph_construction.py is to construct a patient-level graph for each patient's image study. Includes functionality to use different edge construction heuristics such as feature similarity and image anatomical location.

Required arguments:
+ input_features - path to the input directory with pre-extracted features
+ output - path to output directory for storing graphs
+ dataroot - path to input directory with ultrasound image studies
+ csv_file - path to the csv file which stores the label and path for each image file
+ edge_method - method for making edges for each graph
+ gpu - Which gpu to use

### 4. Training - main.py

The purpose of main.py is to train the MIL- or graph-based models either on the pre-extracted image features or patient graphs to predict nodule malignancy.

Required arguments:

+ output - path to results directory
+ exp - name of experiment
+ dataroot - path to input directory with ultrasound image studies
+ features_root - path to the input directory with pre-extracted features (if training a graph model, pick the equivalent directory with patient graphs)
+ csv_file - path to the csv file which stores the label and path for each image file
+ splitroot - path to cross validation splits
+ log - include parameter to turn on tensorboard logging
+ gpu - Which gpu to use

### 5. Evaluation - evaluation.py

The purpose of evaluation.py is to evaluate any trained models and generate final performance metrics. It is also used to generate attention heatmaps for the MIL-based models. Moving forward, it will also include implementation to generate the thyroid mesh model heatmaps to visualize attention for the graph-based models.

## Additional Files

Will include brief descriptions of auxiliary files in codebase.
