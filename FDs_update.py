import shutil
from utils.load_dataset import DatasetLoader,GSLDataset
from utils.generate_file import DataProcessor
from utils.graph_train import GraphTrainer
from utils.semantic_embedding import SemanticEmbedder
from utils.func import cluster_by_attribute,sort_clusters_by_members_length,split_into_clusters
from utils.clustering import ClusterAnalyzer,KMeansClusterer
from utils.LLM_dialog import DetectorRule
from utils.func import extract_first_function,execute_first_function,calculate_f1_with_smoothing,find_unique_false_rows,extract_and_make_callable, find_element_in_clusters
from utils.correlation import _weighted_entropy,find_categorical_dependencies,correct_dataframe
from sklearn.metrics import precision_score,recall_score,f1_score

import numpy as np
from tqdm import tqdm
tqdm.pandas(desc='pandas bar')
import argparse
import copy
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_name', type=str, default='Hospital', help='dataset name')
parser.add_argument('--base_path', type=str, default='GEIL_Data',help='Dataset Base Path')

args = parser.parse_args()
dataset_name = args.dataset_name
base_path = args.base_path

### Hyper Parameters

alpha = 5 ## lambda in paper, avoid critical parameter duplicate
correlation_thres = 0.99
one_to_one_variable_FDs = True
skip_index = True

### Load previous file

detector_result = np.load('output/{}/detector/detection_result.npy'.format(dataset_name),allow_pickle=True)
coreset_all = np.where(detector_result.sum(axis=1)==0)[0]
correction = pd.read_csv('output/{}/correction/correction.csv'.format(dataset_name)).astype(str)

### Load dirty/clean table

loader = DatasetLoader(base_path=base_path)
dirty_table, clean_table = loader.load_dataset(dataset_name)

### Find FDs
if skip_index: ## default index is is 1st location
    FDs = find_categorical_dependencies(correction.iloc[:,1:],coreset_indices=coreset_all, alpha=alpha, threshold=correlation_thres)

FD_list = list(FDs.keys())

## Voting for correction

correted_fds = correct_dataframe(correction_df = correction,dependencies=FD_list,training_df = correction,coreset_indices = coreset_all,alpha=alpha)

correted_fds.to_csv('output/{}/correction/correted_fds.csv'.format(dataset_name), index=False)

## evaluation

All_Data_Error = 0
All_Fixed_Error = 0
Correct_Fixed_Error = 0
correction = correted_fds
for i in range(len(dirty_table)):
# for i in tqdm(tax_error):
    for j in range(dirty_table.shape[1]):
        dirty_cell = dirty_table.iloc[i,j]
        clean_cell = clean_table.iloc[i,j]
        correct_cell = correction.iloc[i,j]
        if(str(correct_cell)!=str(dirty_cell)):
            All_Fixed_Error += 1
        if(str(clean_cell)!=str(dirty_cell)):
            All_Data_Error += 1
            if(str(correct_cell)==str(clean_cell)):
                Correct_Fixed_Error += 1
Precision_hospital = Correct_Fixed_Error / All_Fixed_Error
Recall_hospital = Correct_Fixed_Error / All_Data_Error
F1_hospital = (2 * Precision_hospital * Recall_hospital) / (Precision_hospital + Recall_hospital)
print(Precision_hospital,Recall_hospital,F1_hospital)