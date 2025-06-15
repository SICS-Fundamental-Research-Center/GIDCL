import numpy as np
from tqdm import tqdm
tqdm.pandas(desc='pandas bar')
import random

import copy
import logging
import pandas as pd
import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from typing import Callable, List, Optional
import subprocess
import json
import yaml
import time
from FlagEmbedding import FlagModel
import shutil
from utils.load_dataset import DatasetLoader,GSLDataset
from utils.generate_file import DataProcessor
from utils.graph_train import GraphTrainer
from utils.semantic_embedding import SemanticEmbedder
from utils.func import cluster_by_attribute,sort_clusters_by_members_length,split_into_clusters
from utils.clustering import ClusterAnalyzer,KMeansClusterer
from utils.LLM_dialog import DetectorRule
from utils.func import extract_first_function,execute_first_function,calculate_f1_with_smoothing,find_unique_false_rows,extract_and_make_callable, find_element_in_clusters
from sklearn.metrics import precision_score,recall_score,f1_score
from types import SimpleNamespace
from openai import OpenAI
parser = argparse.ArgumentParser()
from tqdm import tqdm


# 读取 YAML 文件


parser.add_argument('--dataset_name', type=str, default='Hospital', help='dataset name')
parser.add_argument('--base_path', type=str, default='GEIL_Data',help='Dataset Base Path')
parser.add_argument('--base_model', type=str, default='qwen',help='Dataset Base Path')

args = parser.parse_args()
dataset_name = args.dataset_name
base_path = args.base_path
base_model = args.base_model

## Hyper-Parameters
top_k = 2

## Load GSL Cluster Result

clusters = np.load('output/{}/GSL/cluster.npy'.format(dataset_name),allow_pickle=True).item()

## Load detector result from previous ste

detector_result = np.load('output/{}/detector/detection_result.npy'.format(dataset_name),allow_pickle=True)

## Load Pseudo-Label training list

detector_train_list = np.load('output/{}/detector/detector_train_list.npy'.format(dataset_name))

## Load previous annotation index

outlier_index_list = np.load('output/{}/GSL/index.npy'.format(dataset_name))

## Load template yaml file

if base_model.lower().__contains__('qwen'):
    yaml_template = 'script/qwen2.5-7B-hospital-template.yaml'
elif base_model.lower().__contains__('mistral'):
    yaml_template = 'script/mistral-7B-hospital-template.yaml'

loader = DatasetLoader(base_path=base_path)
dirty_table, clean_table = loader.load_dataset(dataset_name)

correction_train_list = []
correction_test_list = []

## Generate training file

col_list = dirty_table.columns

coreset_all = np.where(detector_result.sum(axis=1)==0)[0]
beer_clean_set = clean_table.iloc[outlier_index_list]
beer_dirty_set = dirty_table.iloc[outlier_index_list]
print(detector_train_list)
for index,col,dirty_value,clean_value in detector_train_list:
    index = int(index)
    col = int(col)
    dirty_list = []
    cluster_result = find_element_in_clusters(clusters,index)
    col_name = col_list[col]
    template_dict = {}
    template_dict[col_name] = ''
    demonstration = ''
    for id,row in beer_dirty_set[beer_dirty_set.iloc[:,col]!=beer_clean_set.iloc[:,col]].iterrows():
        # print(hospital_clean.iloc[index,i],hospital_dirty.iloc[index,i])
        ## Avoid directly listing label values
        dirty_list.append([clean_table.iloc[id,col],dirty_table.iloc[id,col]])
    if dirty_list!=[]: ## Add self-generated case?
        # dirty_list = []
        demonstration = "The input \n\n%s\n\nare [clean,dirty] cell pairs from table %s column %s." % (dirty_list,dataset_name,col_name)

    ground_truth_dict = {}
    ground_truth_dict[col_name] = clean_value
    RAG = ''
    temp_dict = dirty_table.iloc[index,1:].to_dict()
    temp_dict[col_name] = dirty_value ## Replace Dirty Value with Random Generation
    text_head = 'You are an expert in Cleaning %s Dataset. Given the dirty row Entity 1, you are required to correct the values of %s in Entity 1.\n\nReturn in json format.\n\nOutput Format Example:\n\n%s\n\nEntity 1:\n\n%s\n\n%s\n\nTake these clean rows as reference:\n\n' % (dataset_name, col_name, json.dumps(template_dict), json.dumps(temp_dict),demonstration)
    coreset = list(set(coreset_all).intersection(set(cluster_result)))
    ground_truth_list = [i for i in list(set(cluster_result).intersection(set(outlier_index_list))) if i!=index]
    for ground_truth_index in ground_truth_list: ## Ground Truth
        RAG += json.dumps(clean_table.iloc[ground_truth_index,1:].to_dict())
        RAG += '\n\n'
    try:
        for RAG_index in np.random.choice(coreset,min(top_k,len(coreset))):
            RAG += json.dumps(dirty_table.iloc[RAG_index,1:].to_dict())
            RAG += '\n\n'
            # print(len(coreset))
    except:
        print(coreset)
    labelling = json.dumps(ground_truth_dict)
    correction_train_list.append([text_head + RAG,'',labelling])

### Generating Test File. Labelling is not necessary for test file.
### Testing file is tied with detector result.

correction_test_list = []
beer_clean_set = clean_table.iloc[outlier_index_list]
beer_dirty_set = dirty_table.iloc[outlier_index_list]
for index,col in np.argwhere(detector_result!=0):
    dirty_list = []
    cluster_result = find_element_in_clusters(clusters,index)
    col_name = col_list[col]
    template_dict = {}
    template_dict[col_name] = ''
    demonstration = ''
    for id,row in beer_dirty_set[beer_dirty_set.iloc[:,col]!=beer_clean_set.iloc[:,col]].iterrows():
        # print(hospital_clean.iloc[index,i],hospital_dirty.iloc[index,i])
        ## Avoid directly listing label values
        dirty_list.append([clean_table.iloc[id,col],dirty_table.iloc[id,col]])
    if dirty_list!=[]: ## Add self-generated case?
        # dirty_list = []
        demonstration = "The input \n\n%s\n\nare [clean,dirty] cell pairs from table %s column %s." % (dirty_list,dataset_name,col_name)
    ground_truth_dict = {}
    dirty_value = dirty_table.iloc[index,col]
    clean_value = clean_table.iloc[index,col]
    ground_truth_dict[col_name] = clean_value
    RAG = ''
    temp_dict = dirty_table.iloc[index,1:].to_dict()
    temp_dict[col_name] = dirty_value ## Replace Dirty Value with Random Generation
    ground_truth_list = [i for i in list(set(cluster_result).intersection(set(outlier_index_list))) if i!=index]
    text_head = 'You are an expert in Cleaning %s Dataset. Given the dirty row Entity 1, you are required to correct the values of %s in Entity 1.\n\nReturn in json format.\n\nOutput Format Example:\n\n%s\n\nEntity 1:\n\n%s\n\n%s\n\nTake these clean rows as reference:\n\n' % (dataset_name, col_name, json.dumps(template_dict), json.dumps(temp_dict),demonstration)
    coreset = list(set(coreset_all).intersection(set(cluster_result)))
    if ground_truth_list==[]:
        ground_truth_list = np.random.choice(outlier_index_list,top_k)
    for ground_truth_index in ground_truth_list: ## Ground Truth
        RAG += json.dumps(clean_table.iloc[ground_truth_index,1:].to_dict())
        RAG += '\n\n'
    try:
        for index in np.random.choice(coreset,min(top_k,len(coreset))):
            RAG += json.dumps(dirty_table.iloc[index,1:].to_dict())
            RAG += '\n\n'
    except:
        print(coreset)
    labelling = json.dumps(ground_truth_dict)
    correction_test_list.append([text_head + RAG,'',labelling])

### Process for Training

correction_train_df = pd.DataFrame(correction_train_list)
correction_test_df = pd.DataFrame(correction_test_list)
correction_train_df.columns = ['instruction','input','output']
correction_test_df.columns = ['instruction','input','output']

### Dumping file
os.makedirs('output/{}/correction'.format(dataset_name),exist_ok=True)

json.dump(correction_train_df.to_dict(orient='records'), open('output/{}/correction/train.json'.format(dataset_name), 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
json.dump(correction_test_df.to_dict(orient='records'), open('output/{}/correction/test.json'.format(dataset_name), 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

## Generate Training Template

with open(yaml_template, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

lora_path = 'lora/{}/{}'.format(base_model, dataset_name)
config['output_dir'] = lora_path
train_file_path = 'output/{}/correction/train.json'.format(dataset_name)
test_file_path = 'output/{}/correction/test.json'.format(dataset_name)
config['train_file_path'] = train_file_path

output_yaml_path = 'script/{}-{}.yaml'.format(base_model, dataset_name)

with open(output_yaml_path, 'w', encoding='utf-8') as file:
    yaml.safe_dump(config, file, default_flow_style=False, allow_unicode=True)

sft_command = 'CUDA_VISIBLE_DEVICES=4,5,6,7 llamafactory-cli train {}'.format(output_yaml_path)
inference_command = 'CUDA_VISIBLE_DEVICES=6,7 python vllm_query_qwen.py --lora_path {} --input_file {} --output_path inference/{}_test.csv'.format(lora_path,test_file_path,dataset_name)

for command in [sft_command, inference_command]:
    print(command)
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    print(result.stdout)

## Evaluate Result
count = 0
correction = dirty_table.copy()

result = pd.read_csv('inference/{}_test.csv'.format(dataset_name))
for i,j in np.argwhere(detector_result==1):
    try:
        predict = list(eval(result.iloc[count,-1]).values())[0]
    except:
        print(result.iloc[count,-1])
        predict = '' 
    correction.iloc[i,j] = predict
    count += 1



correction.to_csv('output/{}/correction/correction.csv'.format(dataset_name), index=False)

# assert count == len(np.argwhere(detector_result==1))
# print(count)

# All_Data_Error = 0
# All_Fixed_Error = 0
# Correct_Fixed_Error = 0
# # clean = clean_table.copy()
# # dirty = dirty_table.copy()
# # correction = correction.copy()
# for i in tqdm(range(len(dirty_table))):
# # for i in tqdm(tax_error):
#     for j in range(dirty_table.shape[1]):
#         dirty_cell = dirty_table.iloc[i,j]
#         clean_cell = clean_table.iloc[i,j]
#         correct_cell = correction.iloc[i,j]
#         if(correct_cell!=dirty_cell):
#             All_Fixed_Error += 1
#         if(clean_cell!=dirty_cell):
#             All_Data_Error += 1
#             if(correct_cell==clean_cell):
#                 Correct_Fixed_Error += 1
# Precision_hospital = Correct_Fixed_Error / All_Fixed_Error
# Recall_hospital = Correct_Fixed_Error / All_Data_Error
# F1_hospital = (2 * Precision_hospital * Recall_hospital) / (Precision_hospital + Recall_hospital)
# print(Precision_hospital,Recall_hospital,F1_hospital)