import numpy as np
import pandas as pd 
from tqdm import tqdm
import pandas as pd
tqdm.pandas(desc='pandas bar')
import random
import tqdm
import copy
import logging
import pandas as pd
import argparse
import os
import json
import os.path as osp
import numpy as np
from sklearn.cluster import KMeans

import json
import time
# from FlagEmbedding import FlagModel
import shutil
from utils.load_dataset import DatasetLoader,GSLDataset
# from utils.generate_file import DataProcessor
# from utils.graph_train import GraphTrainer
# from utils.semantic_embedding import SemanticEmbedder
# from utils.func import cluster_by_attribute,sort_clusters_by_members_length,split_into_clusters
# from utils.clustering import ClusterAnalyzer,KMeansClusterer
from utils.LLM_dialog import DetectorRule
from utils.func import extract_first_function,execute_first_function,calculate_f1_with_smoothing,find_unique_false_rows,extract_and_make_callable

from sklearn.metrics import precision_score,recall_score,f1_score
from types import SimpleNamespace
from openai import OpenAI
import yaml
parser = argparse.ArgumentParser()

parser.add_argument('--dataset_name', type=str, default='Hospital', help='dataset name')
parser.add_argument('--base_path', type=str, default='GEIL_Data',help='Dataset Base Path')
parser.add_argument('--config_path', type=str, default='',help='config for online model')

args = parser.parse_args()

dataset_name = args.dataset_name
base_path = args.base_path 
config_path = args.config_path
output_directory = 'output/{}/detector'.format(dataset_name)

## define LLM model path and url

if config_path!='':
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    func_model_path = config['func_model_path']
    base_url = config['base_url']
    api_key = config['api_key']
else:
    func_model_path = '/home/user/model/Qwen2.5-Coder-7B'
    base_url = "http://192.168.12.43:8000/v1"
    api_key = 'token-example'


detector_model_path = '../roberta-base/'

load_previous_func = False

# flag for sequence control
generate_detector_from_scratch = True
generate_generator_from_scratch = True
train_detector = False
save_detector_training_result = True
save_pseudo_label_result = True

### define threshold
detector_thres = 0.71
generate_thres = 0.71
correction_thres = 0.75
# augment_maximum =  60
augment_maximum = 500 ## for Rayyan
epoch = 10
### define client
client = OpenAI(
    base_url=base_url,
    api_key=api_key,
)

detector_rule_manager = DetectorRule(
    client=client,
    func_model_path=func_model_path,
    detector_thres=detector_thres
)

## Load Previous Generation Result 

outlier_index_list = np.load('output/{}/GSL/index.npy'.format(dataset_name))
clusters = np.load('output/{}/GSL/cluster.npy'.format(dataset_name),allow_pickle=True).item()


loader = DatasetLoader(base_path=base_path)
dirty_table, clean_table = loader.load_dataset(dataset_name)
outlier = np.array(dirty_table.iloc[outlier_index_list]!=clean_table.iloc[outlier_index_list])


if generate_detector_from_scratch:
    function_list = detector_rule_manager.generate_and_validate_rules(
        outlier=outlier,
        outlier_index_list = outlier_index_list,
        dirty_table=dirty_table,
        clean_table=clean_table,
        output_path=output_directory,
        dataset_name=dataset_name,
        overwrite_func_list=True, # 首次运行通常为 True，后续迭代可以为 False
        max_detector_retries=30, # 减少尝试次数以加快测试
        multi_turn_dialogue = True
    )
else:
    function_list = np.load('output/{}/detector/function_list.npy'.format(dataset_name),allow_pickle=True).item() ## load_previous result
    
if generate_generator_from_scratch:
    function_list_result = detector_rule_manager.generate_and_validate_corruption_rules(
        outlier=outlier,
        outlier_index_list = outlier_index_list,
        dirty_table=dirty_table,
        clean_table=clean_table,
        function_list = function_list,
        dataset_name=dataset_name,
        max_generator_retries=30, 
        generate_thres = generate_thres,
        multi_turn_dialogue = True
    )
    np.save('output/{}/detector/function_list_generator.npy'.format(dataset_name),function_list_result)
else:
    function_list = np.load('output/{}/detector/function_list_generator.npy'.format(dataset_name),allow_pickle=True).item()


## self-generate training data

select_col = np.where(np.array(outlier).sum(axis=0)!=0)[0]

detector_train = []
detector_train_list = []
detector_test = []
for index in outlier_index_list:
    context_cell = ''
    for i in range(len(dirty_table.columns)):
        context_cell += 'COL %s VAL %s ' % (dirty_table.columns[i],dirty_table.iloc[index,i])
    for col in select_col:     
        dirty_value = dirty_table.iloc[index,col]
        clean_value = clean_table.iloc[index,col]
        detect_cell = 'COL %s VAL %s ' % (dirty_table.columns[col],dirty_value)
        clean_cell =  'COL %s VAL %s ' % (dirty_table.columns[col],clean_value)
        if(str(dirty_value)!=str(clean_value)):
            label = 1 ## Outlier
            detector_train.append([context_cell,detect_cell,1])
            detector_train_list.append([index,col,dirty_value,clean_value])
            detector_train.append([context_cell,clean_cell,0])
        else:
            label = 0 ## Normal
            detector_train.append([context_cell,detect_cell,0])
detector_ground_truth = detector_train
# Augmented File
for col in select_col:
    print(col,clean_table.columns[col])
    # try:
    augment_index = find_unique_false_rows(dirty_table, col, function_list, augment_maximum)
    for index in augment_index:
        context_cell = ''
        generation_func = extract_and_make_callable(function_list[col]['generator_prompt_output'])
        for i in range(len(dirty_table.columns)):
            context_cell += 'COL %s VAL %s ' % (dirty_table.columns[i],dirty_table.iloc[index,i])
        clean_value = dirty_table.iloc[index,col]
        # dirty_value = Beer_Row_Generation(beer_dirty.columns[col],clean_value)
        dirty_value = generation_func(clean_value)
        detect_cell = 'COL %s VAL %s ' % (dirty_table.columns[col],dirty_value)
        clean_cell =  'COL %s VAL %s ' % (dirty_table.columns[col],clean_value)
        detector_train_list.append([index,col,dirty_value,clean_value])
        detector_train.append([context_cell,detect_cell,1])
        detector_train.append([context_cell,clean_cell,0])
    # except:
    #     continue


for index in range(len(dirty_table)):
    context_cell = ''
    for i in range(len(dirty_table.columns)):
        context_cell += 'COL %s VAL %s ' % (dirty_table.columns[i],dirty_table.iloc[index,i])
    for col in range(len(dirty_table.columns)):
        
        detect_cell = 'COL %s VAL %s ' % (dirty_table.columns[col],dirty_table.iloc[index,col])
        dirty_value = dirty_table.iloc[index,col]
        clean_value = clean_table.iloc[index,col]
        if(str(dirty_value)!=str(clean_value)):
            label = 1 ## Outlier
            detector_test.append([context_cell,detect_cell,1])
        else:
            label = 0 ## Normal
            detector_test.append([context_cell,detect_cell,0])
            
train_all = pd.DataFrame(detector_train).sample(frac=1)


test_all = pd.DataFrame(detector_test)



## whether to save training files
if save_detector_training_result:
    os.makedirs('PyG_Dataset/{}/detector'.format(dataset_name),exist_ok=True)
    train_all.to_csv('PyG_Dataset/{}/detector/train.csv'.format(dataset_name))
    test_all.to_csv('PyG_Dataset/{}/detector/test.csv'.format(dataset_name))
    valid_all =pd.DataFrame(detector_ground_truth)
    valid_all.to_csv('PyG_Dataset/{}/detector/valid.csv'.format(dataset_name))

if save_pseudo_label_result:
    np.save('output/{}/detector/detector_train_list.npy'.format(dataset_name),detector_train_list)


print(len(train_all))
## Start Training

if train_detector:
    from ditto.model import DittoModel,DittoDataset,load_model,to_str,classify,train,simple_train,simple_train_update
    train_dataset = DittoDataset(train_all,max_len=128,lm = detector_model_path)
    valid_dataset = DittoDataset(pd.DataFrame(detector_ground_truth),max_len=128,lm = detector_model_path)
    test_dataset_sample = DittoDataset(test_all.sample(n=1000),max_len=128,lm = detector_model_path)
    test_dataset = DittoDataset(pd.DataFrame(detector_test),max_len=128,lm = detector_model_path)

    hp_simple = SimpleNamespace(task='{}-detector-train'.format(dataset_name),
                        batch_size=64,
                        max_len=128,
                        lr=3e-5,
                        n_epochs=epoch,
                        save_model=True,
                        logdir="detector_model/", ## Checkpoint save path
                        lm=detector_model_path, ## roberta-base model, please change to your own model path
                        fp16=True,
                        alpha_aug=0.8)

    model_output = simple_train(train_dataset,valid_dataset,test_dataset_sample,hp_simple)

    start_time = time.time()
    predict = classify(test_dataset,model=model_output,lm=detector_model_path,max_len=128,threshold=0.5) ## Inference
    end_time = time.time()
    print(f"inference time：{end_time - start_time} s")
    print("prec: ", precision_score(y_pred=predict[0],y_true=test_all.iloc[:,-1].astype(int)), " recall: ",  recall_score(y_pred=predict[0],y_true=test_all.iloc[:,-1].astype(int)), ", f1: ", f1_score(y_pred=predict[0],y_true=test_all.iloc[:,-1].astype(int)))

    ## save detection result
    detection_result = np.array(predict[0]).reshape(len(dirty_table),len(dirty_table.columns))
    np.save('output/{}/detector/detection_result.npy'.format(dataset_name),detection_result)

## save pseudo_label result
