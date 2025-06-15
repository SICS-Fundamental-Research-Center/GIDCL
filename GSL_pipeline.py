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
import os.path as osp
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F
from torch.nn import Linear
from typing import Callable, List, Optional
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.datasets import MovieLens
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
)
import json
from FlagEmbedding import FlagModel
import shutil
from utils.load_dataset import DatasetLoader,GSLDataset
from utils.generate_file import DataProcessor
from utils.graph_train import GraphTrainer
from utils.semantic_embedding import SemanticEmbedder
from utils.func import cluster_by_attribute,sort_clusters_by_members_length,split_into_clusters
from utils.clustering import ClusterAnalyzer,KMeansClusterer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
### Define the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='Hospital', help='dataset name')
parser.add_argument('--base_path', type=str, default='GEIL_Data',help='Dataset Base Path')
parser.add_argument('--embedding_model_path', type=str, default='../sentence_transformers/bge-small-en-1.5/', help='embedding model path')
args = parser.parse_args()

dataset_name = args.dataset_name
base_path = args.base_path
embedding_model_path = args.embedding_model_path


## Control Symbol
generate_GSL_file = False
remove_GSL_cache = True
add_semantic_embedding = True
cluster_by_attr = False
device_map = 'cuda:0'
## Cluster Parameter
cluster_division_number = 20
select_number = 1
labelling_budget = 20

if __name__ == '__main__':
    ### Load clean and dirty table
    ### clean table is only used for annotation within labelling budget
    loader = DatasetLoader(base_path=base_path)
    dirty_table, clean_table = loader.load_dataset(dataset_name)
    if generate_GSL_file:
 # 实例化 DataProcessor 类
        processor = DataProcessor(dirty_table=dirty_table, dataset_name=dataset_name)

        # 调用 process_and_save 方法来执行数据处理和文件保存
        processor.process_and_save()
    if remove_GSL_cache:
        folder_path = "PyG_Dataset/{}/processed".format(dataset_name)
        if os.path.exists(folder_path):
            try:
                shutil.rmtree(folder_path)
                print(f"文件夹 '{folder_path}' 及其内容已成功删除。")
            except OSError as e:
                print(f"删除文件夹 '{folder_path}' 失败: {e}")
        else:
            print(f"文件夹 '{folder_path}' 不存在。")
    ## Loading Dataset
    dataset = GSLDataset('PyG_Dataset/{}'.format(dataset_name),dataset_name=dataset_name,model_name=embedding_model_path)

    device = torch.device(device_map)
    ### Start Training
    
    trainer = GraphTrainer(data=dataset[0], hidden_channels=128, learning_rate=0.01, epochs=200, device=device) # 
    
    trainer.train_model()

    triple_embedding = trainer.get_triple_embedding()
    
    ## Start Semantic Embedding
    
    if add_semantic_embedding:
        embedder = SemanticEmbedder(
        embedding_model_path=embedding_model_path,
        use_fp16=False,
        device=device_map # Or 'cpu' if no CUDA
        )   
        semantic_embeddings_matrix = embedder.generate_embeddings(dirty_table=dirty_table)
    
        triple_embedding = np.concatenate((triple_embedding,semantic_embeddings_matrix),axis=1)
    # print(triple_embedding.shape)
    
    ## Start Clustering
    if cluster_by_attr:
        clusters = cluster_by_attribute(dirty_table,cluster_number=cluster_division_number)
        analyzer = ClusterAnalyzer(embeddings=triple_embedding)
        analyzed_clusters = analyzer.analyze_clusters(clusters=clusters)

    else:
        clusterer = KMeansClusterer(n_clusters=cluster_division_number, random_state=42)
        analyzed_clusters = clusterer.perform_clustering(embeddings=triple_embedding)
    sort_clusters_by_members_length(analyzed_clusters)
    clusters = analyzed_clusters
    ### Annotating
    outlier = []
    outlier_index_list = []
    for k in range(min(len(clusters),labelling_budget)):
        select_num = select_number
        cluster_count = 0
        for i in range(len(clusters[k]['members'])):
            outlier_index = clusters[k]['members'][i] ## outlier node index,  after sorted by distance
            ground_truth = np.array(clean_table.iloc[outlier_index]!=dirty_table.iloc[outlier_index]).astype(int)
            if(np.sum(ground_truth)!=0):
                outlier.append(ground_truth)
                outlier_index_list.append(outlier_index)
                cluster_count += 1
            if(cluster_count>=select_num):
                break
    ### Saving Results
    os.makedirs('output/{}/GSL'.format(dataset_name), exist_ok=True)
    np.save('output/{}/GSL/cluster.npy'.format(dataset_name),clusters)
    np.save('output/{}/GSL/index.npy'.format(dataset_name),outlier_index_list)
    
    print(np.array(outlier).sum(axis=0),np.array(outlier).sum(),np.where(np.array(outlier).sum(axis=0))[0].shape,np.where(np.array(dirty_table!=clean_table).sum(axis=0))[0].shape)
