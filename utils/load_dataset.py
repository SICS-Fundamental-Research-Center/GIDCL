import numpy as np
import pandas as pd
import os
from tqdm import tqdm
tqdm.pandas(desc='pandas bar')

import os.path as osp
from typing import Callable, List, Optional

import torch
from FlagEmbedding import FlagModel
from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
)
class DatasetLoader:
    def __init__(self, base_path='GEIL_Data'):
        self.base_path = base_path

    def load_dataset(self, dataset_name):
        dirty_table = None
        clean_table = None

        if dataset_name.lower() == 'beers':
            beer_clean = pd.read_csv(os.path.join(self.base_path, 'beers', 'original', 'clean.csv')).fillna('')
            beer_dirty = pd.read_csv(os.path.join(self.base_path, 'beers', 'original', 'dirty.csv')).fillna('')
            
            beer_dirty.columns = beer_clean.columns
            
            def try_convert_to_int(row):
                for x,y in row.items():
                    if(x in ['ounces','ibu']):
                        try:
                            row[x] = int(y)
                        except:
                            row[x] = y
                return row
            
            beer_dirty = beer_dirty.apply(try_convert_to_int,axis=1).astype(str)
            beer_clean = beer_clean.apply(try_convert_to_int,axis=1).astype(str)
            dirty_table = beer_dirty
            clean_table = beer_clean
            
        elif dataset_name.lower() == 'hospital':
            hospital_clean = pd.read_csv('GEIL_Data/hospital/original/clean.csv').astype(str)
            hospital_dirty = pd.read_csv('GEIL_Data/hospital/original/dirty.csv').astype(str)
            hospital_dirty.columns = hospital_clean.columns
            dirty_table = hospital_dirty
            clean_table = hospital_clean
            
        elif dataset_name.lower() == 'flights':
            beer_clean = pd.read_csv(os.path.join(self.base_path, 'flights', 'original', 'clean.csv')).fillna('')
            beer_dirty = pd.read_csv(os.path.join(self.base_path, 'flights', 'original', 'dirty.csv')).fillna('')
            
            beer_dirty.columns = beer_clean.columns
            dirty_table = beer_dirty
            clean_table = beer_clean
        
        elif dataset_name.lower() == 'rayyan':
            rayyan_clean = pd.read_csv('GEIL_Data/rayyan/original/clean.csv').fillna('')
            rayyan_dirty = pd.read_csv('GEIL_Data/rayyan/original/dirty.csv').fillna('')
            def Str2Int(row):
                for index in range(11):
                    temp = row[index]
                    try:
                        row[index] = str(int(temp))
                    except:
                        continue
                return row
            rayyan_clean = rayyan_clean.apply(Str2Int,axis=1)
            rayyan_dirty = rayyan_dirty.apply(Str2Int,axis=1)
            dirty_table = rayyan_dirty
            clean_table = rayyan_clean
            
        elif dataset_name.lower() == 'inpatient':
            inpatient_clean = pd.read_csv('BClean-main/dataset/Inpatient/Inpatient_clean.csv')
            inpatient_dirty = pd.read_csv('BClean-main/dataset/Inpatient/Inpatient_dirty_10.csv',index_col=0).fillna('')
            def Str2Int(row):
                for index in range(11):
                    temp = row[index]
                    try:
                        row[index] = str(int(temp))
                    except:
                        continue
                return row
            inpatient_clean = inpatient_clean.apply(Str2Int,axis=1)
            inpatient_dirty = inpatient_dirty.apply(Str2Int,axis=1)
            dirty_table = inpatient_dirty
            clean_table = inpatient_clean
            
        elif dataset_name.lower() == 'facilities':
            inpatient_clean = pd.read_csv('BClean-main/dataset/facilities/facilities_clean.csv')
            inpatient_dirty = pd.read_csv('BClean-main/dataset/facilities/facilities_dirty10.csv',index_col=0).fillna('')
            def Str2Int(row):
                for index in range(10):
                    temp = row[index]
                    try:
                        row[index] = str(int(temp))
                    except:
                        continue
                return row
            inpatient_clean = inpatient_clean.apply(Str2Int,axis=1)
            inpatient_dirty = inpatient_dirty.apply(Str2Int,axis=1)
            dirty_table = inpatient_dirty
            clean_table = inpatient_clean
            
        if dataset_name.lower() == 'tax':
            tax_clean = pd.read_csv('GEIL_Data/tax/original/clean.csv').fillna('').astype(str)
            tax_dirty = pd.read_csv('GEIL_Data/tax/original/dirty.csv').fillna('').astype(str)
            dirty_table = tax_dirty
            clean_table = tax_clean
    
        elif dataset_name.lower() == 'imdb':
            imdb_clean = pd.read_csv('GEIL_Data/imdb/original/clean.csv').fillna('')
            imdb_dirty = pd.read_csv('GEIL_Data/imdb/original/dirty.csv').fillna('')
            def Str2Int(row):
                for index in range(6):
                    temp = row[index]
                    try:
                        row[index] = str(int(temp))
                    except:
                        continue
                return row
            imdb_clean = imdb_clean.apply(Str2Int,axis=1)
            imdb_dirty = imdb_dirty.apply(Str2Int,axis=1)
            dirty_table = imdb_dirty
            clean_table = imdb_clean
        
        return dirty_table, clean_table

class GSLDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        dataset_name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        model_name: Optional[str] = '../sentence_transformer_model/bge-large-en-1.5/',
        force_reload: bool = False,
    ) -> None:
        self.model_name = model_name
        self.dataset_name = dataset_name
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self) -> List[str]:
        return [
            osp.join(self.dataset_name, 'entity_df.csv'),
            osp.join(self.dataset_name, 'triple_df.csv'),
        ]

    @property
    def processed_file_names(self) -> str:
        return f'data_{self.model_name}.pt'


    def process(self) -> None:
        import pandas as pd

        data = HeteroData()

        df = pd.read_csv(self.raw_paths[0], index_col='movieId')
        movie_mapping = {idx: i for i, idx in enumerate(df.index)}

        model = FlagModel(self.model_name,use_fp16=False,devices='cuda:0')
        emb = model.encode(list(df['title'].astype(str).values),convert_to_numpy=False)
        data['movie'].x = torch.cat([emb], dim=-1)

        df = pd.read_csv(self.raw_paths[1])
        print(len(df))
        user_mapping = {idx: i for i, idx in enumerate(df['userId'].unique())}
        data['user'].num_nodes = len(user_mapping)

        src = [user_mapping[idx] for idx in df['userId']]
        dst = [movie_mapping[idx] for idx in df['movieId']]
        edge_index = torch.tensor([src, dst])

        rating = torch.from_numpy(df['rating'].values).to(torch.long)

        data['user', 'rates', 'movie'].edge_index = edge_index
        data['user', 'rates', 'movie'].edge_label = rating

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])