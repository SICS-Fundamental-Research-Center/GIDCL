import numpy as np
import pandas as pd
import os
from tqdm import tqdm
tqdm.pandas(desc='pandas bar')

import os.path as osp
from typing import Callable, List, Optional


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

