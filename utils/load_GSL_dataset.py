import torch
from FlagEmbedding import FlagModel
from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from typing import Callable, List, Optional

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