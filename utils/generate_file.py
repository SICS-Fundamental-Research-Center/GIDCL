import pandas as pd
import os

class DataProcessor:
    def __init__(self, dirty_table: pd.DataFrame, dataset_name: str):
        if not isinstance(dirty_table, pd.DataFrame):
            raise TypeError("dirty_table 必须是一个 pandas DataFrame。")
        if not isinstance(dataset_name, str):
            raise TypeError("dataset_name 必须是一个字符串。")

        self.dirty_table = dirty_table
        self.dataset_name = dataset_name
        self.root_dir = 'PyG_Dataset'

    def process_and_save(self):
        print(f"开始处理数据集: {self.dataset_name}")

        all_data = list(self.dirty_table.columns)
        all_data.extend(list(set(self.dirty_table.values.flatten())))

        unique_dict = {item: index for index, item in enumerate(all_data)}

        triple_list = []
        for index, row in self.dirty_table.iterrows():
            for x, y in row.items():
                triple_list.append([index, float(unique_dict[x]), unique_dict[y]])

        triple_pd = pd.DataFrame(triple_list)
        entity_df = pd.DataFrame(all_data)
        entity_df.columns = ['title']
        entity_df['movieId'] = entity_df.index

        triple_pd.columns = ['userId', 'rating', 'movieId']

        offset = len(self.dirty_table)
        triple_pd['movieId'] = triple_pd['movieId'] + offset
        entity_df['movieId'] = entity_df['movieId'] + offset

        raw_dir = os.path.join(self.root_dir, self.dataset_name, 'raw', self.dataset_name)
        processed_dir = os.path.join(self.root_dir, self.dataset_name, 'processed')

        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)

        entity_file_path = os.path.join(raw_dir, 'entity_df.csv')
        triple_file_path = os.path.join(raw_dir, 'triple_df.csv')

        entity_df.to_csv(entity_file_path, index=False)
        triple_pd.to_csv(triple_file_path, index=False)

        print(f"数据处理完成。文件已保存到：")
        print(f" - 实体文件: {entity_file_path}")
        print(f" - 三元组文件: {triple_file_path}")