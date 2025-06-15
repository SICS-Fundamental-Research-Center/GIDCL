import pandas as pd
import os

class DataProcessor:
    """
    处理脏数据表并生成用于PyG数据集的实体和三元组文件。

    这个类接收一个DataFrame (dirty_table) 和一个数据集名称 (dataset_name)，
    然后将数据转换为特定的格式，并保存为CSV文件，以便后续用于PyG图神经网络。
    """

    def __init__(self, dirty_table: pd.DataFrame, dataset_name: str):
        """
        初始化DataProcessor。

        Args:
            dirty_table (pd.DataFrame): 包含原始“脏”数据的Pandas DataFrame。
            dataset_name (str): 用于组织输出文件的数据集名称。
        """
        if not isinstance(dirty_table, pd.DataFrame):
            raise TypeError("dirty_table 必须是一个 pandas DataFrame。")
        if not isinstance(dataset_name, str):
            raise TypeError("dataset_name 必须是一个字符串。")

        self.dirty_table = dirty_table
        self.dataset_name = dataset_name
        self.root_dir = 'PyG_Dataset' # PyG_Dataset 文件夹位于根目录

    def process_and_save(self):
        """
        执行数据处理并将结果保存到指定路径。

        此方法将：
        1. 从 dirty_table 中提取所有唯一的列名和单元格值。
        2. 为这些唯一项创建唯一的ID映射。
        3. 将原始表转换为三元组列表 (userId, rating, movieId)。
        4. 创建实体DataFrame和三元组DataFrame。
        5. 将处理后的数据保存到 'PyG_Dataset/{dataset_name}/raw/{dataset_name}/'
           和 'PyG_Dataset/{dataset_name}/processed/' 目录中。
        """
        print(f"开始处理数据集: {self.dataset_name}")

        # 收集所有列名和所有扁平化的单元格值
        all_data = list(self.dirty_table.columns)
        all_data.extend(list(set(self.dirty_table.values.flatten())))

        # 为所有唯一的数据项创建ID映射
        unique_dict = {item: index for index, item in enumerate(all_data)}

        triple_list = []
        for index, row in self.dirty_table.iterrows():
            for x, y in row.items():  # x 是列名，y 是单元格值
                # 假设 index 是 userId，unique_dict[x] 是 rating (列名的ID)，unique_dict[y] 是 movieId (单元格值的ID)
                # 根据您原始代码中的 triple_pd.columns = ['userId','rating','movieId']，
                # 这里的x (列名) 被映射为 rating，y (单元格值) 被映射为 movieId。
                # 同时，原始的行索引 (index) 被用作 userId。
                triple_list.append([index, float(unique_dict[x]), unique_dict[y]])

        triple_pd = pd.DataFrame(triple_list)
        entity_df = pd.DataFrame(all_data)
        entity_df.columns = ['title']
        entity_df['movieId'] = entity_df.index

        triple_pd.columns = ['userId', 'rating', 'movieId']

        # 调整 movieId，使其与实体ID不重叠。
        # 这里的 movieId 的偏移量是为了确保它们在实体ID空间中是唯一的，
        # 且与原始dirty_table的行数（作为userId的范围）有所区分。
        offset = len(self.dirty_table)
        triple_pd['movieId'] = triple_pd['movieId'] + offset
        entity_df['movieId'] = entity_df['movieId'] + offset

        ### 写入文件
        # 构建输出目录路径
        raw_dir = os.path.join(self.root_dir, self.dataset_name, 'raw', self.dataset_name)
        processed_dir = os.path.join(self.root_dir, self.dataset_name, 'processed')

        # 创建目录
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True) # 尽管这里没有直接保存processed文件，但根据您的路径存在此目录

        # 保存DataFrame到CSV文件
        entity_file_path = os.path.join(raw_dir, 'entity_df.csv')
        triple_file_path = os.path.join(raw_dir, 'triple_df.csv')

        entity_df.to_csv(entity_file_path, index=False)
        triple_pd.to_csv(triple_file_path, index=False)

        print(f"数据处理完成。文件已保存到：")
        print(f" - 实体文件: {entity_file_path}")
        print(f" - 三元组文件: {triple_file_path}")