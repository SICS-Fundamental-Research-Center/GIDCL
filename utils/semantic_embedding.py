import pandas as pd
from FlagEmbedding import FlagModel
import os
import json # Import json for converting dictionary to string

class SemanticEmbedder:
    """
    一个用于从脏数据表中生成语义嵌入矩阵的类。

    该类使用 FlagModel（通常是 Sentence Transformer 模型）
    将表格数据转换为文本序列，然后生成这些文本的语义嵌入。
    """

    def __init__(self, embedding_model_path: str, use_fp16: bool = False, device: str = 'cuda:0'):
        """
        初始化 SemanticEmbedder。

        Args:
            embedding_model_path (str): 预训练的 FlagModel 嵌入模型的本地路径。
            use_fp16 (bool): 是否使用 FP16 (半精度浮点数) 进行推理以节省内存和加速。
                             请确保您的硬件支持 FP16。默认为 False。
            device (str): 用于模型推理的设备，例如 'cuda:0' 或 'cpu'。默认为 'cuda:0'。
        """
        if not os.path.exists(embedding_model_path):
            raise FileNotFoundError(f"嵌入模型路径不存在: {embedding_model_path}")
        if not isinstance(use_fp16, bool):
            raise TypeError("use_fp16 必须是布尔值。")
        if not isinstance(device, str):
            raise TypeError("device 必须是字符串。")

        self.model = FlagModel(embedding_model_path, use_fp16=use_fp16, devices=device)
        print(f"FlagModel 已加载，使用设备: {device}")

    def generate_embeddings(self, dirty_table: pd.DataFrame):
        """
        从输入的脏数据表生成语义嵌入。

        该方法遍历 dirty_table 的每一行，将其转换为一个包含列名和值的字典，
        然后将该字典转化为 JSON 字符串，最后使用加载的 FlagModel 对这些字符串生成嵌入。
        该过程会跳过列名中包含“index”的列。

        Args:
            dirty_table (pd.DataFrame): 包含原始“脏”数据的 Pandas DataFrame。

        Returns:
            numpy.ndarray: 包含每行语义嵌入的矩阵。
        """
        if not isinstance(dirty_table, pd.DataFrame):
            raise TypeError("dirty_table 必须是一个 pandas DataFrame。")
        if dirty_table.empty:
            print("Warning: dirty_table 为空，将返回空嵌入矩阵。")
            return None

        semantic_encode_list = []
        for _, row in dirty_table.iterrows():
            row_dict = {}
            # Iterate through columns, skipping those containing 'index' in their name (case-insensitive)
            # and skipping the first column (index 0) if it's implicitly acting as an ID.
            # Here, `row.items()` gives (column_name, value) pairs.
            for col_name, value in row.items():
                if "index" in str(col_name).lower(): # Check if 'index' is in the column name (case-insensitive)
                    continue # Skip this column

                # Convert value to string to ensure json.dumps can handle it.
                # For example, if value is a list or another complex type.
                row_dict[str(col_name)] = str(value)

            # Convert the dictionary to a JSON string.
            # json.dumps ensures proper formatting for embedding models.
            # ensure_ascii=False for non-ASCII characters (e.g., Chinese)
            # separators=(',', ':') removes extra whitespace for more compact string
            text = json.dumps(row_dict, ensure_ascii=False, separators=(',', ':'))
            semantic_encode_list.append(text)

        print(f"已生成 {len(semantic_encode_list)} 个文本字符串进行编码。")
        semantic_embedding = self.model.encode(semantic_encode_list)
        print(f"语义嵌入矩阵形状: {semantic_embedding.shape}")

        return semantic_embedding