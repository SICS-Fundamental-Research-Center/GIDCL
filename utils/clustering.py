import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, List, Union
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class KMeansClusterer:
    """
    一个用于执行 K-Means 聚类并分析聚类结果的类。

    该类接收一个嵌入矩阵和期望的聚类数量，执行 K-Means 算法，
    并为每个生成的聚类计算其质心、识别离群点，并按到质心的距离排序成员。
    """

    def __init__(self, n_clusters: int, random_state: Union[int, None] = None):
        """
        初始化 KMeansClusterer。

        Args:
            n_clusters (int): 目标聚类的数量。必须为正整数。
            random_state (Union[int, None]): K-Means 算法的随机种子，用于结果复现。
                                               默认为 None，表示不固定随机种子。
        """
        if not isinstance(n_clusters, int) or n_clusters <= 0:
            raise ValueError("n_clusters 必须是正整数。")
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto') # n_init='auto' is recommended for newer scikit-learn versions

    def perform_clustering(self, embeddings: np.ndarray) -> Dict[int, Dict[str, Union[List[int], np.ndarray, int, None]]]:
        """
        在给定的嵌入数据上执行 K-Means 聚类。

        此方法将：
        1. 对 `embeddings` 执行 K-Means 聚类，生成 `n_clusters` 个聚类。
        2. 对于每个聚类，识别其成员、计算质心，并将成员按到质心的距离降序排序。
        3. 识别每个聚类中距离质心最远的成员作为离群点。

        Args:
            embeddings (np.ndarray): 输入的嵌入矩阵，形状应为 (num_samples, embedding_dim)。
                                     例如：triple_embedding, semantic_embedding, 或 add_embedding。

        Returns:
            Dict[int, Dict[str, Union[List[int], np.ndarray, int, None]]]: 
            一个包含聚类结果的字典。字典的键是聚类ID (int)，
            值是另一个字典，包含：
            - 'members': 该聚类中数据点在 `embeddings` 矩阵中的索引列表，按到质心的距离降序排序。
            - 'centroid': 该聚类的质心向量 (np.ndarray)。
            - 'outlier_index': 距离质心最远的数据点的原始索引 (int)。
                              如果聚类为空，则为 None。
        """
        if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
            raise ValueError("embeddings 必须是二维的 NumPy 数组。")
        if embeddings.shape[0] < self.n_clusters:
            raise ValueError(f"样本数量 ({embeddings.shape[0]}) 小于聚类数量 ({self.n_clusters})，无法执行 K-Means。")

        print(f"开始在 {embeddings.shape} 形状的嵌入上执行 K-Means 聚类，目标聚类数: {self.n_clusters}")

        # 执行 K-Means 拟合
        self.kmeans.fit(embeddings)

        centroids = self.kmeans.cluster_centers_
        labels = self.kmeans.labels_

        clusters_result = {}
        for i in range(self.n_clusters):
            # 获取当前聚类的成员索引
            members = np.where(labels == i)[0]
            centroid = centroids[i]

            if members.size > 0:
                # 计算成员到质心的距离
                distances = np.linalg.norm(embeddings[members] - centroid, axis=1)
                
                # 将成员索引和距离配对
                members_distances = list(zip(members.tolist(), distances.tolist())) # Convert to list for sorting

                # 根据距离降序排序成员
                sorted_members_distances = sorted(members_distances, key=lambda x: x[1], reverse=True)
                
                # 解包排序后的成员索引
                sorted_members = [member for member, _ in sorted_members_distances]
                
                # 离群点是距离最远的点
                outlier_index = sorted_members[0]
                
                clusters_result[i] = {
                    'members': sorted_members,
                    'centroid': centroid,
                    'outlier_index': outlier_index
                }
            else:
                # 处理空聚类的情况
                clusters_result[i] = {
                    'members': [],
                    'centroid': centroid, # K-Means 即使聚类为空也会提供质心
                    'outlier_index': None
                }
        print(f"K-Means 聚类完成，生成了 {len(clusters_result)} 个聚类。")
        return clusters_result
    
class ClusterAnalyzer:
    """
    一个用于分析预定义聚类（计算质心和识别离群点）的类。

    该类接收一个嵌入矩阵和一组预先分配好的聚类，
    然后为每个聚类计算其成员的质心，并识别出距离质心最远的成员作为离群点。
    聚类成员的分配不会被改变。
    """

    def __init__(self, embeddings: np.ndarray):
        """
        初始化 ClusterAnalyzer。

        Args:
            embeddings (np.ndarray): 用于计算距离和质心的嵌入矩阵
                                     （例如，triple_embedding, semantic_embedding, 或 add_embedding）。
                                     形状应为 (num_samples, embedding_dim)。
        """
        if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
            raise ValueError("embeddings 必须是二维的 NumPy 数组。")
        self.embeddings = embeddings

    def analyze_clusters(self, clusters: Dict[int, Dict[str, List[int]]]) -> Dict[int, Dict[str, Union[List[int], np.ndarray, int, None]]]:
        """
        分析预定义的聚类，计算质心并识别离群点。

        该方法遍历输入的聚类字典。对于每个聚类：
        1. 根据其 'members' 列表从 embeddings 中提取对应的向量。
        2. 计算这些成员向量的平均值作为聚类质心。
        3. 计算每个成员到质心的欧氏距离。
        4. 识别距离质心最远的成员的索引作为离群点。
        5. 在不改变原始成员列表的前提下，将成员按到质心的距离降序排序。

        Args:
            clusters (Dict[int, Dict[str, List[int]]]): 一个字典，其中键是聚类ID (int)，
                                                     值是另一个字典，包含至少一个键 'members'，
                                                     其值为该聚类中数据点在 embeddings 矩阵中的索引列表。
                                                     例如: {0: {'members': [10, 25, 30]}, ...}

        Returns:
            Dict[int, Dict[str, Union[List[int], np.ndarray, int, None]]]: 
            一个更新后的聚类字典，每个聚类包含：
            - 'members': 原始的成员索引，但按到质心的距离降序排序。
            - 'centroid': 该聚类的质心向量 (np.ndarray)。
            - 'outlier_index': 距离质心最远的数据点的原始索引 (int)。
                              如果聚类为空，则为 None。
        """
        if not isinstance(clusters, dict):
            raise TypeError("clusters 必须是一个字典。")

        updated_clusters = {}
        for cluster_id, cluster_info in clusters.items():
            members_indices = cluster_info.get('members', [])

            if not isinstance(members_indices, list):
                raise ValueError(f"聚类ID {cluster_id} 的 'members' 必须是一个列表。")

            if len(members_indices) > 0:
                # 提取聚类成员的嵌入向量
                cluster_members_embeddings = self.embeddings[members_indices]

                # 计算质心
                centroid = np.mean(cluster_members_embeddings, axis=0)

                # 计算所有成员到质心的距离
                distances = np.linalg.norm(cluster_members_embeddings - centroid, axis=1)

                # 将成员索引和它们的距离配对
                members_distances = list(zip(members_indices, distances))

                # 根据距离降序排序成员
                # 注意：这里是创建了一个新的排序后的成员列表，原始传入的 members_indices 列表不变
                sorted_members_distances = sorted(members_distances, key=lambda x: x[1], reverse=True)
                sorted_members = [member for member, _ in sorted_members_distances]

                # 离群点是距离最远的点
                outlier_index = sorted_members[0]

                updated_clusters[cluster_id] = {
                    'members': sorted_members, # 返回排序后的成员列表
                    'centroid': centroid,
                    'outlier_index': outlier_index
                }
            else:
                # 处理空聚类的情况
                # 质心仍然可以基于空成员计算 (np.mean([]) 会返回 nan，但这里我们直接使用外部传入的质心或者None)
                # 如果传入的 clusters 字典中没有为这个空的 cluster_id 预设 centroid，
                # 那么这里 centroid 可以是 zeros 或 None。为了与原始代码逻辑更贴近，
                # 如果是空聚类，我们这里就只返回一个 None 的质心或者一个零向量。
                # 由于原始代码中 Kmeans 已经计算了所有k个centroid，
                # 并且这里明确了clusters[i]['members']是传入，
                # 所以这里假设如果members是空，centroid可能已经由外部KMeans计算好传入了，
                # 但为了这个Class的独立性，我们处理为None或者zero vector
                
                # 如果预期的 centroid 也在 cluster_info 中，可以这样获取：
                # centroid = cluster_info.get('centroid', np.zeros(self.embeddings.shape[1])) 
                # 但根据您的要求，这里只计算传入 members 的 centroid
                
                # 对于空聚类，质心和离群点都设为 None 或一个零向量（取决于下游需求）
                # 为了保持输出一致性，对于空聚类，质心可以是一个与嵌入维度匹配的零向量
                # 或者更合理地，如果计算不出，就为 None
                
                # 这里根据原始代码逻辑，如果 members.size > 0 才有 centroid 计算，
                # 所以空聚类时，centroid 设为 None 较为合理，因为没有成员来计算。
                updated_clusters[cluster_id] = {
                    'members': [],
                    'centroid': None,
                    'outlier_index': None
                }
        return updated_clusters