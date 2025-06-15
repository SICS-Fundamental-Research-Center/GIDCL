import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, List, Union
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class KMeansClusterer:
    def __init__(self, n_clusters: int, random_state: Union[int, None] = None):
        if not isinstance(n_clusters, int) or n_clusters <= 0:
            raise ValueError("n_clusters 必须是正整数。")
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')

    def perform_clustering(self, embeddings: np.ndarray) -> Dict[int, Dict[str, Union[List[int], np.ndarray, int, None]]]:
        if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
            raise ValueError("embeddings 必须是二维的 NumPy 数组。")
        if embeddings.shape[0] < self.n_clusters:
            raise ValueError(f"样本数量 ({embeddings.shape[0]}) 小于聚类数量 ({self.n_clusters})，无法执行 K-Means。")

        print(f"开始在 {embeddings.shape} 形状的嵌入上执行 K-Means 聚类，目标聚类数: {self.n_clusters}")

        self.kmeans.fit(embeddings)

        centroids = self.kmeans.cluster_centers_
        labels = self.kmeans.labels_

        clusters_result = {}
        for i in range(self.n_clusters):
            members = np.where(labels == i)[0]
            centroid = centroids[i]

            if members.size > 0:
                distances = np.linalg.norm(embeddings[members] - centroid, axis=1)
                
                members_distances = list(zip(members.tolist(), distances.tolist()))

                sorted_members_distances = sorted(members_distances, key=lambda x: x[1], reverse=True)
                
                sorted_members = [member for member, _ in sorted_members_distances]
                
                outlier_index = sorted_members[0]
                
                clusters_result[i] = {
                    'members': sorted_members,
                    'centroid': centroid,
                    'outlier_index': outlier_index
                }
            else:
                clusters_result[i] = {
                    'members': [],
                    'centroid': centroid,
                    'outlier_index': None
                }
        print(f"K-Means 聚类完成，生成了 {len(clusters_result)} 个聚类。")
        return clusters_result
    
class ClusterAnalyzer:
    def __init__(self, embeddings: np.ndarray):
        if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
            raise ValueError("embeddings 必须是二维的 NumPy 数组。")
        self.embeddings = embeddings

    def analyze_clusters(self, clusters: Dict[int, Dict[str, List[int]]]) -> Dict[int, Dict[str, Union[List[int], np.ndarray, int, None]]]:
        if not isinstance(clusters, dict):
            raise TypeError("clusters 必须是一个字典。")

        updated_clusters = {}
        for cluster_id, cluster_info in clusters.items():
            members_indices = cluster_info.get('members', [])

            if not isinstance(members_indices, list):
                raise ValueError(f"聚类ID {cluster_id} 的 'members' 必须是一个列表。")

            if len(members_indices) > 0:
                cluster_members_embeddings = self.embeddings[members_indices]

                centroid = np.mean(cluster_members_embeddings, axis=0)

                distances = np.linalg.norm(cluster_members_embeddings - centroid, axis=1)

                members_distances = list(zip(members_indices, distances))

                sorted_members_distances = sorted(members_distances, key=lambda x: x[1], reverse=True)
                sorted_members = [member for member, _ in sorted_members_distances]

                outlier_index = sorted_members[0]

                updated_clusters[cluster_id] = {
                    'members': sorted_members,
                    'centroid': centroid,
                    'outlier_index': outlier_index
                }
            else:
                updated_clusters[cluster_id] = {
                    'members': [],
                    'centroid': None,
                    'outlier_index': None
                }
        return updated_clusters