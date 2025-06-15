import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.transforms import ToUndirected, RandomLinkSplit
from torch_geometric.nn import SAGEConv, to_hetero

# --- Model internal class definitions (remain outside GraphTrainer for clarity) ---

class GNNEncoder(torch.nn.Module):
    """
    图神经网络编码器，使用 GraphSAGE 聚合。
    """
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        # SAGEConv((-1, -1)) 允许自动推断输入特征维度
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        # Ensure edge_index is indeed a torch.LongTensor with shape [2, num_edges]
        # This is typically handled by PyG's data loading and transforms,
        # but worth noting the expectation.
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class EdgeDecoder(torch.nn.Module):
    """
    边解码器，用于从节点嵌入预测边（如评分）。
    """
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        # Concatenate user and movie node embeddings
        # Ensure 'user' and 'movie' keys exist in z_dict
        z = torch.cat([z_dict['user'][row], z_dict['movie'][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1) # Flatten to a 1D tensor

class Model(torch.nn.Module):
    """
    完整的异构图模型，包含 GNN 编码器和边解码器。
    """
    def __init__(self, hidden_channels):
        super().__init__()
        # Initialize the encoder and decoder.
        # The 'to_hetero' conversion will happen *outside* this class,
        # specifically in the GraphTrainer's _setup_model method.
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        # self.encoder here is assumed to have been wrapped by to_hetero already.
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)

    def get_embeddings(self, x_dict, edge_index_dict):
        """
        获取节点的嵌入表示。
        """
        z_dict = self.encoder(x_dict, edge_index_dict)
        return z_dict

# --- GraphTrainer Class Definition ---

class GraphTrainer:
    """
    一个用于训练异构图神经网络模型并获取节点嵌入的类。
    """

    def __init__(self, data, hidden_channels=128, learning_rate=0.01, epochs=200, device='cpu'):
        """
        初始化 GraphTrainer。
        """
        # ... (input validation as before) ...

        self.data = data.to(device)
        self.hidden_channels = hidden_channels
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = device

        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.weight = None
        self.model = None
        self.optimizer = None

        self._prepare_data()
        self._setup_model() # This is where the crucial change will be

    def _prepare_data(self):
        """
        准备图数据，包括特征初始化、无向化和数据集划分。
        """
        # Initialize user features with identity matrix
        self.data['user'].x = torch.eye(self.data['user'].num_nodes, device=self.device)
        if hasattr(self.data['user'], 'num_nodes'):
            del self.data['user'].num_nodes

        T = ToUndirected()
        self.data = T(self.data)

        # Check and potentially delete edge_label from reverse edge type
        # This part of the logic is retained from your original code.
        if ('movie', 'rev_rates', 'user') in self.data.edge_types and hasattr(self.data['movie', 'rev_rates', 'user'], 'edge_label'):
             del self.data['movie', 'rev_rates', 'user'].edge_label


        splitter = RandomLinkSplit(
            num_val=0.03,
            num_test=0.03,
            neg_sampling_ratio=0.0,
            edge_types=[('user', 'rates', 'movie')],
            rev_edge_types=[('movie', 'rev_rates', 'user')],
        )
        self.train_data, self.val_data, self.test_data = splitter(self.data)

        if ('user', 'movie') in self.train_data.edge_types and hasattr(self.train_data['user', 'movie'], 'edge_label'):
            if self.train_data['user', 'movie'].edge_label.numel() > 0:
                self.weight = torch.bincount(self.train_data['user', 'movie'].edge_label.flatten())
                self.weight = self.weight.max() / (self.weight + 1e-6)
            else:
                self.weight = None
        else:
            self.weight = None

    def _setup_model(self):
        """
        设置 GNN 模型和优化器。
        """
        # Instantiate the Model class first
        self.model = Model(hidden_channels=self.hidden_channels)

        # --- CRITICAL FIX: Apply to_hetero to the encoder here! ---
        # The encoder needs to be transformed to handle heterogeneous data
        # using the metadata from the 'data' object.
        self.model.encoder = to_hetero(self.model.encoder, self.data.metadata(), aggr='sum')

        # Move the entire model to the device
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    @staticmethod
    def weighted_mse_loss(pred, target, weight=None):
        """
        计算加权均方误差损失。
        """
        weight = 1.0 if weight is None else weight[target].to(pred.dtype)
        return (weight * (pred - target.to(pred.dtype)).pow(2)).mean()

    def _train_epoch(self):
        """
        执行一个训练 epoch。
        """
        self.model.train()
        self.optimizer.zero_grad()
        pred = self.model(
            self.train_data.x_dict,
            self.train_data.edge_index_dict,
            self.train_data['user', 'movie'].edge_label_index
        )
        target = self.train_data['user', 'movie'].edge_label
        loss = self.weighted_mse_loss(pred, target, self.weight)
        loss.backward()
        self.optimizer.step()
        return float(loss)

    @torch.no_grad()
    def _test_epoch(self, data):
        """
        在给定数据集上评估模型。
        """
        self.model.eval()
        pred = self.model(
            data.x_dict,
            data.edge_index_dict,
            data['user', 'movie'].edge_label_index
        )
        pred = pred.clamp(min=0, max=5)
        target = data['user', 'movie'].edge_label.float()
        rmse = F.mse_loss(pred, target).sqrt()
        return float(rmse)

    def train_model(self):
        """
        执行模型的完整训练过程。
        """
        print(f"开始训练模型 (epochs: {self.epochs}, hidden_channels: {self.hidden_channels}, lr: {self.learning_rate})")
        for epoch in range(1, self.epochs + 1):
            loss = self._train_epoch()
            train_rmse = self._test_epoch(self.train_data)
            val_rmse = self._test_epoch(self.val_data)
            test_rmse = self._test_epoch(self.test_data)
        #     print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train RMSE: {train_rmse:.4f}, '
        #           f'Val RMSE: {val_rmse:.4f}, Test RMSE: {test_rmse:.4f}')
        # print("模型训练完成。")

    def get_triple_embedding(self):
        """
        获取用户节点的嵌入（即 triple_embedding）。
        """
        if self.model is None:
            raise RuntimeError("模型尚未训练。请先调用 train_model()。")

        self.model.eval()
        with torch.no_grad():
            node_embeddings = self.model.get_embeddings(self.data.x_dict, self.data.edge_index_dict)

        if 'user' in node_embeddings:
            return node_embeddings['user'].detach().cpu().numpy()
        else:
            raise KeyError("'user' 节点类型在生成的嵌入中不存在。请检查模型或数据。")