import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.transforms import ToUndirected, RandomLinkSplit
from torch_geometric.nn import SAGEConv, to_hetero

class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['user'][row], z_dict['movie'][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)

class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)

    def get_embeddings(self, x_dict, edge_index_dict):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return z_dict

class GraphTrainer:
    def __init__(self, data, hidden_channels=128, learning_rate=0.01, epochs=200, device='cpu'):
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
        self._setup_model()

    def _prepare_data(self):
        self.data['user'].x = torch.eye(self.data['user'].num_nodes, device=self.device)
        if hasattr(self.data['user'], 'num_nodes'):
            del self.data['user'].num_nodes

        T = ToUndirected()
        self.data = T(self.data)

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
        self.model = Model(hidden_channels=self.hidden_channels)

        self.model.encoder = to_hetero(self.model.encoder, self.data.metadata(), aggr='sum')

        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    @staticmethod
    def weighted_mse_loss(pred, target, weight=None):
        weight = 1.0 if weight is None else weight[target].to(pred.dtype)
        return (weight * (pred - target.to(pred.dtype)).pow(2)).mean()

    def _train_epoch(self):
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
        print(f"开始训练模型 (epochs: {self.epochs}, hidden_channels: {self.hidden_channels}, lr: {self.learning_rate})")
        for epoch in range(1, self.epochs + 1):
            loss = self._train_epoch()
            train_rmse = self._test_epoch(self.train_data)
            val_rmse = self._test_epoch(self.val_data)
            test_rmse = self._test_epoch(self.test_data)

    def get_triple_embedding(self):
        if self.model is None:
            raise RuntimeError("模型尚未训练。请先调用 train_model()。")

        self.model.eval()
        with torch.no_grad():
            node_embeddings = self.model.get_embeddings(self.data.x_dict, self.data.edge_index_dict)

        if 'user' in node_embeddings:
            return node_embeddings['user'].detach().cpu().numpy()
        else:
            raise KeyError("'user' 节点类型在生成的嵌入中不存在。请检查模型或数据。")