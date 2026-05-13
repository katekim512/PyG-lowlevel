import argparse

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_sparse import SparseTensor, fill_diag, matmul, mul, sum as sparsesum

from gcn_utils import add_training_args, load_planetoid_dataset, run_training


def build_normalized_sparse_tensor(edge_index: Tensor, num_nodes: int) -> SparseTensor:
    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
    adj = fill_diag(adj, 1.0)

    deg = sparsesum(adj, dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(torch.isinf(deg_inv_sqrt), 0.0)

    adj = mul(adj, deg_inv_sqrt.view(-1, 1))
    adj = mul(adj, deg_inv_sqrt.view(1, -1))
    return adj


class SpatialGCNLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.lin.reset_parameters()
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, adj_t: SparseTensor) -> Tensor:
        support = self.lin(x)
        out = matmul(adj_t, support)
        return out + self.bias


class SpatialGCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.conv1 = SpatialGCNLayer(in_channels, hidden_channels)
        self.conv2 = SpatialGCNLayer(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, data) -> Tensor:
        x = self.conv1(data.x, data.adj_t)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, data.adj_t)
        return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spatial GCN implemented with torch_sparse SparseTensor and sparse matmul."
    )
    return add_training_args(parser).parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    dataset, data = load_planetoid_dataset(args.root, args.dataset)
    data.adj_t = build_normalized_sparse_tensor(data.edge_index, data.num_nodes)

    model = SpatialGCN(
        in_channels=dataset.num_features,
        hidden_channels=args.hidden_channels,
        out_channels=dataset.num_classes,
        dropout=args.dropout,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    run_training(model, data, optimizer, args.epochs)


if __name__ == "__main__":
    main()
