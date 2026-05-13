import argparse

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.utils import add_self_loops, to_torch_coo_tensor

from gcn_utils import add_training_args, load_planetoid_dataset, run_training


def build_normalized_adjacency(edge_index: Tensor, num_nodes: int) -> Tensor:
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    adj = to_torch_coo_tensor(edge_index, size=(num_nodes, num_nodes)).coalesce()

    row, col = adj.indices()
    deg = torch.sparse.sum(adj, dim=1).to_dense()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0

    norm_values = deg_inv_sqrt[row] * adj.values() * deg_inv_sqrt[col]
    return torch.sparse_coo_tensor(
        indices=adj.indices(),
        values=norm_values,
        size=adj.size(),
        device=adj.device,
    ).coalesce()


class SpectralGCNLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.lin.reset_parameters()
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, adj_norm: Tensor) -> Tensor:
        support = self.lin(x)
        out = torch.sparse.mm(adj_norm, support)
        return out + self.bias


class SpectralGCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.conv1 = SpectralGCNLayer(in_channels, hidden_channels)
        self.conv2 = SpectralGCNLayer(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, data) -> Tensor:
        x = self.conv1(data.x, data.adj_norm)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, data.adj_norm)
        return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spectral GCN implemented as normalized adjacency matrix multiplication."
    )
    return add_training_args(parser).parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    dataset, data = load_planetoid_dataset(args.root, args.dataset)
    data.adj_norm = build_normalized_adjacency(data.edge_index, data.num_nodes)

    model = SpectralGCN(
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
