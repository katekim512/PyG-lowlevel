import argparse

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pure_torch_gcn_utils import add_training_args, make_toy_graph, run_training


def add_self_loops(edge_index: Tensor, num_nodes: int) -> Tensor:
    loops = torch.arange(num_nodes, device=edge_index.device)
    loop_edges = torch.stack([loops, loops], dim=0)
    return torch.cat([edge_index, loop_edges], dim=1)


def compute_gcn_norm(edge_index: Tensor, num_nodes: int) -> tuple[Tensor, Tensor]:
    edge_index = add_self_loops(edge_index, num_nodes)
    src, dst = edge_index

    deg = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device)
    deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))

    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    norm = deg_inv_sqrt[dst] * deg_inv_sqrt[src]
    return edge_index, norm


class SpatialGCNLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.lin.reset_parameters()
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, edge_index: Tensor, norm: Tensor) -> Tensor:
        src, dst = edge_index
        support = self.lin(x)

        messages = norm.unsqueeze(-1) * support[src]
        out = torch.zeros_like(support)
        out.index_add_(0, dst, messages)
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
        x = self.conv1(data.x, data.edge_index_with_loops, data.norm)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, data.edge_index_with_loops, data.norm)
        return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pure PyTorch spatial GCN with explicit edge-wise message aggregation."
    )
    return add_training_args(parser).parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    data = make_toy_graph(
        num_classes=args.num_classes,
        nodes_per_class=args.nodes_per_class,
        feature_dim=args.feature_dim,
        intra_class_prob=args.intra_class_prob,
        inter_class_prob=args.inter_class_prob,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    data.edge_index_with_loops, data.norm = compute_gcn_norm(data.edge_index, data.num_nodes)

    model = SpatialGCN(
        in_channels=data.num_features,
        hidden_channels=args.hidden_channels,
        out_channels=data.num_classes,
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
