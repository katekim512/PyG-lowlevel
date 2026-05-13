import argparse

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.utils import add_self_loops, degree


class LowLevelGCNLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(aggr="add")
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.lin.reset_parameters()
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        x = self.lin(x)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        row, col = edge_index
        deg = degree(col, num_nodes=x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = self.propagate(edge_index, x=x, norm=norm)
        return out + self.bias

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out: Tensor) -> Tensor:
        return aggr_out


class LowLevelGCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.conv1 = LowLevelGCNLayer(in_channels, hidden_channels)
        self.conv2 = LowLevelGCNLayer(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def train_step(model: nn.Module, data, optimizer: torch.optim.Optimizer) -> float:
    model.train()
    optimizer.zero_grad()

    logits = model(data.x, data.edge_index)
    loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model: nn.Module, data) -> dict[str, float]:
    model.eval()
    logits = model(data.x, data.edge_index)
    pred = logits.argmax(dim=-1)

    metrics = {}
    for split_name in ("train", "val", "test"):
        mask = getattr(data, f"{split_name}_mask")
        correct = (pred[mask] == data.y[mask]).sum().item()
        metrics[split_name] = correct / int(mask.sum())
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Low-level PyG example with a custom message-passing layer."
    )
    parser.add_argument("--root", type=str, default="data/Planetoid")
    parser.add_argument("--dataset", type=str, default="Cora")
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    dataset = Planetoid(root=args.root, name=args.dataset)
    data = dataset[0]

    model = LowLevelGCN(
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

    best_val = 0.0
    best_test = 0.0

    for epoch in range(1, args.epochs + 1):
        loss = train_step(model, data, optimizer)
        metrics = evaluate(model, data)

        if metrics["val"] > best_val:
            best_val = metrics["val"]
            best_test = metrics["test"]

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d} | "
                f"Loss: {loss:.4f} | "
                f"Train: {metrics['train']:.4f} | "
                f"Val: {metrics['val']:.4f} | "
                f"Test: {metrics['test']:.4f}"
            )

    print(f"Best Val: {best_val:.4f} | Test at Best Val: {best_test:.4f}")


if __name__ == "__main__":
    main()
