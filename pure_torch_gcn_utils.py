import argparse
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class GraphData:
    x: Tensor
    y: Tensor
    edge_index: Tensor
    train_mask: Tensor
    val_mask: Tensor
    test_mask: Tensor

    @property
    def num_nodes(self) -> int:
        return self.x.size(0)

    @property
    def num_features(self) -> int:
        return self.x.size(1)

    @property
    def num_classes(self) -> int:
        return int(self.y.max().item()) + 1


def add_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nodes-per-class", type=int, default=40)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--feature-dim", type=int, default=16)
    parser.add_argument("--intra-class-prob", type=float, default=0.30)
    parser.add_argument("--inter-class-prob", type=float, default=0.03)
    parser.add_argument("--train-ratio", type=float, default=0.5)
    parser.add_argument("--val-ratio", type=float, default=0.25)
    return parser


def make_toy_graph(
    num_classes: int,
    nodes_per_class: int,
    feature_dim: int,
    intra_class_prob: float,
    inter_class_prob: float,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> GraphData:
    generator = torch.Generator().manual_seed(seed)
    num_nodes = num_classes * nodes_per_class

    y = torch.arange(num_classes).repeat_interleave(nodes_per_class)

    class_centers = torch.randn(num_classes, feature_dim, generator=generator) * 2.0
    noise = torch.randn(num_nodes, feature_dim, generator=generator) * 0.6
    x = class_centers[y] + noise

    node_ids = torch.arange(num_nodes)
    src, dst = torch.meshgrid(node_ids, node_ids, indexing="ij")
    same_class = y[src] == y[dst]

    probs = torch.full((num_nodes, num_nodes), inter_class_prob)
    probs[same_class] = intra_class_prob
    probs.fill_diagonal_(0.0)

    sampled = torch.rand((num_nodes, num_nodes), generator=generator) < probs
    adjacency = torch.logical_or(sampled, sampled.t())
    adjacency.fill_diagonal_(0.0)

    edge_index = adjacency.nonzero(as_tuple=False).t().contiguous()

    permutation = torch.randperm(num_nodes, generator=generator)
    train_cutoff = int(num_nodes * train_ratio)
    val_cutoff = int(num_nodes * (train_ratio + val_ratio))

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[permutation[:train_cutoff]] = True
    val_mask[permutation[train_cutoff:val_cutoff]] = True
    test_mask[permutation[val_cutoff:]] = True

    return GraphData(
        x=x,
        y=y,
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


def train_step(model: nn.Module, data: GraphData, optimizer: torch.optim.Optimizer) -> float:
    model.train()
    optimizer.zero_grad()
    logits = model(data)
    loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model: nn.Module, data: GraphData) -> dict[str, float]:
    model.eval()
    logits = model(data)
    pred = logits.argmax(dim=-1)

    metrics = {}
    for split_name in ("train", "val", "test"):
        mask = getattr(data, f"{split_name}_mask")
        correct = (pred[mask] == data.y[mask]).sum().item()
        metrics[split_name] = correct / int(mask.sum())
    return metrics


def run_training(model: nn.Module, data: GraphData, optimizer: torch.optim.Optimizer, epochs: int) -> None:
    best_val = 0.0
    best_test = 0.0

    for epoch in range(1, epochs + 1):
        loss = train_step(model, data, optimizer)
        metrics = evaluate(model, data)

        if metrics["val"] > best_val:
            best_val = metrics["val"]
            best_test = metrics["test"]

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:03d} | "
                f"Loss: {loss:.4f} | "
                f"Train: {metrics['train']:.4f} | "
                f"Val: {metrics['val']:.4f} | "
                f"Test: {metrics['test']:.4f}"
            )

    print(f"Best Val: {best_val:.4f} | Test at Best Val: {best_test:.4f}")
