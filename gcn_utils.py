import argparse

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.datasets import Planetoid


def add_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--root", type=str, default="data/Planetoid")
    parser.add_argument("--dataset", type=str, default="Cora")
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def load_planetoid_dataset(root: str, name: str):
    dataset = Planetoid(root=root, name=name)
    return dataset, dataset[0]


def train_step(model: nn.Module, data, optimizer: torch.optim.Optimizer) -> float:
    model.train()
    optimizer.zero_grad()

    logits = model(data)
    loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model: nn.Module, data) -> dict[str, float]:
    model.eval()
    logits = model(data)
    pred = logits.argmax(dim=-1)

    metrics = {}
    for split_name in ("train", "val", "test"):
        mask = getattr(data, f"{split_name}_mask")
        correct = (pred[mask] == data.y[mask]).sum().item()
        metrics[split_name] = correct / int(mask.sum())
    return metrics


def run_training(model: nn.Module, data, optimizer: torch.optim.Optimizer, epochs: int) -> None:
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
