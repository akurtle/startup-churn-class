from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class StartupMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


@dataclass
class MLPTrainingConfig:
    epochs: int = 45
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


def train_mlp(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    *,
    seed: int,
    config: MLPTrainingConfig | None = None,
) -> StartupMLP:
    cfg = config or MLPTrainingConfig()
    torch.manual_seed(seed)

    model = StartupMLP(input_dim=train_features.shape[1])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    positives = float(train_targets.sum())
    negatives = float(len(train_targets) - positives)
    pos_weight = torch.tensor([max(negatives / max(positives, 1.0), 1.0)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    dataset = TensorDataset(
        torch.tensor(train_features, dtype=torch.float32),
        torch.tensor(train_targets, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model.train()
    for _ in range(cfg.epochs):
        for batch_features, batch_targets in loader:
            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()

    return model


def predict_probabilities(model: StartupMLP, features: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(features, dtype=torch.float32))
        return torch.sigmoid(logits).cpu().numpy()
