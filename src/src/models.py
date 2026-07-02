from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import EncoderWrapper


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 512, dropout: float = 0.20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)


class CEHyperKGRetModel(nn.Module):
    def __init__(self, encoder: EncoderWrapper, num_cuis: int, projection_dim: int = 512, dropout: float = 0.20):
        super().__init__()
        self.encoder = encoder
        self.projection = ProjectionHead(encoder.output_dim, projection_dim, dropout)
        self.cui_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(projection_dim, num_cuis),
        )

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.encoder(image)
        embedding = self.projection(features)
        logits = self.cui_head(embedding)
        return {"features": features, "embedding": embedding, "logits": logits, "probs": torch.sigmoid(logits)}


def freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = True


def configure_trainable_layers(model: CEHyperKGRetModel, mode: str = "all", last_n_children: int = 3) -> None:
    if mode == "all":
        unfreeze_module(model.encoder)
        return
    if mode == "frozen":
        freeze_module(model.encoder)
        return
    if mode == "last_n":
        freeze_module(model.encoder)
        children = list(model.encoder.model.children())
        for child in children[-last_n_children:]:
            unfreeze_module(child)
        return
    raise ValueError(f"Unsupported fine tune mode {mode}")
