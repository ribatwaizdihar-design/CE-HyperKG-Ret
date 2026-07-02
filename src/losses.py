from typing import Optional

import torch
import torch.nn.functional as F


def multilabel_cui_loss(logits: torch.Tensor, targets: torch.Tensor, pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets.float(), pos_weight=pos_weight)


def supervised_contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    embeddings = F.normalize(embeddings, dim=1)
    logits = embeddings @ embeddings.T / temperature
    n = logits.shape[0]
    eye = torch.eye(n, device=logits.device, dtype=torch.bool)
    shared = (labels.float() @ labels.float().T) > 0
    positives = shared & ~eye
    logits = logits.masked_fill(eye, -1e9)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    denom = positives.sum(dim=1).clamp_min(1)
    loss = -(log_prob * positives.float()).sum(dim=1) / denom
    valid = positives.any(dim=1)
    if not valid.any():
        return embeddings.new_tensor(0.0)
    return loss[valid].mean()


def kg_consistency_loss(probs: torch.Tensor, expanded_probs: torch.Tensor) -> torch.Tensor:
    probs = F.normalize(probs.float(), dim=1)
    expanded_probs = F.normalize(expanded_probs.float(), dim=1)
    return 1.0 - (probs * expanded_probs).sum(dim=1).mean()


def counterfactual_margin_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.20) -> torch.Tensor:
    embeddings = F.normalize(embeddings, dim=1)
    sim = embeddings @ embeddings.T
    shared = (labels.float() @ labels.float().T) > 0
    eye = torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)
    pos_mask = shared & ~eye
    neg_mask = ~shared & ~eye
    losses = []
    for i in range(sim.shape[0]):
        if pos_mask[i].any() and neg_mask[i].any():
            pos = sim[i][pos_mask[i]].max()
            neg = sim[i][neg_mask[i]].max()
            losses.append(F.relu(margin + neg - pos))
    if not losses:
        return embeddings.new_tensor(0.0)
    return torch.stack(losses).mean()
