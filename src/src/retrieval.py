from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .constants import FUSION_RULE
from .cui import binary_jaccard_matrix, selected_binary_matrix
from .graph import expand_probabilities


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return l2_normalize(a) @ l2_normalize(b).T


def rank_by_similarity(scores: np.ndarray, self_exclude: bool = False) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32).copy()
    if self_exclude and scores.shape[0] == scores.shape[1]:
        np.fill_diagonal(scores, -np.inf)
    return np.argsort(-scores, axis=1)


def top_n_visual_candidates(query_embeddings: np.ndarray, db_embeddings: np.ndarray, top_n: int = 200, self_exclude: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    scores = cosine_similarity(query_embeddings, db_embeddings)
    if self_exclude and scores.shape[0] == scores.shape[1]:
        np.fill_diagonal(scores, -np.inf)
    top_n = min(top_n, scores.shape[1])
    idx = np.argpartition(-scores, kth=top_n - 1, axis=1)[:, :top_n]
    local = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-local, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    local = np.take_along_axis(local, order, axis=1)
    return idx, local


def fusion_weights(confidence: float, rule: Sequence[Tuple[float, float, float, float]] = FUSION_RULE) -> Tuple[float, float, float]:
    for upper, alpha, beta, gamma in rule:
        if confidence < upper:
            return alpha, beta, gamma
    _, alpha, beta, gamma = rule[-1]
    return alpha, beta, gamma


def rerank_ce_hyperkg(query_embeddings: np.ndarray, db_embeddings: np.ndarray, query_probs: np.ndarray, db_probs: np.ndarray, propagation: object, top_n: int = 200, threshold: float = 0.30, top_m: int = 10, fusion_rule: Sequence[Tuple[float, float, float, float]] = FUSION_RULE, self_exclude: bool = False) -> Dict[str, np.ndarray]:
    visual_idx, visual_scores = top_n_visual_candidates(query_embeddings, db_embeddings, top_n, self_exclude)
    query_selected = selected_binary_matrix(query_probs, threshold, top_m)
    db_selected = selected_binary_matrix(db_probs, threshold, top_m)
    query_expanded = expand_probabilities(query_selected, propagation)
    db_expanded = expand_probabilities(db_selected, propagation)
    ranked = np.zeros_like(visual_idx)
    final_scores = np.zeros_like(visual_scores)
    visual_components = np.zeros_like(visual_scores)
    pred_components = np.zeros_like(visual_scores)
    hyper_components = np.zeros_like(visual_scores)
    for i in range(query_embeddings.shape[0]):
        cand = visual_idx[i]
        sv = visual_scores[i]
        sp = binary_jaccard_matrix(query_selected[i:i + 1], db_selected[cand]).ravel()
        sh = cosine_similarity(query_expanded[i:i + 1], db_expanded[cand]).ravel()
        alpha, beta, gamma = fusion_weights(float(query_probs[i].max()), fusion_rule)
        score = alpha * sv + beta * sp + gamma * sh
        order = np.argsort(-score)
        ranked[i] = cand[order]
        final_scores[i] = score[order]
        visual_components[i] = sv[order]
        pred_components[i] = sp[order]
        hyper_components[i] = sh[order]
    return {
        "ranked_indices": ranked,
        "final_scores": final_scores,
        "visual_scores": visual_components,
        "predicted_cui_scores": pred_components,
        "hypergraph_scores": hyper_components,
    }


def collate_batch(batch):
    images = torch.stack([item["image"] for item in batch])
    image_ids = [item["image_id"] for item in batch]
    cuis = [item["cuis"] for item in batch]
    out = {"image": images, "image_id": image_ids, "cuis": cuis}
    if "label" in batch[0]:
        labels = torch.tensor(np.stack([item["label"] for item in batch]), dtype=torch.float32)
        out["label"] = labels
    return out


def compute_model_outputs(model: torch.nn.Module, dataset, device: str = "cpu", batch_size: int = 32, num_workers: int = 0) -> Dict[str, np.ndarray | list]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_batch)
    model.eval()
    embeddings = []
    logits = []
    probs = []
    ids = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            out = model(image)
            if isinstance(out, dict):
                emb = out.get("embedding", out.get("features"))
                lg = out.get("logits")
                pr = out.get("probs")
            else:
                emb = out
                lg = None
                pr = None
            embeddings.append(emb.detach().cpu().numpy())
            if lg is not None:
                logits.append(lg.detach().cpu().numpy())
            if pr is not None:
                probs.append(pr.detach().cpu().numpy())
            ids.extend(batch["image_id"])
    result = {"image_id": ids, "embedding": np.concatenate(embeddings, axis=0)}
    if logits:
        result["logits"] = np.concatenate(logits, axis=0)
    if probs:
        result["probs"] = np.concatenate(probs, axis=0)
    return result
