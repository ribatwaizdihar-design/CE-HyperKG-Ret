from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .cui import jaccard


def relevance(query_cuis: Sequence[str], candidate_cuis: Sequence[str]) -> bool:
    return len(set(query_cuis) & set(candidate_cuis)) > 0


def average_precision(binary_relevance: Sequence[int], k: int) -> float:
    hits = 0
    score = 0.0
    for i, rel in enumerate(binary_relevance[:k], start=1):
        if rel:
            hits += 1
            score += hits / i
    return score / max(1, hits)


def reciprocal_rank(binary_relevance: Sequence[int], k: int) -> float:
    for i, rel in enumerate(binary_relevance[:k], start=1):
        if rel:
            return 1.0 / i
    return 0.0


def ndcg_from_gains(gains: Sequence[float], k: int) -> float:
    gains = np.asarray(gains[:k], dtype=np.float32)
    if gains.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    dcg = float((gains * discounts).sum())
    ideal = np.sort(gains)[::-1]
    idcg = float((ideal * discounts).sum())
    return dcg / idcg if idcg > 0 else 0.0


def metrics_for_query(query_cuis: Sequence[str], ranked_indices: Sequence[int], db_cui_sets: Sequence[Sequence[str]], k: int = 5) -> Dict[str, float]:
    top = list(ranked_indices[:k])
    binary = [1 if relevance(query_cuis, db_cui_sets[i]) else 0 for i in top]
    jac = [jaccard(query_cuis, db_cui_sets[i]) for i in top]
    relevant_total = sum(1 for cuis in db_cui_sets if relevance(query_cuis, cuis))
    precision = float(np.mean(binary)) if binary else 0.0
    recall = float(sum(binary) / max(1, relevant_total))
    return {
        f"cui_at_{k}": float(any(binary)),
        f"precision_at_{k}": precision,
        f"recall_at_{k}": recall,
        f"map_at_{k}": average_precision(binary, k),
        f"mrr_at_{k}": reciprocal_rank(binary, k),
        f"ndcg_at_{k}": ndcg_from_gains(jac, k),
        f"mean_jaccard_at_{k}": float(np.mean(jac)) if jac else 0.0,
    }


def evaluate_rankings(query_cui_sets: Sequence[Sequence[str]], ranked_indices: np.ndarray, db_cui_sets: Sequence[Sequence[str]], k_values: Sequence[int] = (5,)) -> Dict[str, float]:
    rows: List[Dict[str, float]] = []
    for q_idx, q_cuis in enumerate(query_cui_sets):
        row: Dict[str, float] = {}
        for k in k_values:
            row.update(metrics_for_query(q_cuis, ranked_indices[q_idx], db_cui_sets, k))
        rows.append(row)
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    return {c: float(df[c].mean()) for c in df.columns}


def per_query_metrics(query_ids: Sequence[str], query_cui_sets: Sequence[Sequence[str]], ranked_indices: np.ndarray, db_ids: Sequence[str], db_cui_sets: Sequence[Sequence[str]], k: int = 5) -> pd.DataFrame:
    rows = []
    for q_idx, q_cuis in enumerate(query_cui_sets):
        idx = ranked_indices[q_idx]
        m = metrics_for_query(q_cuis, idx, db_cui_sets, k)
        row = {"query_id": query_ids[q_idx], "retrieved_ids": " ".join(db_ids[i] for i in idx[:k])}
        row.update(m)
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_metric(values: Sequence[float], rounds: int = 10000, seed: int = 42) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float32)
    rng = np.random.default_rng(seed)
    if values.size == 0:
        return {"mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    samples = rng.choice(values, size=(rounds, values.size), replace=True).mean(axis=1)
    return {"mean": float(values.mean()), "std": float(samples.std(ddof=1)), "ci_low": float(np.percentile(samples, 2.5)), "ci_high": float(np.percentile(samples, 97.5))}


def summarize_seed_metrics(paths: Sequence[str]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    metric_cols = [c for c in df.columns if c not in {"seed", "model", "config"}]
    rows = []
    for metric in metric_cols:
        rows.append({"metric": metric, "mean": float(df[metric].mean()), "std": float(df[metric].std(ddof=1))})
    return pd.DataFrame(rows)
