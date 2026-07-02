from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from .config import cfg_get, load_config, parse_args
from .cui import build_vocabulary, filter_to_vocab
from .datasets import load_split_records
from .metrics import evaluate_rankings, per_query_metrics
from .retrieval import rank_by_similarity, cosine_similarity
from .utils import ensure_dir, write_dataframe


def _split_file(cfg: Dict[str, Any], split: str, kind: str) -> str | None:
    return cfg_get(cfg, f"dataset.{kind}_files.{split}")


def evaluate_saved_embeddings(cfg: Dict[str, Any], embedding_dir: str | Path, output: str | Path | None = None) -> Dict[str, float]:
    embedding_dir = Path(embedding_dir)
    data_root = cfg_get(cfg, "dataset.data_root")
    train_records = load_split_records(data_root, "train", _split_file(cfg, "train", "caption"), _split_file(cfg, "train", "concept"))
    query_split = cfg_get(cfg, "evaluation.query_split", "test")
    db_split = cfg_get(cfg, "evaluation.retrieval_pool", query_split)
    query_records = load_split_records(data_root, query_split, _split_file(cfg, query_split, "caption"), _split_file(cfg, query_split, "concept"))
    db_records = query_records if db_split == query_split else load_split_records(data_root, db_split, _split_file(cfg, db_split, "caption"), _split_file(cfg, db_split, "concept"))
    vocab, vocab_index, _ = build_vocabulary([r.cuis for r in train_records], int(cfg_get(cfg, "dataset.min_cui_freq", 5)), int(cfg_get(cfg, "dataset.max_vocabulary_cap", cfg_get(cfg, "dataset.retained_vocabulary_size_M", 2048))))
    for records in [query_records, db_records]:
        for r in records:
            r.cuis = filter_to_vocab(r.cuis, vocab_index)
    q = np.load(embedding_dir / "query_embeddings.npy")
    d = np.load(embedding_dir / "database_embeddings.npy")
    ranked_path = embedding_dir / "ranked_indices.npy"
    ranked = np.load(ranked_path) if ranked_path.exists() else rank_by_similarity(cosine_similarity(q, d), bool(cfg_get(cfg, "evaluation.same_split_self_exclude", True)) and query_split == db_split)
    metrics = evaluate_rankings([r.cuis for r in query_records], ranked, [r.cuis for r in db_records], cfg_get(cfg, "evaluation.k_values", [5]))
    if output:
        output = ensure_dir(output)
        write_dataframe(pd.DataFrame([metrics]), Path(output) / "metrics.csv")
        write_dataframe(per_query_metrics([r.image_id for r in query_records], [r.cuis for r in query_records], ranked, [r.image_id for r in db_records], [r.cuis for r in db_records], 5), Path(output) / "per_query_metrics.csv")
    return metrics


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    embedding_dir = args.output or cfg_get(cfg, "paths.output_dir")
    metrics = evaluate_saved_embeddings(cfg, embedding_dir, embedding_dir)
    print(pd.DataFrame([metrics]).to_string(index=False))


if __name__ == "__main__":
    main()
