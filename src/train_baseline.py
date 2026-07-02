from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

from .config import cfg_get, load_config, parse_args
from .cui import build_vocabulary, filter_to_vocab
from .datasets import RocoImageDataset, load_split_records, records_to_dataframe
from .encoders import build_encoder_from_config
from .metrics import evaluate_rankings, per_query_metrics
from .validation import validate_baseline_config
from .retrieval import compute_model_outputs, rank_by_similarity, cosine_similarity
from .transforms import build_image_transform, normalization_for_family
from .utils import device_from_arg, ensure_dir, save_json, set_seed, write_dataframe


def _split_file(cfg: Dict, split: str, kind: str) -> str | None:
    return cfg_get(cfg, f"dataset.{kind}_files.{split}")


def run_baseline(cfg: Dict, seed: int | None = None, device: str | None = None) -> Dict[str, float]:
    errors = validate_baseline_config(cfg)
    if errors:
        raise ValueError("Baseline config validation failed: " + "; ".join(errors))
    seed = int(seed if seed is not None else cfg_get(cfg, "training.random_seeds", [42])[0])
    set_seed(seed)
    data_root = cfg_get(cfg, "dataset.data_root")
    output_dir = ensure_dir(cfg_get(cfg, "paths.output_dir", "../results/baseline") + f"/seed_{seed}")
    train_records = load_split_records(data_root, "train", _split_file(cfg, "train", "caption"), _split_file(cfg, "train", "concept"), limit=cfg_get(cfg, "runtime.limit_train"))
    query_split = cfg_get(cfg, "evaluation.query_split", "test")
    db_split = cfg_get(cfg, "evaluation.retrieval_pool", query_split)
    query_records = load_split_records(data_root, query_split, _split_file(cfg, query_split, "caption"), _split_file(cfg, query_split, "concept"), limit=cfg_get(cfg, f"runtime.limit_{query_split}"))
    db_records = query_records if db_split == query_split else load_split_records(data_root, db_split, _split_file(cfg, db_split, "caption"), _split_file(cfg, db_split, "concept"), limit=cfg_get(cfg, f"runtime.limit_{db_split}"))
    vocab, vocab_index, counter = build_vocabulary([r.cuis for r in train_records], int(cfg_get(cfg, "dataset.min_cui_freq", 5)), int(cfg_get(cfg, "dataset.max_vocabulary_cap", 2048)))
    for records in [query_records, db_records]:
        for r in records:
            r.cuis = filter_to_vocab(r.cuis, vocab_index)
    model_cfg = cfg.get("model", {})
    family = model_cfg.get("family", "torchvision")
    image_size = int(model_cfg.get("image_size", 224))
    transform = build_image_transform(image_size, normalization_for_family(family), train=False)
    encoder = build_encoder_from_config(cfg).to(device_from_arg(device))
    device_final = next(encoder.parameters()).device.type
    batch_size = int(cfg_get(cfg, "training.batch_size", 32))
    num_workers = int(cfg_get(cfg, "training.num_workers", 0))
    query_ds = RocoImageDataset(query_records, transform=transform)
    db_ds = RocoImageDataset(db_records, transform=transform)
    q_out = compute_model_outputs(encoder, query_ds, device_final, batch_size, num_workers)
    d_out = q_out if db_records is query_records else compute_model_outputs(encoder, db_ds, device_final, batch_size, num_workers)
    scores = cosine_similarity(q_out["embedding"], d_out["embedding"])
    ranked = rank_by_similarity(scores, bool(cfg_get(cfg, "evaluation.same_split_self_exclude", True)) and db_split == query_split)
    metrics = evaluate_rankings([r.cuis for r in query_records], ranked, [r.cuis for r in db_records], cfg_get(cfg, "evaluation.k_values", [5]))
    metrics.update({"seed": seed, "model": model_cfg.get("display_name", model_cfg.get("tag", "baseline"))})
    write_dataframe(pd.DataFrame([metrics]), Path(output_dir) / "metrics.csv")
    write_dataframe(per_query_metrics([r.image_id for r in query_records], [r.cuis for r in query_records], ranked, [r.image_id for r in db_records], [r.cuis for r in db_records], 5), Path(output_dir) / "per_query_metrics.csv")
    np.save(Path(output_dir) / "query_embeddings.npy", q_out["embedding"])
    np.save(Path(output_dir) / "database_embeddings.npy", d_out["embedding"])
    np.save(Path(output_dir) / "ranked_indices.npy", ranked)
    write_dataframe(records_to_dataframe(query_records), Path(output_dir) / "query_manifest.csv")
    write_dataframe(records_to_dataframe(db_records), Path(output_dir) / "database_manifest.csv")
    save_json({"vocab": vocab, "seed": seed, "vocab_size": len(vocab), "raw_unique_training_cuis": len(counter)}, Path(output_dir) / "vocabulary.json")
    return metrics


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.limit_train is not None:
        cfg.setdefault("runtime", {})["limit_train"] = args.limit_train
    if args.limit_valid is not None:
        cfg.setdefault("runtime", {})["limit_valid"] = args.limit_valid
    if args.limit_test is not None:
        cfg.setdefault("runtime", {})["limit_test"] = args.limit_test
    metrics = run_baseline(cfg, args.seed, args.device)
    print(pd.DataFrame([metrics]).to_string(index=False))


if __name__ == "__main__":
    main()
