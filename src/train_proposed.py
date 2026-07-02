from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import cfg_get, load_config, parse_args
from .cui import build_vocabulary, encode_cui_sets, filter_to_vocab, positive_class_weights
from .datasets import RocoImageDataset, load_split_records, records_to_dataframe
from .encoders import build_encoder_from_config
from .graph import build_hypergraph_incidence, hypergraph_propagation_matrix, hypergraph_statistics
from .losses import counterfactual_margin_loss, kg_consistency_loss, multilabel_cui_loss, supervised_contrastive_loss
from .metrics import evaluate_rankings, per_query_metrics
from .models import CEHyperKGRetModel, configure_trainable_layers
from .validation import validate_proposed_v6_config
from .retrieval import collate_batch, compute_model_outputs, rerank_ce_hyperkg
from .transforms import build_image_transform
from .utils import device_from_arg, ensure_dir, save_json, set_seed, write_dataframe


def _split_file(cfg: Dict[str, Any], split: str, kind: str) -> str | None:
    return cfg_get(cfg, f"dataset.{kind}_files.{split}")


def _load_records_and_vocab(cfg: Dict[str, Any], seed: int):
    data_root = cfg_get(cfg, "dataset.data_root")
    train_records = load_split_records(data_root, "train", _split_file(cfg, "train", "caption"), _split_file(cfg, "train", "concept"), limit=cfg_get(cfg, "runtime.limit_train"))
    valid_records = load_split_records(data_root, "valid", _split_file(cfg, "valid", "caption"), _split_file(cfg, "valid", "concept"), limit=cfg_get(cfg, "runtime.limit_valid"))
    test_records = load_split_records(data_root, "test", _split_file(cfg, "test", "caption"), _split_file(cfg, "test", "concept"), limit=cfg_get(cfg, "runtime.limit_test"))
    vocab, vocab_index, counter = build_vocabulary([r.cuis for r in train_records], int(cfg_get(cfg, "dataset.min_cui_freq", 5)), int(cfg_get(cfg, "dataset.retained_vocabulary_size_M", 1916)))
    for records in [train_records, valid_records, test_records]:
        for r in records:
            r.cuis = filter_to_vocab(r.cuis, vocab_index)
    return train_records, valid_records, test_records, vocab, vocab_index, counter


def _build_model(cfg: Dict[str, Any], num_cuis: int, device: str) -> CEHyperKGRetModel:
    encoder = build_encoder_from_config(cfg)
    projection_dim = int(cfg_get(cfg, "model.projection_dim", 512))
    dropout = float(cfg_get(cfg, "model.dropout", 0.20))
    model = CEHyperKGRetModel(encoder, num_cuis, projection_dim, dropout)
    mode = cfg_get(cfg, "model.encoder_mode", "all")
    if mode in {"fine_tune", "finetune", "trainable"}:
        mode = "all"
    configure_trainable_layers(model, mode, int(cfg_get(cfg, "model.unfreeze_last_n_children", 3)))
    return model.to(device)


def _make_loader(records, labels, transform, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(RocoImageDataset(records, labels=labels, transform=transform), batch_size=batch_size, shuffle=shuffle, num_workers=workers, collate_fn=collate_batch, pin_memory=torch.cuda.is_available())


def train_one_seed(cfg: Dict[str, Any], seed: int, device: str | None = None) -> Dict[str, float]:
    errors = validate_proposed_v6_config(cfg)
    if errors:
        raise ValueError("Proposed V6 config validation failed: " + "; ".join(errors))
    set_seed(seed)
    device = device_from_arg(device)
    output_dir = ensure_dir(Path(cfg_get(cfg, "paths.output_dir", "../results/ce_hyperkg_ret_v6")) / f"seed_{seed}")
    train_records, valid_records, test_records, vocab, vocab_index, counter = _load_records_and_vocab(cfg, seed)
    train_y = encode_cui_sets([r.cuis for r in train_records], vocab_index)
    valid_y = encode_cui_sets([r.cuis for r in valid_records], vocab_index)
    test_y = encode_cui_sets([r.cuis for r in test_records], vocab_index)
    image_size = int(cfg_get(cfg, "model.input_resolution", [512, 512])[0] if isinstance(cfg_get(cfg, "model.input_resolution", 512), list) else cfg_get(cfg, "model.input_resolution", 512))
    train_transform = build_image_transform(image_size, "imagenet", train=True)
    eval_transform = build_image_transform(image_size, "imagenet", train=False)
    batch_size = int(cfg_get(cfg, "training.batch_size", 32))
    workers = int(cfg_get(cfg, "training.num_workers", 0))
    train_loader = _make_loader(train_records, train_y, train_transform, batch_size, True, workers)
    valid_ds = RocoImageDataset(valid_records, labels=valid_y, transform=eval_transform)
    test_ds = RocoImageDataset(test_records, labels=test_y, transform=eval_transform)
    model = _build_model(cfg, len(vocab), device)
    pos_weight = torch.tensor(positive_class_weights(train_y, float(cfg_get(cfg, "losses.pos_weight_max", 20.0))), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg_get(cfg, "training.learning_rate", 1e-4)), weight_decay=float(cfg_get(cfg, "training.weight_decay", 1e-4)))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg_get(cfg, "training.mixed_precision", True)) and device == "cuda")
    H = build_hypergraph_incidence([r.cuis for r in train_records], vocab_index)
    P = hypergraph_propagation_matrix(H)
    P_dense = torch.tensor(P.toarray(), dtype=torch.float32, device=device)
    best_metric = -1.0
    best_path = Path(output_dir) / "best_model.pt"
    patience = int(cfg_get(cfg, "training.early_stopping_patience", 5))
    bad_epochs = 0
    epochs = int(cfg_get(cfg, "training.maximum_epochs", 25))
    lambda_cui = float(cfg_get(cfg, "losses.lambda_cui", cfg_get(cfg, "loss.lambda_cui_bce", 1.0)))
    lambda_kg = float(cfg_get(cfg, "losses.lambda_kg", cfg_get(cfg, "loss.lambda_kg", 0.10)))
    lambda_cf = float(cfg_get(cfg, "losses.lambda_counterfactual", cfg_get(cfg, "loss.lambda_counterfactual", 0.10)))
    margin = float(cfg_get(cfg, "losses.margin", cfg_get(cfg, "loss.counterfactual_margin", 0.20)))
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            image = batch["image"].to(device)
            label = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                out = model(image)
                expanded = torch.matmul(out["probs"], P_dense.T)
                loss = supervised_contrastive_loss(out["embedding"], label) + lambda_cui * multilabel_cui_loss(out["logits"], label, pos_weight) + lambda_kg * kg_consistency_loss(out["probs"], expanded) + lambda_cf * counterfactual_margin_loss(out["embedding"], label, margin)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg_get(cfg, "training.grad_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        valid_metric = evaluate_model_split(model, valid_ds, valid_records, valid_records, P, cfg, device)["ndcg_at_5"]
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else 0.0, "valid_ndcg_at_5": valid_metric})
        if valid_metric > best_metric:
            best_metric = valid_metric
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "vocab": vocab, "seed": seed, "cfg": cfg}, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate_model_split(model, test_ds, test_records, test_records, P, cfg, device)
    test_metrics.update({"seed": seed, "model": cfg_get(cfg, "model.display_name", "CE-HyperKG-Ret V6")})
    write_dataframe(pd.DataFrame(history), Path(output_dir) / "training_history.csv")
    write_dataframe(pd.DataFrame([test_metrics]), Path(output_dir) / "metrics.csv")
    write_dataframe(records_to_dataframe(train_records), Path(output_dir) / "train_manifest.csv")
    write_dataframe(records_to_dataframe(valid_records), Path(output_dir) / "valid_manifest.csv")
    write_dataframe(records_to_dataframe(test_records), Path(output_dir) / "test_manifest.csv")
    save_json({"vocab": vocab, "vocab_size": len(vocab), "raw_unique_training_cuis": len(counter), "hypergraph": hypergraph_statistics([r.cuis for r in train_records], vocab_index)}, Path(output_dir) / "vocabulary_and_graph.json")
    return test_metrics


def evaluate_model_split(model, dataset, query_records, db_records, propagation, cfg, device: str) -> Dict[str, float]:
    batch_size = int(cfg_get(cfg, "training.batch_size", 32)) * 2
    workers = int(cfg_get(cfg, "training.num_workers", 0))
    out = compute_model_outputs(model, dataset, device, batch_size, workers)
    threshold = float(cfg_get(cfg, "retrieval.cui_threshold", cfg_get(cfg, "retrieval.predicted_cui_threshold_tau", 0.30)))
    reranked = rerank_ce_hyperkg(out["embedding"], out["embedding"], out["probs"], out["probs"], propagation, int(cfg_get(cfg, "retrieval.top_n_visual_candidates", 200)), threshold, int(cfg_get(cfg, "retrieval.top_l_predicted_cuis", 10)), self_exclude=bool(cfg_get(cfg, "evaluation.same_split_self_exclude", True)))
    metrics = evaluate_rankings([r.cuis for r in query_records], reranked["ranked_indices"], [r.cuis for r in db_records], cfg_get(cfg, "evaluation.k_values", [5]))
    output_dir = cfg_get(cfg, "runtime.current_eval_output_dir")
    if output_dir:
        write_dataframe(per_query_metrics([r.image_id for r in query_records], [r.cuis for r in query_records], reranked["ranked_indices"], [r.image_id for r in db_records], [r.cuis for r in db_records], 5), Path(output_dir) / "per_query_metrics.csv")
    return metrics


def run_proposed(cfg: Dict[str, Any], seed: int | None = None, device: str | None = None) -> pd.DataFrame:
    seeds = [int(seed)] if seed is not None else [int(s) for s in cfg_get(cfg, "training.random_seeds", [42, 123, 2025])]
    rows = [train_one_seed(cfg, s, device) for s in seeds]
    output_dir = ensure_dir(cfg_get(cfg, "paths.output_dir", "../results/ce_hyperkg_ret_v6"))
    df = pd.DataFrame(rows)
    write_dataframe(df, Path(output_dir) / "all_seed_metrics.csv")
    metric_cols = [c for c in df.columns if c not in {"seed", "model"}]
    summary = pd.DataFrame([{"metric": c, "mean": float(df[c].mean()), "std": float(df[c].std(ddof=1))} for c in metric_cols])
    write_dataframe(summary, Path(output_dir) / "all_seed_summary.csv")
    return df


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.limit_train is not None:
        cfg.setdefault("runtime", {})["limit_train"] = args.limit_train
    if args.limit_valid is not None:
        cfg.setdefault("runtime", {})["limit_valid"] = args.limit_valid
    if args.limit_test is not None:
        cfg.setdefault("runtime", {})["limit_test"] = args.limit_test
    df = run_proposed(cfg, args.seed, args.device)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
