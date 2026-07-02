from typing import Any, Dict, List

from .config import cfg_get
from .constants import FINAL_V6


def _first(cfg: Dict[str, Any], *keys: str):
    for key in keys:
        value = cfg_get(cfg, key)
        if value is not None:
            return value
    return None


def _metric_name(value: Any) -> str:
    return str(value).lower().replace("@", "_at_").replace("-", "_").replace(" ", "_")


def validate_proposed_v6_config(cfg: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    checks = [
        (("model.backbone",), FINAL_V6["backbone"]),
        (("dataset.min_cui_freq",), FINAL_V6["min_cui_freq"]),
        (("dataset.retained_vocabulary_size_M",), FINAL_V6["retained_vocabulary_size"]),
        (("retrieval.top_n_visual_candidates",), FINAL_V6["top_n_visual"]),
        (("retrieval.top_l_predicted_cuis",), FINAL_V6["top_l_predicted_cuis"]),
        (("retrieval.cui_threshold", "retrieval.predicted_cui_threshold_tau"), FINAL_V6["cui_threshold"]),
        (("training.batch_size",), FINAL_V6["batch_size"]),
        (("training.maximum_epochs",), FINAL_V6["epochs"]),
        (("training.early_stopping_patience",), FINAL_V6["early_stopping_patience"]),
        (("training.learning_rate",), FINAL_V6["learning_rate"]),
        (("training.weight_decay",), FINAL_V6["weight_decay"]),
        (("losses.lambda_cui", "loss.lambda_cui_bce"), FINAL_V6["lambda_cui"]),
        (("losses.lambda_kg", "loss.lambda_kg"), FINAL_V6["lambda_kg"]),
        (("losses.lambda_counterfactual", "loss.lambda_counterfactual"), FINAL_V6["lambda_counterfactual"]),
        (("losses.margin", "loss.counterfactual_margin"), FINAL_V6["margin"]),
        (("statistics.bootstrap_resamples", "evaluation.bootstrap_resamples"), FINAL_V6["bootstrap_resamples"]),
    ]
    for keys, expected in checks:
        value = _first(cfg, *keys)
        if value is None:
            continue
        if isinstance(expected, float):
            ok = abs(float(value) - expected) < 1e-12
        elif keys == ("model.backbone",):
            ok = str(value).lower() == str(expected).lower()
        else:
            ok = value == expected
        if not ok:
            errors.append(f"{'/'.join(keys)}={value} expected {expected}")
    metric = cfg_get(cfg, "training.validation_metric")
    if metric is not None and "generic_filtered" not in _metric_name(metric) and "ndcg_at_5" not in _metric_name(metric):
        errors.append(f"training.validation_metric={metric} expected Generic-filtered NDCG@5")
    seeds = tuple(cfg_get(cfg, "training.random_seeds", ()))
    if seeds and seeds != FINAL_V6["seeds"]:
        errors.append(f"training.random_seeds={seeds} expected {FINAL_V6['seeds']}")
    return errors


def validate_baseline_config(cfg: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required_false = [
        "retrieval.use_cui_prediction_for_ranking",
        "retrieval.use_pairwise_graph",
        "retrieval.use_hypergraph",
        "retrieval.use_counterfactual_negatives",
        "retrieval.use_uncertainty_fusion",
    ]
    for key in required_false:
        if bool(cfg_get(cfg, key, False)):
            errors.append(f"{key} must be false for image-only baseline")
    if cfg_get(cfg, "dataset.min_cui_freq", 5) != 5:
        errors.append("dataset.min_cui_freq must be 5")
    if cfg_get(cfg, "dataset.max_vocabulary_cap", 2048) != 2048:
        errors.append("dataset.max_vocabulary_cap must be 2048")
    return errors
