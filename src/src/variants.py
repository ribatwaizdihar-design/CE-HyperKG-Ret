from copy import deepcopy
from typing import Any, Dict

VARIANT_COMPONENTS = {
    "V1": ("visual",),
    "V2": ("visual", "cui_prediction_head"),
    "V3": ("visual", "cui_prediction_head", "pairwise_graph_reranking"),
    "V4": ("visual", "cui_prediction_head", "hypergraph_reranking"),
    "V5": ("visual", "cui_prediction_head", "hypergraph_reranking", "counterfactual_hard_negatives"),
    "V6": ("visual", "cui_prediction_head", "hypergraph_reranking", "counterfactual_hard_negatives", "uncertainty_aware_fusion"),
}


def apply_variant(cfg: Dict[str, Any], variant: str) -> Dict[str, Any]:
    variant = variant.upper()
    if variant not in VARIANT_COMPONENTS:
        raise ValueError(f"Unknown variant {variant}")
    out = deepcopy(cfg)
    components = set(VARIANT_COMPONENTS[variant])
    retrieval = out.setdefault("retrieval", {})
    retrieval["use_visual_similarity"] = "visual" in components
    retrieval["use_cui_prediction_for_ranking"] = "cui_prediction_head" in components
    retrieval["use_pairwise_graph"] = "pairwise_graph_reranking" in components
    retrieval["use_hypergraph"] = "hypergraph_reranking" in components
    retrieval["use_counterfactual_negatives"] = "counterfactual_hard_negatives" in components
    retrieval["use_uncertainty_fusion"] = "uncertainty_aware_fusion" in components
    out.setdefault("experiment", {})["variant"] = variant
    return out


def variant_name_from_config(cfg: Dict[str, Any]) -> str:
    return str(cfg.get("experiment", {}).get("variant", "V6")).upper()
