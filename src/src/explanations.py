from typing import Dict, Sequence

from .cui import jaccard


def retrieval_explanation(query_cuis: Sequence[str], candidate_cuis: Sequence[str], visual_score: float, predicted_cui_score: float, hypergraph_score: float, final_score: float) -> Dict[str, object]:
    query_set = set(query_cuis)
    candidate_set = set(candidate_cuis)
    shared = sorted(query_set & candidate_set)
    return {
        "shared_cuis": shared,
        "num_shared_cuis": len(shared),
        "query_only_cuis": sorted(query_set - candidate_set),
        "candidate_only_cuis": sorted(candidate_set - query_set),
        "jaccard": jaccard(query_set, candidate_set),
        "visual_score": float(visual_score),
        "predicted_cui_score": float(predicted_cui_score),
        "hypergraph_score": float(hypergraph_score),
        "final_score": float(final_score),
    }
