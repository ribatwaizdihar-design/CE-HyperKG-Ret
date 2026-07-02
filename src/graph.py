from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix, diags


def build_pairwise_graph(cui_sets: Iterable[Sequence[str]], vocab_index: Dict[str, int]) -> Tuple[np.ndarray, List[Tuple[str, str, int]]]:
    counter = Counter()
    for cuis in cui_sets:
        idx = sorted({vocab_index[c] for c in cuis if c in vocab_index})
        for i, a in enumerate(idx):
            for b in idx[i + 1:]:
                counter[(a, b)] += 1
    edges = [(a, b, w) for (a, b), w in counter.items()]
    adj = np.zeros((len(vocab_index), len(vocab_index)), dtype=np.float32)
    inv = {v: k for k, v in vocab_index.items()}
    named = []
    for a, b, w in edges:
        adj[a, b] = float(w)
        adj[b, a] = float(w)
        named.append((inv[a], inv[b], int(w)))
    return adj, named


def build_hypergraph_incidence(cui_sets: Sequence[Sequence[str]], vocab_index: Dict[str, int]) -> csr_matrix:
    rows = []
    cols = []
    data = []
    e = 0
    for cuis in cui_sets:
        idx = sorted({vocab_index[c] for c in cuis if c in vocab_index})
        if not idx:
            continue
        for i in idx:
            rows.append(i)
            cols.append(e)
            data.append(1.0)
        e += 1
    return csr_matrix((data, (rows, cols)), shape=(len(vocab_index), e), dtype=np.float32)


def hypergraph_propagation_matrix(H: csr_matrix) -> csr_matrix:
    if H.shape[1] == 0:
        return csr_matrix((H.shape[0], H.shape[0]), dtype=np.float32)
    edge_degree = np.asarray(H.sum(axis=0)).ravel().astype(np.float32)
    edge_degree = np.maximum(edge_degree, 1.0)
    inv_de = diags(1.0 / edge_degree)
    return (H @ inv_de @ H.T).tocsr().astype(np.float32)


def expand_probabilities(probs: np.ndarray, propagation: csr_matrix) -> np.ndarray:
    expanded = probs @ propagation.T
    norm = np.linalg.norm(expanded, axis=1, keepdims=True) + 1e-12
    return (expanded / norm).astype(np.float32)


def hypergraph_statistics(cui_sets: Sequence[Sequence[str]], vocab_index: Dict[str, int]) -> Dict[str, float]:
    retained = [sorted({c for c in cuis if c in vocab_index}) for cuis in cui_sets]
    non_empty = [x for x in retained if x]
    sizes = [len(x) for x in non_empty]
    _, edges = build_pairwise_graph(non_empty, vocab_index)
    return {
        "hypergraph_nodes": float(len(vocab_index)),
        "hyperedges": float(len(non_empty)),
        "mean_cuis_per_hyperedge": float(np.mean(sizes) if sizes else 0.0),
        "max_cuis_per_hyperedge": float(max(sizes) if sizes else 0.0),
        "pairwise_graph_nodes": float(len(vocab_index)),
        "pairwise_graph_edges": float(len(edges)),
    }
