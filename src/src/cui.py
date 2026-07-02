import ast
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

CUI_RE = re.compile(r"C\d{7,8}")


def parse_cuis_from_value(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, (list, tuple, set)):
        raw = " ".join(map(str, value))
    else:
        raw = str(value)
        stripped = raw.strip()
        if stripped.startswith(("[", "(", "{")):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, dict):
                    raw = " ".join(map(str, parsed.values()))
                elif isinstance(parsed, (list, tuple, set)):
                    raw = " ".join(map(str, parsed))
            except Exception:
                pass
    out = []
    seen = set()
    for cui in CUI_RE.findall(raw.upper()):
        if cui not in seen:
            seen.add(cui)
            out.append(cui)
    return out


def normalize_cui_set(values: Iterable[Any]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        for cui in parse_cuis_from_value(value):
            if cui not in seen:
                seen.add(cui)
                out.append(cui)
    return out


def build_vocabulary(cui_sets: Iterable[Sequence[str]], min_freq: int = 5, max_cuis: int | None = None) -> Tuple[List[str], Dict[str, int], Counter]:
    counter = Counter()
    for cuis in cui_sets:
        counter.update(set(cuis))
    items = [(cui, count) for cui, count in counter.most_common() if count >= min_freq]
    if max_cuis is not None:
        items = items[:max_cuis]
    vocab = [cui for cui, _ in items]
    return vocab, {cui: idx for idx, cui in enumerate(vocab)}, counter


def filter_to_vocab(cuis: Sequence[str], vocab_index: Dict[str, int]) -> List[str]:
    return [c for c in cuis if c in vocab_index]


def encode_cui_sets(cui_sets: Sequence[Sequence[str]], vocab_index: Dict[str, int]) -> np.ndarray:
    y = np.zeros((len(cui_sets), len(vocab_index)), dtype=np.float32)
    for i, cuis in enumerate(cui_sets):
        for cui in cuis:
            j = vocab_index.get(cui)
            if j is not None:
                y[i, j] = 1.0
    return y


def decode_cui_vector(vec: np.ndarray, vocab: Sequence[str], threshold: float = 0.5) -> List[str]:
    return [vocab[i] for i, val in enumerate(vec) if float(val) >= threshold]


def select_predicted_cuis(probs: np.ndarray, vocab: Sequence[str], threshold: float = 0.30, top_m: int = 10) -> List[str]:
    probs = np.asarray(probs, dtype=np.float32)
    if probs.ndim != 1:
        raise ValueError("select_predicted_cuis expects one probability vector")
    idx = np.where(probs >= threshold)[0]
    if idx.size == 0:
        idx = np.array([int(np.argmax(probs))])
    order = idx[np.argsort(probs[idx])[::-1]]
    order = order[:top_m]
    return [vocab[int(i)] for i in order]


def selected_binary_matrix(probs: np.ndarray, threshold: float = 0.30, top_m: int = 10) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float32)
    y = np.zeros_like(probs, dtype=np.float32)
    for i in range(probs.shape[0]):
        idx = np.where(probs[i] >= threshold)[0]
        if idx.size == 0:
            idx = np.array([int(np.argmax(probs[i]))])
        order = idx[np.argsort(probs[i, idx])[::-1]][:top_m]
        y[i, order] = 1.0
    return y


def jaccard(a: Sequence[str] | Set[str], b: Sequence[str] | Set[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def binary_jaccard_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = (np.asarray(a) > 0).astype(np.float32)
    b = (np.asarray(b) > 0).astype(np.float32)
    inter = a @ b.T
    union = a.sum(axis=1, keepdims=True) + b.sum(axis=1, keepdims=True).T - inter
    return inter / np.maximum(union, 1e-12)


def positive_class_weights(y: np.ndarray, max_weight: float = 20.0) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    weight = neg / np.maximum(pos, 1.0)
    return np.clip(weight, 1.0, max_weight).astype(np.float32)
