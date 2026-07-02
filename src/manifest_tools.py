from pathlib import Path
from typing import Dict

import pandas as pd

from .cui import build_vocabulary
from .datasets import load_split_records, records_to_dataframe
from .graph import hypergraph_statistics
from .utils import save_json, write_dataframe


def export_manifests(data_root: str, output_dir: str, min_cui_freq: int = 5, max_cuis: int = 1916) -> Dict[str, float]:
    output_dir = Path(output_dir)
    records = {split: load_split_records(data_root, split) for split in ["train", "valid", "test"]}
    vocab, vocab_index, counter = build_vocabulary([r.cuis for r in records["train"]], min_cui_freq, max_cuis)
    for split, recs in records.items():
        write_dataframe(records_to_dataframe(recs), output_dir / f"{split}_manifest.csv")
    stats = hypergraph_statistics([r.cuis for r in records["train"]], vocab_index)
    stats.update({"raw_unique_training_cuis": float(len(counter)), "retained_vocabulary_size": float(len(vocab)), "removed_low_frequency_cuis": float(len(counter) - len(vocab))})
    save_json({"vocab": vocab, "stats": stats}, output_dir / "train_derived_vocabulary_and_graph.json")
    return stats


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", default="../results/manifests")
    parser.add_argument("--min-cui-freq", type=int, default=5)
    parser.add_argument("--max-cuis", type=int, default=1916)
    args = parser.parse_args()
    stats = export_manifests(args.data_root, args.output_dir, args.min_cui_freq, args.max_cuis)
    print(pd.DataFrame([stats]).to_string(index=False))


if __name__ == "__main__":
    main()
