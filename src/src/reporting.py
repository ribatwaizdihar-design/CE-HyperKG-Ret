from pathlib import Path
from typing import Dict, Iterable, Sequence

import pandas as pd


def mean_std_table(df: pd.DataFrame, group_col: str = "model", metric_cols: Sequence[str] | None = None) -> pd.DataFrame:
    if metric_cols is None:
        metric_cols = [c for c in df.columns if c not in {group_col, "seed", "config"}]
    rows = []
    for name, group in df.groupby(group_col):
        row: Dict[str, str] = {group_col: name}
        for metric in metric_cols:
            row[metric] = f"{group[metric].mean():.4f} ± {group[metric].std(ddof=1):.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


def collect_metric_files(paths: Iterable[str | Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        p = Path(path)
        if p.is_dir():
            p = p / "metrics.csv"
        frames.append(pd.read_csv(p))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def save_results_table(metrics: pd.DataFrame, output: str | Path, group_col: str = "model") -> pd.DataFrame:
    table = mean_std_table(metrics, group_col)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    return table
