from pathlib import Path

import pandas as pd

from .config import load_config
from .train_baseline import run_baseline
from .utils import write_dataframe


def run_all(config_dir: str | Path, output: str | Path, device: str | None = None) -> pd.DataFrame:
    config_paths = sorted(Path(config_dir).glob("*.yaml"))
    rows = []
    for path in config_paths:
        cfg = load_config(path)
        seeds = cfg.get("training", {}).get("random_seeds", [42, 123, 2025])
        for seed in seeds:
            row = run_baseline(cfg, int(seed), device)
            row["config"] = str(path)
            rows.append(row)
    df = pd.DataFrame(rows)
    write_dataframe(df, output)
    metric_cols = [c for c in df.columns if c not in {"seed", "model", "config"}]
    summary = df.groupby("model")[metric_cols].agg(["mean", "std"])
    write_dataframe(summary.reset_index(), Path(output).with_name(Path(output).stem + "_summary.csv"))
    return df


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--output", default="../results/all_baselines_metrics.csv")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    df = run_all(args.config_dir, args.output, args.device)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
