"""
Optional supplementary experiments for the v5 stock alpha project.
Run from the project root directory, for example:

python src/run_additional_experiments.py \
  --csv-path data/raw/all_stocks_5yr.csv \
  --device cuda --amp --seeds 0 1 2

This script launches two groups of experiments:
1. Loss-function ablation for PatchTST.
2. Portfolio-construction ablation for PatchTST.

It assumes that src/run_multi_seed.py from the v5 project is available.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def run(cmd: List[str], dry_run: bool = False) -> None:
    print("\n" + "=" * 100)
    print(" ".join(cmd))
    print("=" * 100)
    if not dry_run:
        subprocess.run(cmd, check=True)


def base_cmd(args: argparse.Namespace, experiment_name: str) -> List[str]:
    cmd = [
        sys.executable,
        "src/run_multi_seed.py",
        "--source", "local",
        "--csv-path", args.csv_path,
        "--experiment-name", experiment_name,
        "--seeds", *[str(s) for s in args.seeds],
        "--preset", args.preset,
        "--models", *args.models,
        "--trials", str(args.trials),
        "--epochs", str(args.epochs),
        "--transaction-cost-bps", str(args.transaction_cost_bps),
        "--horizons", str(args.horizon),
        "--objective", args.objective,
        "--device", args.device,
    ]
    if args.amp:
        cmd.append("--amp")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", default="data/raw/all_stocks_5yr.csv")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--preset", default="quick")
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--objective", default="val_alpha_combo")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--models", nargs="+", default=["patchtst"])
    args = parser.parse_args()

    if not Path(args.csv_path).exists():
        raise FileNotFoundError(f"CSV not found: {args.csv_path}")

    # 1. Loss-function ablation: isolate the training objective.
    losses = [
        "smoothl1",
        "smoothl1_ic",
        "smoothl1_pairwise",
        "smoothl1_ic_pairwise",
    ]
    for loss in losses:
        cmd = base_cmd(args, f"ablation_loss_{loss}_h{args.horizon}")
        cmd += [
            "--loss", loss,
            "--score-emas", "1.0",
            "--portfolio-weightings", "equal",
        ]
        run(cmd, args.dry_run)

    # 2. Portfolio-construction ablation: isolate smoothing and weighting.
    portfolio_settings = [
        ("equal_no_smooth", "1.0", "equal"),
        ("equal_ema07", "0.7", "equal"),
        ("invvol_no_smooth", "1.0", "inv_vol"),
        ("invvol_ema07", "0.7", "inv_vol"),
    ]
    for name, ema, weighting in portfolio_settings:
        cmd = base_cmd(args, f"ablation_portfolio_{name}_h{args.horizon}")
        cmd += [
            "--loss", "smoothl1_ic",
            "--score-emas", ema,
            "--portfolio-weightings", weighting,
        ]
        run(cmd, args.dry_run)


if __name__ == "__main__":
    main()
