"""
Copy figures from an experiment directory into the LaTeX report figure layout.
Example:
python scripts/copy_figures_for_report.py --experiment-dir experiments/alpha_v5_h10_ensemble_20260517_193700 --report-dir report
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def copy_if_exists(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(src, dst)
        print(f"copied: {src} -> {dst}")
    else:
        print(f"missing: {src}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--report-dir", default="report")
    args = parser.parse_args()

    exp = Path(args.experiment_dir)
    report = Path(args.report_dir)
    fig_root = report / "figures"

    for seed in [0, 1, 2]:
        seed_dir = exp / f"seed_{seed}" / "figures"
        copy_if_exists(seed_dir / "best_by_model_equity_curve.png", fig_root / f"seed_{seed}" / "best_by_model_equity_curve.png")
        copy_if_exists(seed_dir / "best_equity_curve.png", fig_root / f"seed_{seed}" / "best_equity_curve.png")

    # Optional: copy a representative global figure from seed_0 if available.
    copy_if_exists(exp / "seed_0" / "figures" / "best_equity_curve.png", fig_root / "best_equity_curve.png")


if __name__ == "__main__":
    main()
