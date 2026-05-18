"""Collect additional ablation results into LaTeX tables and copy figures.

Run from the project root, e.g.
python scripts/collect_ablation_results.py --experiments-dir experiments --report-dir report

The script scans experiment folders whose names start with the expected ablation prefixes,
reads summary/model_comparison_mean_std_pretty.csv, and writes LaTeX tables to
report/tables/. It also copies seed figures to report/figures/ablations/.
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

LOSS_EXPS = {
    "smoothl1": "SmoothL1",
    "smoothl1_ic": "SmoothL1+IC",
    "smoothl1_pairwise": "SmoothL1+Pairwise",
    "smoothl1_ic_pairwise": "SmoothL1+IC+Pairwise",
}
PORT_EXPS = {
    "portfolio_equal_no_smooth": "Equal, no smoothing",
    "portfolio_equal_ema07": "Equal, EMA(0.7)",
    "portfolio_invvol_no_smooth": "Inv-vol, no smoothing",
    "portfolio_invvol_ema07": "Inv-vol, EMA(0.7)",
}
COLS = [
    "test_ic",
    "test_long_only_total_return",
    "test_excess_long_only_return",
    "test_long_short_total_return",
    "test_long_only_sharpe",
    "test_avg_long_turnover",
]


def latest_matching(base: Path, prefix: str) -> Path | None:
    matches = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith(prefix)], key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def read_pretty(path: Path) -> dict[str, str] | None:
    csv = path / "summary" / "model_comparison_mean_std_pretty.csv"
    if not csv.exists():
        return None
    # This file is comma-separated in v5. If values contain commas in your locale, use pandas.
    import pandas as pd
    df = pd.read_csv(csv)
    row = df.iloc[0].to_dict()
    return {c: str(row.get(c, "")) for c in COLS}


def tex_escape(s: str) -> str:
    return s.replace("%", r"\%").replace("±", r"\pm")


def make_table(rows: list[tuple[str, dict[str, str]]], caption: str, label: str) -> str:
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(r"设置 & Test IC & Long-only 收益 & 超额收益 & Long-short 收益 & Sharpe & Turnover \\")
    lines.append(r"\midrule")
    for name, d in rows:
        vals = [tex_escape(d.get(c, "")) for c in COLS]
        lines.append(f"{name} & " + " & ".join([f"${v}$" if v else r"\TODO{结果}" for v in vals]) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def copy_figures(exp: Path, report_dir: Path, key: str) -> None:
    out_base = report_dir / "figures" / "ablations" / key
    for seed_dir in exp.glob("seed_*"):
        m = re.search(r"seed_(\d+)", seed_dir.name)
        if not m:
            continue
        dst = out_base / f"seed{m.group(1)}"
        dst.mkdir(parents=True, exist_ok=True)
        for fname in ["best_by_model_equity_curve.png", "best_equity_curve.png"]:
            src = seed_dir / "figures" / fname
            if src.exists():
                shutil.copy2(src, dst / fname)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-dir", default="experiments")
    parser.add_argument("--report-dir", default="report")
    args = parser.parse_args()
    exp_dir = Path(args.experiments_dir)
    report_dir = Path(args.report_dir)
    tables_dir = report_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    loss_rows = []
    for key, name in LOSS_EXPS.items():
        exp = latest_matching(exp_dir, f"ablation_loss_{key}_h10")
        if exp is None:
            print(f"[missing] {key}")
            continue
        row = read_pretty(exp)
        if row:
            loss_rows.append((name, row))
        copy_figures(exp, report_dir, f"ablation_loss_{key}_h10")

    port_rows = []
    for key, name in PORT_EXPS.items():
        exp = latest_matching(exp_dir, f"ablation_{key}_h10")
        if exp is None:
            print(f"[missing] {key}")
            continue
        row = read_pretty(exp)
        if row:
            port_rows.append((name, row))
        copy_figures(exp, report_dir, f"ablation_{key}_h10")

    if loss_rows:
        (tables_dir / "loss_ablation_table.tex").write_text(make_table(loss_rows, "损失函数消融实验自动汇总。", "tab:loss-ablation-auto"), encoding="utf-8")
        print("wrote", tables_dir / "loss_ablation_table.tex")
    if port_rows:
        (tables_dir / "portfolio_ablation_table.tex").write_text(make_table(port_rows, "组合构建消融实验自动汇总。", "tab:portfolio-ablation-auto"), encoding="utf-8")
        print("wrote", tables_dir / "portfolio_ablation_table.tex")


if __name__ == "__main__":
    main()
