from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


REPORT_METRICS = [
    "test_ic",
    "test_long_only_total_return",
    "test_buy_hold_total_return",
    "test_excess_long_only_return",
    "test_long_only_sharpe",
    "test_buy_hold_sharpe",
    "test_long_short_total_return",
    "test_long_short_sharpe",
    "test_avg_long_turnover",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Run train_alpha.py over multiple random seeds and aggregate the report-ready results."
    )
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2], help="Seeds to run in sequence")
    p.add_argument("--experiment-name", type=str, default="alpha_v5_multiseed")
    p.add_argument("--output-root", type=str, default="experiments")
    p.add_argument("--no-timestamp", action="store_true", help="Do not append timestamp to experiment directory")

    # Core train_alpha.py arguments.
    p.add_argument("--source", choices=["kaggle_sp500", "local"], default="local")
    p.add_argument("--csv-path", type=str, default="data/raw/all_stocks_5yr.csv")
    p.add_argument("--preset", choices=["smoke", "quick", "full"], default="quick")
    p.add_argument("--models", nargs="*", default=["resmlp", "gru", "lstm", "tcn", "dlinear", "patchtst"])
    p.add_argument("--trials", type=int, default=12)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--transaction-cost-bps", type=float, default=5.0)
    p.add_argument("--cost-sweep-bps", nargs="*", type=float, default=[0.0, 5.0, 10.0])
    p.add_argument("--target-mode", choices=["raw", "zscore", "rank", "rank_zscore"], default="rank")
    p.add_argument("--loss", choices=["smoothl1", "huber", "mse", "smoothl1_ic", "smoothl1_pairwise", "smoothl1_ic_pairwise", "ic", "pairwise"], default="smoothl1_ic")
    p.add_argument(
        "--objective",
        choices=[
            "val_net_long_only_excess_return",
            "val_net_long_only_sharpe",
            "val_net_long_short_sharpe",
            "val_ic",
            "val_icir",
            "val_combo",
            "val_alpha_combo",
        ],
        default="val_alpha_combo",
    )
    p.add_argument("--lookbacks", nargs="*", type=int, default=None)
    p.add_argument("--horizons", nargs="*", type=int, default=None)
    p.add_argument("--top-fracs", nargs="*", type=float, default=None)
    p.add_argument("--score-emas", nargs="*", type=float, default=None)
    p.add_argument("--portfolio-weightings", nargs="*", default=None, choices=["equal", "inv_vol"])
    p.add_argument("--ic-loss-weight", type=float, default=0.10)
    p.add_argument("--pairwise-loss-weight", type=float, default=0.05)
    p.add_argument("--turnover-penalty", type=float, default=0.10)
    p.add_argument("--enable-ensemble", action="store_true")
    p.add_argument("--ensemble-models", nargs="*", default=["lstm", "resmlp", "patchtst"])
    p.add_argument("--ensemble-top-frac", type=float, default=0.05)
    p.add_argument("--ensemble-score-ema", type=float, default=0.7)
    p.add_argument("--ensemble-weighting", choices=["equal", "inv_vol"], default="equal")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--torch-threads", type=int, default=4)
    p.add_argument("--max-tickers", type=int, default=None)
    p.add_argument("--min-rows", type=int, default=900)
    p.add_argument("--train-end", type=str, default="2016-12-31")
    p.add_argument("--val-end", type=str, default="2017-06-30")
    p.add_argument("--save-trial-predictions", action="store_true")
    p.add_argument("--save-histories", action="store_true")
    p.add_argument("--continue-on-error", action="store_true", help="Continue other seeds if one seed fails")
    return p.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def tee_subprocess(cmd: list[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("COMMAND:\n" + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        proc.wait()
        return int(proc.returncode)


def append_list_arg(cmd: list[str], flag: str, values):
    if values is not None and len(values) > 0:
        cmd.append(flag)
        cmd.extend(str(v) for v in values)


def build_train_cmd(args, seed: int, seed_dir: Path) -> list[str]:
    train_script = project_root() / "src" / "train_alpha.py"
    cmd = [sys.executable, str(train_script)]
    cmd += ["--source", args.source]
    if args.source == "local":
        cmd += ["--csv-path", args.csv_path]
    cmd += ["--preset", args.preset]
    cmd += ["--models", *args.models]
    cmd += ["--trials", str(args.trials)]
    cmd += ["--epochs", str(args.epochs)]
    cmd += ["--seed", str(seed)]
    cmd += ["--transaction-cost-bps", str(args.transaction_cost_bps)]
    append_list_arg(cmd, "--cost-sweep-bps", args.cost_sweep_bps)
    cmd += ["--target-mode", args.target_mode]
    cmd += ["--loss", args.loss]
    cmd += ["--objective", args.objective]
    cmd += ["--ic-loss-weight", str(args.ic_loss_weight)]
    cmd += ["--pairwise-loss-weight", str(args.pairwise_loss_weight)]
    cmd += ["--turnover-penalty", str(args.turnover_penalty)]
    cmd += ["--device", args.device]
    cmd += ["--batch-size", str(args.batch_size)]
    cmd += ["--num-workers", str(args.num_workers)]
    cmd += ["--torch-threads", str(args.torch_threads)]
    cmd += ["--train-end", args.train_end, "--val-end", args.val_end]
    cmd += ["--min-rows", str(args.min_rows)]
    if args.max_tickers is not None:
        cmd += ["--max-tickers", str(args.max_tickers)]
    append_list_arg(cmd, "--lookbacks", args.lookbacks)
    append_list_arg(cmd, "--horizons", args.horizons)
    append_list_arg(cmd, "--top-fracs", args.top_fracs)
    append_list_arg(cmd, "--score-emas", args.score_emas)
    append_list_arg(cmd, "--portfolio-weightings", args.portfolio_weightings)
    if args.enable_ensemble:
        cmd.append("--enable-ensemble")
        append_list_arg(cmd, "--ensemble-models", args.ensemble_models)
        cmd += ["--ensemble-top-frac", str(args.ensemble_top_frac)]
        cmd += ["--ensemble-score-ema", str(args.ensemble_score_ema)]
        cmd += ["--ensemble-weighting", args.ensemble_weighting]
    if args.amp:
        cmd.append("--amp")
    if args.save_trial_predictions:
        cmd.append("--save-trial-predictions")
    if args.save_histories:
        cmd.append("--save-histories")
    cmd += ["--output-dir", str(seed_dir)]
    return cmd


def metric_mean_std_table(all_cmp: pd.DataFrame) -> pd.DataFrame:
    metrics = [m for m in REPORT_METRICS if m in all_cmp.columns]
    grouped = all_cmp.groupby("model", dropna=False)[metrics]
    mean = grouped.mean().add_suffix("_mean")
    std = grouped.std(ddof=1).fillna(0.0).add_suffix("_std")
    n = grouped.count().iloc[:, :1].rename(columns={metrics[0]: "n_seeds"}) if metrics else pd.DataFrame()
    return pd.concat([n, mean, std], axis=1).reset_index()


def metric_pretty_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in summary.iterrows():
        out = {"model": r["model"], "n_seeds": int(r.get("n_seeds", 0))}
        for m in REPORT_METRICS:
            mean_col = f"{m}_mean"
            std_col = f"{m}_std"
            if mean_col in summary.columns:
                out[m] = f"{r[mean_col]:.4f} ± {r[std_col]:.4f}"
        rows.append(out)
    return pd.DataFrame(rows)


def aggregate(exp_dir: Path, seeds: list[int]) -> None:
    summary_dir = exp_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    cmp_frames = []
    cost_frames = []
    trial_frames = []
    missing = []
    for seed in seeds:
        seed_dir = exp_dir / f"seed_{seed}"
        cmp_path = seed_dir / "model_comparison.csv"
        cost_path = seed_dir / "cost_sensitivity.csv"
        trials_path = seed_dir / "trials.csv"
        if not cmp_path.exists():
            missing.append(seed)
            continue
        cmp = pd.read_csv(cmp_path)
        cmp.insert(0, "seed", seed)
        cmp_frames.append(cmp)
        if cost_path.exists():
            cost = pd.read_csv(cost_path)
            cost.insert(0, "seed", seed)
            cost_frames.append(cost)
        if trials_path.exists():
            tr = pd.read_csv(trials_path)
            tr.insert(0, "seed", seed)
            trial_frames.append(tr)

    if missing:
        print(f"[aggregate] warning: missing model_comparison.csv for seeds: {missing}")
    if not cmp_frames:
        print("[aggregate] no finished seeds to aggregate.")
        return

    all_cmp = pd.concat(cmp_frames, ignore_index=True)
    all_cmp.to_csv(summary_dir / "model_comparison_all_seeds.csv", index=False)

    numeric_summary = metric_mean_std_table(all_cmp)
    numeric_summary.to_csv(summary_dir / "model_comparison_mean_std.csv", index=False)

    pretty = metric_pretty_table(numeric_summary)
    pretty.to_csv(summary_dir / "model_comparison_mean_std_pretty.csv", index=False)
    with (summary_dir / "model_comparison_mean_std_latex.tex").open("w", encoding="utf-8") as f:
        f.write(pretty.to_latex(index=False))

    # Best model by test excess return for each seed, for a quick robustness check.
    best_by_seed = (
        all_cmp.sort_values(["seed", "test_excess_long_only_return"], ascending=[True, False])
        .groupby("seed", as_index=False)
        .head(1)
    )
    best_by_seed.to_csv(summary_dir / "best_model_by_seed_test_excess.csv", index=False)

    if cost_frames:
        all_cost = pd.concat(cost_frames, ignore_index=True)
        all_cost.to_csv(summary_dir / "cost_sensitivity_all_seeds.csv", index=False)
        cost_metrics = [
            "test_long_only_total_return",
            "test_excess_long_only_return",
            "test_long_only_sharpe",
            "test_avg_long_turnover",
        ]
        existing = [c for c in cost_metrics if c in all_cost.columns]
        cost_summary = pd.concat(
            [
                all_cost.groupby(["model", "cost_bps"])[existing].mean().add_suffix("_mean"),
                all_cost.groupby(["model", "cost_bps"])[existing].std(ddof=1).fillna(0.0).add_suffix("_std"),
            ],
            axis=1,
        ).reset_index()
        cost_summary.to_csv(summary_dir / "cost_sensitivity_mean_std.csv", index=False)

    if trial_frames:
        all_trials = pd.concat(trial_frames, ignore_index=True)
        all_trials.to_csv(summary_dir / "trials_all_seeds.csv", index=False)

    print("\n" + "=" * 88)
    print("Multi-seed summary written to:")
    print(f"  {summary_dir / 'model_comparison_all_seeds.csv'}")
    print(f"  {summary_dir / 'model_comparison_mean_std.csv'}")
    print(f"  {summary_dir / 'model_comparison_mean_std_pretty.csv'}")
    print(f"  {summary_dir / 'best_model_by_seed_test_excess.csv'}")
    if cost_frames:
        print(f"  {summary_dir / 'cost_sensitivity_mean_std.csv'}")
    print("\nMean ± std table:")
    print(pretty.to_string(index=False))


def main():
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.experiment_name if args.no_timestamp else f"{args.experiment_name}_{stamp}"
    exp_dir = Path(args.output_root) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    with (exp_dir / "multi_seed_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print("=" * 88)
    print("Multi-seed experiment")
    print(f"experiment_dir={exp_dir}")
    print(f"seeds={args.seeds}")
    print("=" * 88)

    failed = []
    for seed in args.seeds:
        seed_dir = exp_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_train_cmd(args, seed, seed_dir)
        (seed_dir / "command.txt").write_text(" ".join(cmd), encoding="utf-8")
        print("\n" + "#" * 88)
        print(f"Running seed={seed}")
        print("#" * 88)
        ret = tee_subprocess(cmd, cwd=project_root(), log_path=seed_dir / "run.log")
        if ret != 0:
            failed.append(seed)
            print(f"[seed={seed}] failed with return code {ret}")
            if not args.continue_on_error:
                raise SystemExit(ret)

    aggregate(exp_dir, args.seeds)
    if failed:
        raise SystemExit(f"Some seeds failed: {failed}")


if __name__ == "__main__":
    main()
