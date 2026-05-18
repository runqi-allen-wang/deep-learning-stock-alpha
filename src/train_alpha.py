from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from stock_alpha_dl.backtest import (
    backtest_cross_section,
    daily_spearman_ic,
    daily_top_bottom_spread,
    evaluate_direction,
)
from stock_alpha_dl.data import data_source_summary, filter_universe, load_kaggle_sp500, load_local_csv
from stock_alpha_dl.dataset import make_sequences
from stock_alpha_dl.features import add_features
from stock_alpha_dl.training import TrainConfig, predict, set_seed, train_model


def parse_args():
    p = argparse.ArgumentParser(description="Deep learning cross-sectional stock alpha project v5 with IC-aware loss, ensemble, score smoothing, and cost-aware portfolio construction")
    p.add_argument("--source", choices=["kaggle_sp500", "local"], default="kaggle_sp500")
    p.add_argument("--csv-path", type=str, default=None)
    p.add_argument("--tickers", nargs="*", default=None)
    p.add_argument("--start", type=str, default="2013-02-08")
    p.add_argument("--end", type=str, default="2018-02-08")
    p.add_argument("--train-end", type=str, default="2016-12-31")
    p.add_argument("--val-end", type=str, default="2017-06-30")
    p.add_argument("--min-rows", type=int, default=900)
    p.add_argument("--max-tickers", type=int, default=None)
    p.add_argument("--preset", choices=["smoke", "quick", "full"], default="quick")
    p.add_argument(
        "--models",
        nargs="*",
        default=None,
        choices=["mlp", "resmlp", "gru", "lstm", "tcn", "dlinear", "transformer", "patchtst"],
    )
    p.add_argument("--trials", type=int, default=None, help="Number of hyperparameter trials PER MODEL")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lookbacks", nargs="*", type=int, default=None, help="Allowed lookback windows for random search, e.g. --lookbacks 10 20 30")
    p.add_argument("--horizons", nargs="*", type=int, default=None, help="Allowed prediction horizons for random search, e.g. --horizons 10")
    p.add_argument("--top-fracs", nargs="*", type=float, default=None, help="Allowed top-k fractions, e.g. --top-fracs 0.05 0.1")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--transaction-cost-bps", type=float, default=5.0, help="Cost per one-way dollar turnover in basis points")
    p.add_argument("--cost-sweep-bps", nargs="*", type=float, default=[0.0, 5.0, 10.0], help="Extra cost levels evaluated for each selected model")
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
    p.add_argument("--ic-loss-weight", type=float, default=0.10)
    p.add_argument("--pairwise-loss-weight", type=float, default=0.05)
    p.add_argument("--score-emas", nargs="*", type=float, default=None, help="Candidate score EMA alpha values. 1=no smoothing; smaller means smoother scores.")
    p.add_argument("--portfolio-weightings", nargs="*", default=None, choices=["equal", "inv_vol"], help="Candidate portfolio weight schemes.")
    p.add_argument("--turnover-penalty", type=float, default=0.10)
    p.add_argument("--enable-ensemble", action="store_true", help="Create an ensemble from selected best model predictions.")
    p.add_argument("--ensemble-models", nargs="*", default=["lstm", "resmlp", "patchtst"], help="Best model families used in ensemble when available.")
    p.add_argument("--ensemble-top-frac", type=float, default=0.05)
    p.add_argument("--ensemble-score-ema", type=float, default=0.7)
    p.add_argument("--ensemble-weighting", choices=["equal", "inv_vol"], default="equal")
    p.add_argument("--output-dir", type=str, default="results_alpha_v5")
    p.add_argument("--save-trial-predictions", action="store_true", help="Save validation/test predictions for every trial. Off by default to keep output folders clean.")
    p.add_argument("--save-histories", action="store_true", help="Save training-loss history for every trial. Off by default to keep output folders clean.")
    p.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    p.add_argument("--amp", action="store_true", help="Use mixed precision training on CUDA for speed and lower GPU memory.")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers. On Windows, 0 is safest; try 2 or 4 on Linux.")
    p.add_argument("--torch-threads", type=int, default=4)
    return p.parse_args()


def preset_defaults(args):
    if args.preset == "smoke":
        return {"max_tickers": 30, "trials": 3, "epochs": 3, "models": ["resmlp", "lstm", "patchtst"]}
    if args.preset == "quick":
        return {"max_tickers": 100, "trials": 10, "epochs": 8, "models": ["resmlp", "gru", "lstm", "tcn", "dlinear", "patchtst"]}
    return {"max_tickers": 300, "trials": 20, "epochs": 15, "models": ["resmlp", "gru", "lstm", "tcn", "dlinear", "transformer", "patchtst"]}


def sample_config(
    rng: random.Random,
    model_choices,
    epochs: int,
    batch_size: int,
    target_mode: str,
    loss: str,
    lookback_choices=None,
    horizon_choices=None,
    top_frac_choices=None,
    score_ema_choices=None,
    weighting_choices=None,
    ic_loss_weight: float = 0.10,
    pairwise_loss_weight: float = 0.05,
) -> TrainConfig:
    model = rng.choice(model_choices)
    # More mass on 10-day horizon because prior runs showed h=10 was more promising.
    lookback = rng.choice(lookback_choices or [10, 20, 30, 60, 60])
    horizon = rng.choice(horizon_choices or [5, 10, 10, 20])
    hidden_dim = rng.choice([64, 128, 128, 256])
    dropout = rng.choice([0.05, 0.10, 0.20, 0.30])
    lr = rng.choice([1e-4, 3e-4, 1e-3])
    weight_decay = rng.choice([1e-5, 1e-4, 1e-3])
    top_frac = rng.choice(top_frac_choices or [0.05, 0.10, 0.15, 0.20])
    score_ema = rng.choice(score_ema_choices or [1.0, 0.7, 0.5])
    portfolio_weighting = rng.choice(weighting_choices or ["equal", "equal", "inv_vol"])
    return TrainConfig(
        model=model,
        lookback=lookback,
        horizon=horizon,
        hidden_dim=hidden_dim,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        epochs=epochs,
        patience=4,
        top_frac=top_frac,
        target_mode=target_mode,
        loss=loss,
        ic_loss_weight=ic_loss_weight,
        pairwise_loss_weight=pairwise_loss_weight,
        score_ema=score_ema,
        portfolio_weighting=portfolio_weighting,
    )


def objective_value(metrics: dict, ic_mean: float, icir: float, spread: float, args) -> float:
    if args.objective == "val_net_long_only_excess_return":
        return metrics.get("excess_long_only_return", float("nan"))
    if args.objective == "val_net_long_only_sharpe":
        return metrics.get("long_only_sharpe", float("nan")) - metrics.get("buy_hold_sharpe", 0.0)
    if args.objective == "val_net_long_short_sharpe":
        return metrics.get("long_short_sharpe", float("nan"))
    if args.objective == "val_ic":
        return ic_mean
    if args.objective == "val_icir":
        return icir
    # Cost-aware combined objectives: avoid selecting high-turnover, rank-useless models.
    excess = metrics.get("excess_long_only_return", 0.0)
    ls_ret = metrics.get("long_short_total_return", 0.0)
    ls_sharpe = metrics.get("long_short_sharpe", 0.0)
    lo_sharpe_excess = metrics.get("long_only_sharpe", 0.0) - metrics.get("buy_hold_sharpe", 0.0)
    turnover = metrics.get("avg_long_turnover", 0.0)
    turnover_pen = args.turnover_penalty * max(0.0, turnover - 0.20)
    if args.objective == "val_alpha_combo":
        return float(1.00 * excess + 0.50 * ls_ret + 0.50 * ic_mean + 0.05 * lo_sharpe_excess + 0.05 * ls_sharpe + 0.10 * spread - turnover_pen)
    return float(excess + 0.05 * lo_sharpe_excess + 0.02 * ls_sharpe + 0.50 * ic_mean + 0.10 * spread - turnover_pen)


def plot_equity(best_daily: pd.DataFrame, output_dir: Path, filename: str = "best_equity_curve.png", title: str = "Test-set equity curves"):
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.plot(best_daily["date"], best_daily["long_only_equity"], label="Top-K long-only net")
    plt.plot(best_daily["date"], best_daily["long_only_equity_gross"], label="Top-K long-only gross", alpha=0.65)
    plt.plot(best_daily["date"], best_daily["long_short_equity"], label="Top-K minus bottom-K net")
    plt.plot(best_daily["date"], best_daily["buy_hold_equity"], label="Equal-weight buy-and-hold")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / filename, dpi=220)
    plt.close()


def plot_equity_by_model(best_daily_by_model: dict[str, pd.DataFrame], output_dir: Path):
    if not best_daily_by_model:
        return
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    buy_hold_drawn = False
    for model_name, daily in best_daily_by_model.items():
        plt.plot(daily["date"], daily["long_only_equity"], label=f"{model_name} top-k net")
        if not buy_hold_drawn:
            plt.plot(daily["date"], daily["buy_hold_equity"], label="Equal-weight buy-and-hold", linestyle="--")
            buy_hold_drawn = True
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.title("Best tuned trial within each model: net test long-only equity")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "best_by_model_equity_curve.png", dpi=220)
    plt.close()


def sanitize_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_").lower()


def eval_prediction(pred_df: pd.DataFrame, cfg: TrainConfig, cost_bps: float):
    metrics, daily, positions = backtest_cross_section(
        pred_df,
        top_frac=cfg.top_frac,
        transaction_cost_bps=cost_bps,
        score_ema=cfg.score_ema,
        portfolio_weighting=cfg.portfolio_weighting,
    )
    ic, icir = daily_spearman_ic(pred_df)
    spread = daily_top_bottom_spread(pred_df, cfg.top_frac)
    direction = evaluate_direction(pred_df)
    return metrics, daily, positions, ic, icir, spread, direction



def resolve_device(requested: str) -> str:
    """Resolve training device and fail loudly when CUDA is requested but unavailable."""
    requested = requested.lower()
    cuda_available = torch.cuda.is_available()
    if requested == "auto":
        return "cuda" if cuda_available else "cpu"
    if requested.startswith("cuda") and not cuda_available:
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "This usually means the current PyTorch build is CPU-only, the NVIDIA driver is missing, "
            "or the machine has no CUDA-capable NVIDIA GPU. Run: python src/check_gpu.py"
        )
    return requested


def print_device_info(device: str, amp: bool) -> None:
    print("[device]")
    print(f"  torch={torch.__version__}")
    print(f"  cuda_available={torch.cuda.is_available()}")
    print(f"  selected={device}")
    if torch.cuda.is_available():
        print(f"  cuda_version={torch.version.cuda}")
        print(f"  gpu_count={torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  gpu[{i}]={props.name}, memory={props.total_memory / 1024**3:.1f} GB")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print(f"  amp_enabled={amp}")
    else:
        print("  note=CUDA is not visible to PyTorch; training will run on CPU.")


def _date_zscore_scores(pred: pd.DataFrame) -> pd.DataFrame:
    p = pred.copy()
    mu = p.groupby("date")["score"].transform("mean")
    sd = p.groupby("date")["score"].transform("std").replace(0, np.nan)
    p["score"] = ((p["score"] - mu) / (sd + 1e-12)).fillna(0.0)
    return p


def make_ensemble_prediction(best_by_model: dict, model_names: list[str]) -> pd.DataFrame | None:
    frames = []
    for name in model_names:
        if name not in best_by_model:
            continue
        p = _date_zscore_scores(best_by_model[name]["test_pred"][["date", "ticker", "future_return", "next_return", "score"]])
        p = p.rename(columns={"score": f"score_{name}"})
        frames.append((name, p))
    if len(frames) < 2:
        return None
    base_name, base = frames[0]
    out = base.copy()
    score_cols = [f"score_{base_name}"]
    for name, f in frames[1:]:
        out = out.merge(f[["date", "ticker", f"score_{name}"]], on=["date", "ticker"], how="inner")
        score_cols.append(f"score_{name}")
    if out.empty:
        return None
    out["score"] = out[score_cols].mean(axis=1)
    return out[["date", "ticker", "future_return", "next_return", "score"]]

def main():
    args = parse_args()
    defaults = preset_defaults(args)
    if args.max_tickers is None:
        args.max_tickers = defaults["max_tickers"]
    if args.trials is None:
        args.trials = defaults["trials"]
    if args.epochs is None:
        args.epochs = defaults["epochs"]
    if args.models is None:
        args.models = defaults["models"]
    args.device = resolve_device(args.device)
    if args.device.startswith("cuda") and args.amp:
        pass
    elif args.amp and not args.device.startswith("cuda"):
        print("[warning] --amp is ignored because selected device is not CUDA.")
        args.amp = False

    if args.torch_threads and args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)
    (output_dir / "histories").mkdir(exist_ok=True)
    set_seed(args.seed)
    rng = random.Random(args.seed)

    print("=" * 96)
    print("Deep Learning Quant Alpha Project v5: IC-aware loss + ensemble + smoothing + cost-aware portfolio")
    print("=" * 96)
    print(
        f"source={args.source}, preset={args.preset}, models={args.models}, trials/model={args.trials}, "
        f"epochs={args.epochs}, max_tickers={args.max_tickers}, device={args.device}, amp={args.amp}, "
        f"num_workers={args.num_workers}, cost={args.transaction_cost_bps} bps, "
        f"target_mode={args.target_mode}, loss={args.loss}, objective={args.objective}, "
        f"lookbacks={args.lookbacks or 'default'}, horizons={args.horizons or 'default'}, top_fracs={args.top_fracs or 'default'}"
    )
    print_device_info(args.device, args.amp)
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False, default=str)

    if args.source == "kaggle_sp500":
        raw, source_name = load_kaggle_sp500()
    else:
        if not args.csv_path:
            raise ValueError("--csv-path is required for --source local")
        raw, source_name = load_local_csv(args.csv_path)

    df = filter_universe(
        raw,
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        min_rows=args.min_rows,
        max_tickers=args.max_tickers,
    )
    summary = data_source_summary(df, source_name)
    summary.to_csv(output_dir / "data_source_summary.csv", index=False)
    print(f"[data] rows={len(df)}, tickers={df['ticker'].nunique()}, dates={df['date'].min()} -> {df['date'].max()}")
    print("[data_source_summary head]")
    print(summary.head(10).to_string(index=False))

    feat_df, feature_cols = add_features(df, horizons=(1, 5, 10, 20))
    print(f"[features] rows={len(feat_df)}, n_features={len(feature_cols)}")
    pd.Series(feature_cols, name="feature").to_csv(output_dir / "feature_columns.csv", index=False)

    trial_rows = []
    best_by_model = {}
    best = {"score": -np.inf, "trial": None}
    global_trial = 0

    for model_name in args.models:
        print("=" * 96)
        print(f"[model tuning] model={model_name}, trials_per_model={args.trials}, objective={args.objective}")
        model_best = {"score": -np.inf, "trial": None, "model_name": model_name}

        for local_trial in range(1, args.trials + 1):
            global_trial += 1
            cfg = sample_config(
                rng,
                [model_name],
                args.epochs,
                args.batch_size,
                args.target_mode,
                args.loss,
                lookback_choices=args.lookbacks,
                horizon_choices=args.horizons,
                top_frac_choices=args.top_fracs,
                score_ema_choices=args.score_emas,
                weighting_choices=args.portfolio_weightings,
                ic_loss_weight=args.ic_loss_weight,
                pairwise_loss_weight=args.pairwise_loss_weight,
            )
            print("-" * 96)
            print(f"[model={model_name} trial {local_trial:03d}/{args.trials} | global {global_trial:03d}] {asdict(cfg)}")
            try:
                splits = make_sequences(
                    feat_df,
                    feature_cols=feature_cols,
                    lookback=cfg.lookback,
                    horizon=cfg.horizon,
                    train_end=args.train_end,
                    val_end=args.val_end,
                    target_mode=cfg.target_mode,
                )
                print(
                    f"    samples: train={len(splits['train'].y)}, val={len(splits['val'].y)}, test={len(splits['test'].y)}, "
                    f"features={len(feature_cols)}"
                )
                model, hist = train_model(
                    cfg,
                    splits["train"],
                    splits["val"],
                    len(feature_cols),
                    args.device,
                    verbose=True,
                    amp=args.amp,
                    num_workers=args.num_workers,
                )
                if args.save_histories:
                    hist.to_csv(output_dir / "histories" / f"history_{sanitize_name(model_name)}_trial_{local_trial:03d}.csv", index=False)

                val_pred = predict(model, splits["val"], cfg.batch_size, args.device, num_workers=args.num_workers)
                test_pred = predict(model, splits["test"], cfg.batch_size, args.device, num_workers=args.num_workers)
                if args.save_trial_predictions:
                    val_pred.to_csv(output_dir / "predictions" / f"val_pred_{sanitize_name(model_name)}_trial_{local_trial:03d}.csv", index=False)
                    test_pred.to_csv(output_dir / "predictions" / f"test_pred_{sanitize_name(model_name)}_trial_{local_trial:03d}.csv", index=False)

                val_metrics, val_daily, _, val_ic, val_icir, val_spread, val_dir = eval_prediction(val_pred, cfg, args.transaction_cost_bps)
                test_metrics, test_daily, test_positions, test_ic, test_icir, test_spread, test_dir = eval_prediction(test_pred, cfg, args.transaction_cost_bps)
                obj = objective_value(val_metrics, val_ic, val_icir, val_spread, args)

                row = {
                    "global_trial": global_trial,
                    "model_trial": local_trial,
                    "trial": global_trial,
                    **asdict(cfg),
                    "objective": args.objective,
                    "transaction_cost_bps": args.transaction_cost_bps,
                    "val_objective_value": obj,
                    "val_ic": val_ic,
                    "val_icir": val_icir,
                    "val_top_bottom_spread": val_spread,
                    "test_ic": test_ic,
                    "test_icir": test_icir,
                    "test_top_bottom_spread": test_spread,
                    **{f"val_{k}": v for k, v in val_dir.items()},
                    **{f"test_{k}": v for k, v in test_dir.items()},
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                    "failed": False,
                    "error": "",
                }
                trial_rows.append(row)
                print(
                    f"    val obj={obj:.4f}, val IC={val_ic:.4f}, test IC={test_ic:.4f}, "
                    f"test long-only net={test_metrics['long_only_total_return']:.3f}, "
                    f"gross={test_metrics['long_only_gross_total_return']:.3f}, "
                    f"buy-hold={test_metrics['buy_hold_total_return']:.3f}, "
                    f"long-short net={test_metrics['long_short_total_return']:.3f}, "
                    f"turnover={test_metrics['avg_long_turnover']:.2f}, ema={cfg.score_ema}, w={cfg.portfolio_weighting}"
                )

                if np.isfinite(obj) and obj > model_best["score"]:
                    model_best = {
                        "score": obj,
                        "trial": global_trial,
                        "model_trial": local_trial,
                        "model_name": model_name,
                        "model": model,
                        "cfg": cfg,
                        "splits": splits,
                        "test_daily": test_daily,
                        "test_positions": test_positions,
                        "test_pred": test_pred,
                        "row": row,
                    }
                    torch.save(model.state_dict(), output_dir / "models" / f"best_{sanitize_name(model_name)}.pt")

                if np.isfinite(obj) and obj > best["score"]:
                    best = {**model_best, "score": obj}
                    torch.save(model.state_dict(), output_dir / "models" / "best_global_model.pt")
            except Exception as exc:
                print(f"    [warning] trial failed: {type(exc).__name__}: {exc}")
                trial_rows.append({"global_trial": global_trial, "model_trial": local_trial, "trial": global_trial, **asdict(cfg), "failed": True, "error": f"{type(exc).__name__}: {exc}"})

            pd.DataFrame(trial_rows).to_csv(output_dir / "trials.csv", index=False)

        if model_best["trial"] is None:
            print(f"[warning] all trials failed for model={model_name}")
        else:
            best_by_model[model_name] = model_best
            model_best["test_daily"].to_csv(output_dir / f"best_test_daily_returns_{sanitize_name(model_name)}.csv", index=False)
            model_best["test_positions"].to_csv(output_dir / f"best_test_positions_{sanitize_name(model_name)}.csv", index=False)
            model_best["test_pred"].to_csv(output_dir / f"best_test_predictions_{sanitize_name(model_name)}.csv", index=False)
            with open(output_dir / f"best_config_{sanitize_name(model_name)}.json", "w", encoding="utf-8") as f:
                json.dump(asdict(model_best["cfg"]), f, indent=2, ensure_ascii=False, default=str)

    trials = pd.DataFrame(trial_rows)
    trials.to_csv(output_dir / "trials.csv", index=False)
    if not best_by_model:
        raise RuntimeError("All trials failed. Check data and hyperparameters.")

    best_by_model_rows = [state["row"] for state in best_by_model.values() if state.get("row") is not None]
    best_by_model_df = pd.DataFrame(best_by_model_rows)

    # Optional model ensemble, useful for stabilizing alpha signals across model families.
    if args.enable_ensemble:
        ens_pred = make_ensemble_prediction(best_by_model, args.ensemble_models)
        if ens_pred is not None:
            ens_cfg = TrainConfig(
                model="ensemble",
                lookback=0,
                horizon=best_by_model_df["horizon"].mode().iloc[0] if "horizon" in best_by_model_df else 10,
                hidden_dim=0,
                dropout=0.0,
                lr=0.0,
                weight_decay=0.0,
                batch_size=args.batch_size,
                epochs=0,
                patience=0,
                top_frac=args.ensemble_top_frac,
                target_mode=args.target_mode,
                loss="ensemble",
                score_ema=args.ensemble_score_ema,
                portfolio_weighting=args.ensemble_weighting,
            )
            ens_metrics, ens_daily, ens_positions, ens_ic, ens_icir, ens_spread, ens_dir = eval_prediction(ens_pred, ens_cfg, args.transaction_cost_bps)
            ens_row = {
                "global_trial": -1, "model_trial": 0, "trial": -1, "model": "ensemble",
                **asdict(ens_cfg),
                "objective": "ensemble",
                "transaction_cost_bps": args.transaction_cost_bps,
                "val_objective_value": np.nan,
                "val_ic": np.nan,
                "test_ic": ens_ic,
                "test_icir": ens_icir,
                "test_top_bottom_spread": ens_spread,
                **{f"test_{k}": v for k, v in ens_dir.items()},
                **{f"test_{k}": v for k, v in ens_metrics.items()},
                "failed": False, "error": "",
            }
            ens_daily.to_csv(output_dir / "best_test_daily_returns_ensemble.csv", index=False)
            ens_positions.to_csv(output_dir / "best_test_positions_ensemble.csv", index=False)
            ens_pred.to_csv(output_dir / "best_test_predictions_ensemble.csv", index=False)
            best_by_model_rows.append(ens_row)
            best_by_model_df = pd.DataFrame(best_by_model_rows)
            best_by_model["ensemble"] = {"score": float(ens_metrics.get("excess_long_only_return", 0.0)), "trial": -1, "model_trial": 0, "model_name": "ensemble", "cfg": ens_cfg, "test_daily": ens_daily, "test_positions": ens_positions, "test_pred": ens_pred, "row": ens_row}
            print(f"[ensemble] models={args.ensemble_models}, IC={ens_ic:.4f}, long-only={ens_metrics['long_only_total_return']:.3f}, excess={ens_metrics['excess_long_only_return']:.3f}, turnover={ens_metrics['avg_long_turnover']:.2f}")
        else:
            print("[ensemble] skipped: not enough compatible best-model predictions.")

    best_by_model_df.to_csv(output_dir / "best_by_model.csv", index=False)

    report_cols = [
        "model", "model_trial", "lookback", "horizon", "hidden_dim", "dropout", "lr", "weight_decay", "top_frac", "target_mode", "loss", "score_ema", "portfolio_weighting", "transaction_cost_bps",
        "val_objective_value", "val_ic", "test_ic", "test_icir", "test_top_bottom_spread",
        "test_long_only_total_return", "test_long_only_gross_total_return", "test_buy_hold_total_return", "test_excess_long_only_return",
        "test_long_only_sharpe", "test_buy_hold_sharpe", "test_excess_long_only_sharpe",
        "test_long_short_total_return", "test_long_short_sharpe", "test_long_only_max_drawdown", "test_buy_hold_max_drawdown",
        "test_avg_long_turnover", "test_total_long_only_cost",
    ]
    report_cols = [c for c in report_cols if c in best_by_model_df.columns]
    best_by_model_df[report_cols].sort_values("model").to_csv(output_dir / "model_comparison.csv", index=False)

    latex_cols = [
        "model", "lookback", "horizon", "top_frac", "score_ema", "portfolio_weighting", "test_ic", "test_long_only_total_return", "test_buy_hold_total_return",
        "test_excess_long_only_return", "test_long_only_sharpe", "test_buy_hold_sharpe", "test_long_short_total_return", "test_avg_long_turnover",
    ]
    latex_cols = [c for c in latex_cols if c in best_by_model_df.columns]
    with open(output_dir / "model_comparison_latex.tex", "w", encoding="utf-8") as f:
        f.write(best_by_model_df[latex_cols].sort_values("model").to_latex(index=False, float_format="%.4f"))

    # Cost-sensitivity table for the selected model of each family.
    cost_rows = []
    for model_name, state in best_by_model.items():
        cfg = state["cfg"]
        pred = state["test_pred"]
        for c in sorted(set(args.cost_sweep_bps + [args.transaction_cost_bps])):
            m, _, _, ic, icir, spread, _ = eval_prediction(pred, cfg, c)
            cost_rows.append({
                "model": model_name,
                "cost_bps": c,
                "lookback": cfg.lookback,
                "horizon": cfg.horizon,
                "top_frac": cfg.top_frac,
                "test_ic": ic,
                "test_icir": icir,
                "test_top_bottom_spread": spread,
                **{f"test_{k}": v for k, v in m.items()},
            })
    cost_df = pd.DataFrame(cost_rows)
    cost_df.to_csv(output_dir / "cost_sensitivity.csv", index=False)

    best_state = max(best_by_model.values(), key=lambda s: s["score"])
    best_daily = best_state["test_daily"]
    best_positions = best_state["test_positions"]
    best_pred = best_state["test_pred"]
    best_daily.to_csv(output_dir / "best_global_test_daily_returns.csv", index=False)
    best_positions.to_csv(output_dir / "best_global_test_positions.csv", index=False)
    best_pred.to_csv(output_dir / "best_global_test_predictions.csv", index=False)
    best_daily.to_csv(output_dir / "best_test_daily_returns.csv", index=False)
    best_positions.to_csv(output_dir / "best_test_positions.csv", index=False)
    best_pred.to_csv(output_dir / "best_test_predictions.csv", index=False)
    with open(output_dir / "best_global_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(best_state["cfg"]), f, indent=2, ensure_ascii=False, default=str)
    with open(output_dir / "best_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(best_state["cfg"]), f, indent=2, ensure_ascii=False, default=str)
    pd.DataFrame([best_state["row"]]).to_csv(output_dir / "best_global_metrics.csv", index=False)
    pd.DataFrame([best_state["row"]]).to_csv(output_dir / "best_metrics.csv", index=False)
    plot_equity(best_daily, output_dir, title="Globally selected model: net-of-cost test equity")
    plot_equity_by_model({m: state["test_daily"] for m, state in best_by_model.items()}, output_dir)

    print("=" * 96)
    print("Finished. Key outputs:")
    print(f"  {output_dir / 'trials.csv'}                         # every hyperparameter trial")
    print(f"  {output_dir / 'best_by_model.csv'}                  # one tuned winner per model family")
    print(f"  {output_dir / 'model_comparison.csv'}               # report-ready net-of-cost comparison")
    print(f"  {output_dir / 'cost_sensitivity.csv'}               # transaction-cost sensitivity")
    print(f"  {output_dir / 'model_comparison_latex.tex'}         # LaTeX table")
    print(f"  {output_dir / 'figures' / 'best_by_model_equity_curve.png'}")
    print("\nBest tuned trial within each model family, net of transaction costs:")
    model_show_cols = [
        "model", "model_trial", "lookback", "horizon", "top_frac", "target_mode", "val_objective_value", "test_ic",
        "test_long_only_total_return", "test_long_only_gross_total_return", "test_buy_hold_total_return", "test_excess_long_only_return",
        "test_long_only_sharpe", "test_buy_hold_sharpe", "test_long_short_total_return", "test_long_short_sharpe", "test_avg_long_turnover",
    ]
    print(best_by_model_df[[c for c in model_show_cols if c in best_by_model_df.columns]].sort_values("model").to_string(index=False))

    print("\nCost sensitivity for selected model of each family:")
    cost_show_cols = ["model", "cost_bps", "test_long_only_total_return", "test_excess_long_only_return", "test_long_only_sharpe", "test_avg_long_turnover"]
    print(cost_df[[c for c in cost_show_cols if c in cost_df.columns]].sort_values(["model", "cost_bps"]).to_string(index=False))

    print("\nGlobal winner across all model families, shown only as an additional reference:")
    best_table = pd.DataFrame([best_state["row"]])
    show_cols = ["trial", "model", "lookback", "horizon", "top_frac", "val_objective_value", "test_ic", "test_long_only_total_return", "test_long_only_sharpe", "test_buy_hold_total_return", "test_buy_hold_sharpe", "test_long_short_total_return", "test_long_short_sharpe"]
    print(best_table[[c for c in show_cols if c in best_table.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
