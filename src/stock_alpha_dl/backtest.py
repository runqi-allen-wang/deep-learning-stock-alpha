from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


def sharpe_ratio(daily_returns: pd.Series) -> float:
    daily_returns = pd.Series(daily_returns).dropna()
    if len(daily_returns) < 2 or daily_returns.std(ddof=1) == 0:
        return float("nan")
    return float(np.sqrt(252) * daily_returns.mean() / daily_returns.std(ddof=1))


def max_drawdown(equity: pd.Series) -> float:
    equity = pd.Series(equity).dropna()
    if equity.empty:
        return float("nan")
    peak = equity.cummax()
    dd = equity / peak - 1
    return float(dd.min())


def daily_spearman_ic(pred_df: pd.DataFrame, target_col: str = "future_return") -> tuple[float, float]:
    ics = []
    for _, g in pred_df.groupby("date"):
        if len(g) < 5:
            continue
        ic = g["score"].rank().corr(g[target_col].rank())
        if np.isfinite(ic):
            ics.append(ic)
    if not ics:
        return float("nan"), float("nan")
    ics_s = pd.Series(ics)
    icir = float(np.sqrt(252) * ics_s.mean() / (ics_s.std(ddof=1) + 1e-12))
    return float(ics_s.mean()), icir


def daily_top_bottom_spread(pred_df: pd.DataFrame, top_frac: float = 0.10, target_col: str = "future_return") -> float:
    spreads = []
    for _, g in pred_df.groupby("date"):
        g = g.dropna(subset=["score", target_col])
        if len(g) < 10:
            continue
        k = max(1, int(round(len(g) * top_frac)))
        ranked = g.sort_values("score", ascending=False)
        spreads.append(float(ranked.head(k)[target_col].mean() - ranked.tail(k)[target_col].mean()))
    return float(np.nanmean(spreads)) if spreads else float("nan")


def evaluate_direction(pred_df: pd.DataFrame) -> dict[str, float]:
    y_true = (pred_df["future_return"] > 0).astype(int).to_numpy()
    score = pred_df["score"].to_numpy()
    y_pred = (score > 0).astype(int)
    out = {"direction_acc": float(accuracy_score(y_true, y_pred))}
    try:
        out["direction_auc"] = float(roc_auc_score(y_true, score))
    except Exception:
        out["direction_auc"] = float("nan")
    return out


def _weight_series(frame: pd.DataFrame, side: str, gross: float = 1.0, weighting: str = "equal") -> dict[str, float]:
    tickers = list(frame["ticker"].astype(str))
    if not tickers:
        return {}
    sign = 1.0 if side == "long" else -1.0
    weighting = weighting.lower()
    if weighting == "inv_vol" and "risk_vol" in frame.columns:
        vol = frame["risk_vol"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(frame["risk_vol"].median())
        inv = 1.0 / (vol.clip(lower=1e-4) + 1e-8)
        if np.isfinite(inv).all() and inv.sum() > 0:
            weights = gross * inv / inv.sum()
            return {str(t): sign * float(w) for t, w in zip(tickers, weights)}
    w = gross / len(tickers)
    return {str(t): sign * w for t in tickers}


def _turnover(new_w: dict[str, float], old_w: dict[str, float]) -> float:
    """Dollar turnover for rebalancing from old weights to new weights.

    We use 0.5 * sum |w_new - w_old|, a common portfolio turnover convention.
    A fully invested long-only portfolio changing every name has turnover close to 1.
    """
    names = set(new_w) | set(old_w)
    return 0.5 * sum(abs(new_w.get(t, 0.0) - old_w.get(t, 0.0)) for t in names)


def backtest_cross_section(
    pred_df: pd.DataFrame,
    top_frac: float = 0.10,
    transaction_cost_bps: float = 0.0,
    long_short: bool = True,
    score_ema: float = 1.0,
    portfolio_weighting: str = "equal",
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Daily-rebalanced cross-sectional top-k backtest with realistic weight turnover.

    The model score is computed at date t; realized PnL uses next_return from t to t+1.
    Long-only holds top_frac of stocks by score. Long-short is long top_frac and short
    bottom_frac, each dollar-neutral with long gross = 1 and short gross = 1.

    Transaction cost is charged as `cost * turnover`, where turnover is computed from
    equal-weight portfolio weights. This is more faithful than set-difference counting
    and makes the reported returns explicitly net-of-fees.
    """
    pred_df = pred_df.copy()
    pred_df["date"] = pd.to_datetime(pred_df["date"])
    pred_df = pred_df.sort_values(["ticker", "date"])
    # Optional score smoothing. This reduces turnover by using an exponential moving
    # average of model scores within each ticker. score_ema=1 means no smoothing.
    alpha = float(score_ema)
    if alpha < 1.0:
        alpha = max(0.01, min(1.0, alpha))
        pred_df["score_raw"] = pred_df["score"]
        pred_df["score"] = pred_df.groupby("ticker")["score"].transform(lambda s: s.ewm(alpha=alpha, adjust=False).mean())
    # Risk estimate for inverse-volatility weighting. Uses only prior next_return
    # observations within the test/validation period, so it is a conservative approximation.
    pred_df["risk_vol"] = pred_df.groupby("ticker")["next_return"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).std())
    pred_df["risk_vol"] = pred_df["risk_vol"].fillna(pred_df["risk_vol"].median()).fillna(0.02)
    pred_df = pred_df.sort_values(["date", "ticker"])

    daily = []
    pos_rows = []
    prev_long_w: dict[str, float] = {}
    prev_ls_w: dict[str, float] = {}
    cost = transaction_cost_bps / 10000.0

    for date, g in pred_df.groupby("date", sort=True):
        g = g.dropna(subset=["score", "next_return"])
        n = len(g)
        if n < 10:
            continue
        k = max(1, int(round(n * top_frac)))
        ranked = g.sort_values("score", ascending=False)
        long = ranked.head(k).copy()
        short = ranked.tail(k).copy()

        long_w = _weight_series(long, "long", gross=1.0, weighting=portfolio_weighting)
        short_w = _weight_series(short, "short", gross=1.0, weighting=portfolio_weighting)
        ls_w = {**long_w}
        for t, w in short_w.items():
            ls_w[t] = ls_w.get(t, 0.0) + w

        long_turnover = _turnover(long_w, prev_long_w)
        ls_turnover = _turnover(ls_w, prev_ls_w)
        prev_long_w, prev_ls_w = long_w, ls_w

        long_gross = float(sum(long_w[str(r.ticker)] * r.next_return for r in long.itertuples(index=False)))
        short_component = float(sum(short_w[str(r.ticker)] * r.next_return for r in short.itertuples(index=False)))
        ls_gross = long_gross + short_component
        buy_hold_ret = float(g["next_return"].mean())

        long_net = long_gross - cost * long_turnover
        ls_net = ls_gross - cost * ls_turnover

        daily.append(
            {
                "date": date,
                "long_only_return_gross": long_gross,
                "long_only_return": long_net,
                "long_only_cost": cost * long_turnover,
                "long_short_return_gross": ls_gross,
                "long_short_return": ls_net if long_short else np.nan,
                "long_short_cost": cost * ls_turnover if long_short else np.nan,
                "buy_hold_return": buy_hold_ret,
                "n_assets": n,
                "k": k,
                "long_turnover": long_turnover,
                "long_short_turnover": ls_turnover,
            }
        )
        for _, row in long.iterrows():
            pos_rows.append({"date": date, "ticker": row["ticker"], "side": "long", "score": row["score"]})
        if long_short:
            for _, row in short.iterrows():
                pos_rows.append({"date": date, "ticker": row["ticker"], "side": "short", "score": row["score"]})

    daily_df = pd.DataFrame(daily)
    if daily_df.empty:
        raise ValueError("Backtest produced no daily rows. Check prediction coverage.")
    positions = pd.DataFrame(pos_rows)

    for col in ["long_only_return", "long_only_return_gross", "long_short_return", "long_short_return_gross", "buy_hold_return"]:
        if col in daily_df.columns:
            daily_df[col.replace("return", "equity")] = (1 + daily_df[col].fillna(0.0)).cumprod()

    metrics: dict[str, float] = {}
    for name, ret_col, eq_col in [
        ("long_only", "long_only_return", "long_only_equity"),
        ("long_only_gross", "long_only_return_gross", "long_only_equity_gross"),
        ("long_short", "long_short_return", "long_short_equity"),
        ("long_short_gross", "long_short_return_gross", "long_short_equity_gross"),
        ("buy_hold", "buy_hold_return", "buy_hold_equity"),
    ]:
        if ret_col not in daily_df.columns or eq_col not in daily_df.columns:
            continue
        total_return = float(daily_df[eq_col].iloc[-1] - 1)
        metrics[f"{name}_total_return"] = total_return
        metrics[f"{name}_sharpe"] = sharpe_ratio(daily_df[ret_col])
        metrics[f"{name}_max_drawdown"] = max_drawdown(daily_df[eq_col])

    metrics["excess_long_only_return"] = metrics.get("long_only_total_return", np.nan) - metrics.get("buy_hold_total_return", np.nan)
    metrics["excess_long_only_sharpe"] = metrics.get("long_only_sharpe", np.nan) - metrics.get("buy_hold_sharpe", np.nan)
    metrics["excess_long_only_return_gross"] = metrics.get("long_only_gross_total_return", np.nan) - metrics.get("buy_hold_total_return", np.nan)
    metrics["avg_long_turnover"] = float(daily_df["long_turnover"].mean())
    metrics["avg_long_short_turnover"] = float(daily_df["long_short_turnover"].mean())
    metrics["avg_turnover"] = metrics["avg_long_turnover"]
    metrics["total_long_only_cost"] = float(daily_df["long_only_cost"].sum())
    metrics["total_long_short_cost"] = float(daily_df["long_short_cost"].sum()) if "long_short_cost" in daily_df else float("nan")
    metrics["score_ema"] = float(score_ema)
    metrics["portfolio_weighting_inv_vol"] = 1.0 if str(portfolio_weighting).lower() == "inv_vol" else 0.0
    return metrics, daily_df, positions
