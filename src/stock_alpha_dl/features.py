from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / window, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / window, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - 100 / (1 + rs)


def add_features(df: pd.DataFrame, horizons: tuple[int, ...] = (1, 5, 10, 20)) -> tuple[pd.DataFrame, list[str]]:
    """Create alpha-style features and future-return targets.

    All input features use information available at the close of date t. Targets are
    future returns after t. In addition to technical indicators, we add market-relative
    features because cross-sectional stock selection often benefits from removing the
    broad market component.
    """
    chunks = []
    for ticker, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").copy()
        close = g["close"]
        high = g["high"]
        low = g["low"]
        open_ = g["open"]
        volume = g["volume"].replace(0, np.nan)

        g["ret_1"] = close.pct_change(1)
        g["log_ret_1"] = np.log(close).diff(1)
        for w in [2, 3, 5, 10, 20, 30, 60]:
            g[f"mom_{w}"] = close / close.shift(w) - 1
            g[f"vol_{w}"] = g["ret_1"].rolling(w).std()
            g[f"ma_ratio_{w}"] = close / close.rolling(w).mean() - 1
            g[f"volume_z_{w}"] = (volume - volume.rolling(w).mean()) / (volume.rolling(w).std() + 1e-12)

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        g["macd_norm"] = macd / (close + 1e-12)
        g["macd_signal_norm"] = signal / (close + 1e-12)
        g["rsi_14"] = _rsi(close, 14) / 100.0
        mid = close.rolling(20).mean()
        std = close.rolling(20).std()
        g["bb_pos_20"] = (close - mid) / (2 * std + 1e-12)
        g["hl_range"] = (high - low) / (close + 1e-12)
        g["co_return"] = (close - open_) / (open_ + 1e-12)
        g["dollar_volume_log"] = np.log1p(close * volume)

        for h in horizons:
            g[f"target_h{h}"] = close.shift(-h) / close - 1
        chunks.append(g)

    out = pd.concat(chunks, axis=0, ignore_index=True)

    # Market-relative features computed using same-day cross-sectional information only.
    market_ret_1 = out.groupby("date")["ret_1"].transform("mean")
    out["mkt_ret_1"] = market_ret_1
    out["rel_ret_1"] = out["ret_1"] - market_ret_1
    for w in [5, 10, 20, 60]:
        mkt_mom = out.groupby("date")[f"mom_{w}"].transform("mean")
        mkt_vol = out.groupby("date")[f"vol_{w}"].transform("mean")
        out[f"rel_mom_{w}"] = out[f"mom_{w}"] - mkt_mom
        out[f"rel_vol_{w}"] = out[f"vol_{w}"] - mkt_vol

    raw_features = ["ret_1", "log_ret_1", "hl_range", "co_return", "dollar_volume_log", "mkt_ret_1", "rel_ret_1"]
    alpha_features = [
        c
        for c in out.columns
        if c.startswith(("mom_", "vol_", "ma_ratio_", "volume_z_", "rel_mom_", "rel_vol_"))
        or c in ["macd_norm", "macd_signal_norm", "rsi_14", "bb_pos_20"]
    ]
    feature_cols = raw_features + alpha_features

    # Cross-sectional standardization by date. This is common for ranking models and
    # uses only same-day information available at portfolio formation.
    for c in feature_cols:
        mu = out.groupby("date")[c].transform("mean")
        sd = out.groupby("date")[c].transform("std")
        out[c] = (out[c] - mu) / (sd + 1e-12)
        out[c] = out[c].clip(-8, 8)

    needed = feature_cols + ["target_h1"]
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=needed)
    return out.sort_values(["ticker", "date"]).reset_index(drop=True), feature_cols
