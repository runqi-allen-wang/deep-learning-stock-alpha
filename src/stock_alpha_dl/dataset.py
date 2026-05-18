from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class SplitData:
    X: np.ndarray
    y: np.ndarray
    meta: pd.DataFrame


class AlphaSequenceDataset(Dataset):
    def __init__(self, split_data: SplitData):
        self.X = torch.tensor(split_data.X, dtype=torch.float32)
        self.y = torch.tensor(split_data.y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def _transform_target(meta: pd.DataFrame, raw_y: np.ndarray, mode: str, clip: float) -> np.ndarray:
    meta = meta.copy()
    meta["raw_y"] = raw_y
    mode = mode.lower()
    if mode == "raw":
        y = np.clip(raw_y, -clip, clip)
    elif mode == "zscore":
        mu = meta.groupby("date")["raw_y"].transform("mean")
        sd = meta.groupby("date")["raw_y"].transform("std").replace(0, np.nan)
        y = ((meta["raw_y"] - mu) / (sd + 1e-12)).clip(-5, 5).fillna(0.0).to_numpy()
    elif mode == "rank":
        # Cross-sectional rank label in [-1, 1]. This directly optimizes the stock-selection task.
        pct = meta.groupby("date")["raw_y"].rank(pct=True, method="average")
        y = (2.0 * (pct - 0.5)).fillna(0.0).to_numpy()
    elif mode == "rank_zscore":
        pct = meta.groupby("date")["raw_y"].rank(pct=True, method="average")
        rank_y = 2.0 * (pct - 0.5)
        mu = rank_y.groupby(meta["date"]).transform("mean")
        sd = rank_y.groupby(meta["date"]).transform("std").replace(0, np.nan)
        y = ((rank_y - mu) / (sd + 1e-12)).clip(-5, 5).fillna(0.0).to_numpy()
    else:
        raise ValueError(f"Unknown target_mode={mode}. Use raw, zscore, rank, or rank_zscore.")
    return np.asarray(y, dtype=np.float32)


def make_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    lookback: int,
    horizon: int,
    train_end: str,
    val_end: str,
    target_clip: float = 0.20,
    target_mode: str = "rank",
) -> dict[str, SplitData]:
    """Create rolling windows split by current decision date.

    Each sample uses features from t-lookback+1,...,t and predicts target_h{horizon}.
    The raw future returns remain in meta for IC and backtesting. The training label can
    be raw, cross-sectionally z-scored, or cross-sectionally rank-normalized.
    """
    target_col = f"target_h{horizon}"
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    train_end_ts = pd.to_datetime(train_end)
    val_end_ts = pd.to_datetime(val_end)
    buckets = {"train": {"X": [], "raw_y": [], "meta": []}, "val": {"X": [], "raw_y": [], "meta": []}, "test": {"X": [], "raw_y": [], "meta": []}}

    for ticker, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        values = g[feature_cols].to_numpy(dtype=np.float32)
        dates = pd.to_datetime(g["date"]).to_numpy()
        target = g[target_col].to_numpy(dtype=np.float32)
        next_ret = g["target_h1"].to_numpy(dtype=np.float32)

        for i in range(lookback - 1, len(g)):
            if not np.isfinite(target[i]) or not np.isfinite(next_ret[i]):
                continue
            x = values[i - lookback + 1 : i + 1]
            if not np.isfinite(x).all():
                continue
            date = pd.Timestamp(dates[i])
            if date <= train_end_ts:
                split = "train"
            elif date <= val_end_ts:
                split = "val"
            else:
                split = "test"
            buckets[split]["X"].append(x)
            buckets[split]["raw_y"].append(float(target[i]))
            buckets[split]["meta"].append(
                {
                    "date": date,
                    "ticker": ticker,
                    "future_return": float(target[i]),
                    "next_return": float(next_ret[i]),
                }
            )

    out: dict[str, SplitData] = {}
    for split, obj in buckets.items():
        if not obj["X"]:
            raise ValueError(f"No samples in {split}; check split dates or lookback.")
        meta = pd.DataFrame(obj["meta"])
        raw_y = np.asarray(obj["raw_y"], dtype=np.float32)
        y = _transform_target(meta, raw_y, target_mode, target_clip)
        out[split] = SplitData(
            X=np.stack(obj["X"]).astype(np.float32),
            y=y,
            meta=meta,
        )
    return out
