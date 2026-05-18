from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

REQUIRED_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]


def _standardize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common OHLCV column names into a single schema."""
    lower_map = {c: str(c).strip().lower() for c in df.columns}
    df = df.rename(columns=lower_map)

    rename = {}
    if "name" in df.columns and "ticker" not in df.columns:
        rename["name"] = "ticker"
    if "symbol" in df.columns and "ticker" not in df.columns:
        rename["symbol"] = "ticker"
    if "adj close" in df.columns and "adj_close" not in df.columns:
        rename["adj close"] = "adj_close"
    df = df.rename(columns=rename)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required columns after standardization: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    out = df[REQUIRED_COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"])
    out["ticker"] = out["ticker"].astype(str).str.upper()
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=REQUIRED_COLUMNS)
    out = out.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"])
    return out.reset_index(drop=True)


def _candidate_cache_roots(dataset: str) -> list[Path]:
    """Return likely local cache roots for KaggleHub datasets.

    This is intentionally conservative: it only searches common project/cache
    directories and never generates synthetic data. It helps when Kaggle API
    access fails due to SSL/rate-limit/network problems but the CSV was already
    downloaded in a previous run.
    """
    owner, name = dataset.split("/", 1) if "/" in dataset else ("", dataset)
    roots: list[Path] = []

    # Project-local locations. Users can manually copy the Kaggle CSV here.
    cwd = Path.cwd()
    roots.extend([
        cwd / "data" / "raw",
        cwd.parent / "data" / "raw",
        Path(__file__).resolve().parents[2] / "data" / "raw",
    ])

    # KaggleHub's default cache on Windows/Linux/macOS.
    home = Path.home()
    roots.extend([
        home / ".cache" / "kagglehub" / "datasets" / owner / name,
        home / ".cache" / "kagglehub" / "datasets",
    ])

    # User-provided cache roots, if any.
    for env_name in ["KAGGLEHUB_CACHE", "KAGGLE_CACHE", "XDG_CACHE_HOME"]:
        val = os.environ.get(env_name)
        if val:
            root = Path(val)
            roots.extend([root, root / "kagglehub" / "datasets" / owner / name])

    # Deduplicate while preserving order.
    out: list[Path] = []
    seen: set[str] = set()
    for r in roots:
        key = str(r.resolve()) if r.exists() else str(r)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def find_cached_kaggle_csv(dataset: str = "camnugent/sandp500") -> Optional[Path]:
    """Find a previously downloaded all_stocks_5yr.csv without hitting network."""
    candidate_names = ["all_stocks_5yr.csv", "sp500_stocks.csv"]
    for root in _candidate_cache_roots(dataset):
        if not root.exists():
            continue
        # Prefer exact known filenames.
        for name in candidate_names:
            hits = sorted(root.rglob(name)) if root.is_dir() else []
            if hits:
                return hits[0]
        # Otherwise use any CSV that looks like OHLCV data.
        hits = sorted(root.rglob("*.csv")) if root.is_dir() else []
        if hits:
            return hits[0]
    return None


def _read_kaggle_csv(csv_path: Path, dataset: str) -> tuple[pd.DataFrame, str]:
    print(f"[kaggle-cache] using file: {csv_path}")
    df = pd.read_csv(csv_path)
    return _standardize_ohlcv_columns(df), f"kaggle:{dataset}/{csv_path.name}"


def load_kaggle_sp500(dataset: str = "camnugent/sandp500") -> tuple[pd.DataFrame, str]:
    """Load Kaggle S&P 500 all_stocks_5yr.csv.

    Robust behavior:
    1. First try local project/cache copies, which avoids SSL failures.
    2. If not found, try kagglehub download.
    3. If kagglehub fails, search the local cache again and then raise a clear
       error with a local-CSV command. This never falls back to synthetic data.
    """
    cached = find_cached_kaggle_csv(dataset)
    if cached is not None:
        return _read_kaggle_csv(cached, dataset)

    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "Please install kagglehub, or download all_stocks_5yr.csv manually and run: "
            "python src/train_alpha.py --source local --csv-path data/raw/all_stocks_5yr.csv ..."
        ) from exc

    print(f"[download] kaggle dataset={dataset}")
    try:
        path = Path(kagglehub.dataset_download(dataset))
        candidates = sorted(path.rglob("*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No CSV file found in Kaggle dataset cache: {path}")
        preferred = [p for p in candidates if "all_stocks_5yr" in p.name]
        csv_path = preferred[0] if preferred else candidates[0]
        print(f"[kaggle] using file: {csv_path.name}")
        df = pd.read_csv(csv_path)
        return _standardize_ohlcv_columns(df), f"kaggle:{dataset}/{csv_path.name}"
    except Exception as exc:
        cached = find_cached_kaggle_csv(dataset)
        if cached is not None:
            print(f"[warning] KaggleHub failed ({type(exc).__name__}: {exc}). Falling back to local cache.")
            return _read_kaggle_csv(cached, dataset)
        raise RuntimeError(
            "KaggleHub failed and no local cached CSV was found. This is usually a network/SSL issue, "
            "not a modeling bug. Fix it by manually downloading Kaggle dataset camnugent/sandp500, "
            "placing all_stocks_5yr.csv under data/raw/, and running:\n"
            "  python src/train_alpha.py --source local --csv-path data/raw/all_stocks_5yr.csv --preset quick\n"
            "On Windows, if you already downloaded it before, you can copy it from KaggleHub cache with:\n"
            "  Get-ChildItem $env:USERPROFILE\\.cache\\kagglehub\\datasets\\camnugent\\sandp500 -Recurse "
            "-Filter all_stocks_5yr.csv | Select-Object -First 1 | Copy-Item -Destination data\\raw\\all_stocks_5yr.csv\n"
        ) from exc


def load_local_csv(csv_path: str | os.PathLike[str]) -> tuple[pd.DataFrame, str]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Local CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return _standardize_ohlcv_columns(df), f"local:{csv_path}"


def filter_universe(
    df: pd.DataFrame,
    tickers: Optional[Iterable[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    min_rows: int = 900,
    max_tickers: Optional[int] = None,
) -> pd.DataFrame:
    out = df.copy()
    if start:
        out = out[out["date"] >= pd.to_datetime(start)]
    if end:
        out = out[out["date"] <= pd.to_datetime(end)]
    if tickers:
        ticker_set = {t.upper() for t in tickers}
        out = out[out["ticker"].isin(ticker_set)]

    counts = out.groupby("ticker").size().sort_values(ascending=False)
    keep = counts[counts >= min_rows].index.tolist()
    if max_tickers is not None:
        keep = keep[:max_tickers]
    out = out[out["ticker"].isin(keep)]
    if out.empty:
        raise ValueError(
            "No ticker remains after filtering. Try smaller --min-rows, wider date range, or different tickers."
        )
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def data_source_summary(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    rows = []
    for ticker, g in df.groupby("ticker"):
        rows.append(
            {
                "Ticker": ticker,
                "DataSource": source_name,
                "n_rows": len(g),
                "start_date": g["date"].min().date().isoformat(),
                "end_date": g["date"].max().date().isoformat(),
            }
        )
    return pd.DataFrame(rows).sort_values("Ticker").reset_index(drop=True)
