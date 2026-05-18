from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import AlphaSequenceDataset, SplitData
from .models import build_model


def _is_cuda(device: str) -> bool:
    return str(device).startswith("cuda")


@dataclass
class TrainConfig:
    model: str
    lookback: int
    horizon: int
    hidden_dim: int
    dropout: float
    lr: float
    weight_decay: float
    batch_size: int
    epochs: int
    patience: int
    top_frac: float
    target_mode: str = "rank"
    loss: str = "smoothl1_ic"
    ic_loss_weight: float = 0.10
    pairwise_loss_weight: float = 0.05
    score_ema: float = 1.0
    portfolio_weighting: str = "equal"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _base_loss(pred: torch.Tensor, y: torch.Tensor, name: str) -> torch.Tensor:
    name = name.lower()
    if name in {"mse"}:
        return nn.functional.mse_loss(pred, y)
    if name in {"huber", "smoothl1", "smoothl1_ic", "smoothl1_pairwise", "smoothl1_ic_pairwise"}:
        return nn.functional.smooth_l1_loss(pred, y, beta=0.05)
    if name in {"ic", "pairwise"}:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    raise ValueError(f"Unknown loss: {name}")


def _corr_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    pred = pred.float()
    y = y.float()
    pred = pred - pred.mean()
    y = y - y.mean()
    denom = pred.std(unbiased=False) * y.std(unbiased=False) + 1e-8
    corr = (pred * y).mean() / denom
    return 1.0 - corr


def _pairwise_rank_loss(pred: torch.Tensor, y: torch.Tensor, max_pairs: int = 4096) -> torch.Tensor:
    # Pairwise logistic ranking loss: if y_i > y_j, encourage s_i > s_j.
    n = pred.shape[0]
    if n < 2:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    if n * n <= max_pairs:
        ds = pred[:, None] - pred[None, :]
        dy = y[:, None] - y[None, :]
        mask = torch.abs(dy) > 1e-6
        if not mask.any():
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        sign = torch.sign(dy[mask])
        return torch.nn.functional.softplus(-sign * ds[mask]).mean()
    idx_i = torch.randint(0, n, (max_pairs,), device=pred.device)
    idx_j = torch.randint(0, n, (max_pairs,), device=pred.device)
    dy = y[idx_i] - y[idx_j]
    mask = torch.abs(dy) > 1e-6
    if not mask.any():
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    ds = pred[idx_i] - pred[idx_j]
    sign = torch.sign(dy[mask])
    return torch.nn.functional.softplus(-sign * ds[mask]).mean()


def compute_loss(pred: torch.Tensor, y: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    name = cfg.loss.lower()
    loss = _base_loss(pred, y, name)
    if "ic" in name:
        loss = loss + cfg.ic_loss_weight * _corr_loss(pred, y)
    if "pairwise" in name:
        loss = loss + cfg.pairwise_loss_weight * _pairwise_rank_loss(pred, y)
    if name == "ic":
        loss = _corr_loss(pred, y)
    if name == "pairwise":
        loss = _pairwise_rank_loss(pred, y)
    return loss


def train_model(
    cfg: TrainConfig,
    train: SplitData,
    val: SplitData,
    n_features: int,
    device: str,
    verbose: bool = True,
    amp: bool = False,
    num_workers: int = 0,
):
    train_ds = AlphaSequenceDataset(train)
    val_ds = AlphaSequenceDataset(val)
    pin_memory = _is_cuda(device)
    loader_kwargs = dict(num_workers=num_workers, pin_memory=pin_memory)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False, **loader_kwargs)

    model = build_model(cfg.model, cfg.lookback, n_features, cfg.hidden_dim, cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs))
    use_amp = bool(amp and _is_cuda(device))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=pin_memory)
            yb = yb.to(device, non_blocking=pin_memory)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(xb)
                loss = compute_loss(pred, yb, cfg)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))
        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=pin_memory)
                yb = yb.to(device, non_blocking=pin_memory)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = model(xb)
                    loss = compute_loss(pred, yb, cfg)
                val_losses.append(float(loss.detach().cpu()))

        tr = float(np.mean(train_losses))
        va = float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va, "lr": opt.param_groups[0]["lr"]})
        if verbose and (epoch == 1 or epoch == cfg.epochs or epoch % 5 == 0):
            print(f"    [{cfg.model}] epoch={epoch:03d} train_loss={tr:.6f} val_loss={va:.6f}")

        if va < best_val - 1e-7:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                if verbose:
                    print(f"    [{cfg.model}] early stop at epoch={epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def predict(model, split: SplitData, batch_size: int, device: str, num_workers: int = 0) -> pd.DataFrame:
    ds = AlphaSequenceDataset(split)
    pin_memory = _is_cuda(device)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=pin_memory)
            score = model(xb).detach().cpu().numpy()
            preds.append(score)
    pred = np.concatenate(preds)
    out = split.meta.copy().reset_index(drop=True)
    out["score"] = pred
    return out
