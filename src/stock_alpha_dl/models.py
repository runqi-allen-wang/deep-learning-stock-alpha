from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPAlpha(nn.Module):
    def __init__(self, lookback: int, n_features: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        inp = lookback * n_features
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(inp),
            nn.Linear(inp, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class ResMLPAlpha(nn.Module):
    def __init__(self, lookback: int, n_features: int, hidden_dim: int = 128, dropout: float = 0.1, depth: int = 3):
        super().__init__()
        inp = lookback * n_features
        self.inp = nn.Sequential(nn.Flatten(), nn.LayerNorm(inp), nn.Linear(inp, hidden_dim), nn.GELU())
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        z = self.inp(x)
        z = self.blocks(z)
        return self.head(z).squeeze(-1)


class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x):
        w = torch.softmax(self.score(x).squeeze(-1), dim=1)
        return torch.sum(x * w.unsqueeze(-1), dim=1)


class RNNAlpha(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        rnn_type: str = "gru",
        attention_pool: bool = True,
    ):
        super().__init__()
        cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = cls(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.pool = AttentionPool(hidden_dim) if attention_pool else None
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        z = self.pool(out) if self.pool is not None else out[:, -1]
        return self.head(z).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerAlpha(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pool = AttentionPool(d_model)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, x):
        z = self.pos(self.proj(x))
        z = self.encoder(z)
        z = self.pool(z)
        return self.head(z).squeeze(-1)


class PatchTSTAlpha(nn.Module):
    """A compact PatchTST-style model for stock-alpha scoring."""

    def __init__(
        self,
        lookback: int,
        n_features: int,
        patch_len: int = 5,
        stride: int = 5,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if lookback < patch_len:
            raise ValueError("lookback must be >= patch_len")
        self.patch_len = patch_len
        self.stride = stride
        self.n_features = n_features
        self.patch_proj = nn.Linear(n_features * patch_len, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pool = AttentionPool(d_model)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, x):
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)  # [B, P, F, patch_len]
        patches = patches.permute(0, 1, 3, 2).contiguous()  # [B, P, patch_len, F]
        patches = patches.flatten(start_dim=2)
        z = self.pos(self.patch_proj(patches))
        z = self.encoder(z)
        z = self.pool(z)
        return self.head(z).squeeze(-1)


class TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.padding = padding

    def forward(self, x):
        y = self.conv(x)
        if self.padding > 0:
            y = y[:, :, :-self.padding]
        y = self.dropout(F.gelu(self.norm(y)))
        return x + y


class TCNAlpha(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int = 64, dropout: float = 0.1, depth: int = 3):
        super().__init__()
        self.inp = nn.Conv1d(n_features, hidden_dim, kernel_size=1)
        self.blocks = nn.Sequential(*[TCNBlock(hidden_dim, 3, 2**i, dropout) for i in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        z = x.transpose(1, 2)  # [B, F, L]
        z = self.inp(z)
        z = self.blocks(z)
        z = z[:, :, -1]
        return self.head(z).squeeze(-1)


class DLinearAlpha(nn.Module):
    """DLinear-style linear time-series baseline with trend/seasonal decomposition."""

    def __init__(self, lookback: int, n_features: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.lookback = lookback
        self.n_features = n_features
        kernel = min(25, lookback if lookback % 2 == 1 else lookback - 1)
        kernel = max(3, kernel)
        self.avg = nn.AvgPool1d(kernel_size=kernel, stride=1, padding=kernel // 2)
        self.seasonal = nn.Linear(lookback * n_features, hidden_dim)
        self.trend = nn.Linear(lookback * n_features, hidden_dim)
        self.head = nn.Sequential(nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * hidden_dim, 1))

    def forward(self, x):
        # x [B, L, F]
        xt = x.transpose(1, 2)  # [B, F, L]
        trend = self.avg(xt)
        if trend.shape[-1] != self.lookback:
            trend = trend[..., : self.lookback]
        trend = trend.transpose(1, 2)
        seasonal = x - trend
        s = self.seasonal(seasonal.flatten(1))
        t = self.trend(trend.flatten(1))
        return self.head(torch.cat([s, t], dim=-1)).squeeze(-1)


def build_model(name: str, lookback: int, n_features: int, hidden_dim: int, dropout: float):
    name = name.lower()
    if name == "mlp":
        return MLPAlpha(lookback, n_features, hidden_dim=hidden_dim, dropout=dropout)
    if name == "resmlp":
        return ResMLPAlpha(lookback, n_features, hidden_dim=hidden_dim, dropout=dropout)
    if name == "gru":
        return RNNAlpha(n_features, hidden_dim=hidden_dim, num_layers=1, dropout=dropout, rnn_type="gru")
    if name == "lstm":
        return RNNAlpha(n_features, hidden_dim=hidden_dim, num_layers=1, dropout=dropout, rnn_type="lstm")
    if name == "tcn":
        return TCNAlpha(n_features, hidden_dim=hidden_dim, dropout=dropout)
    if name == "dlinear":
        return DLinearAlpha(lookback, n_features, hidden_dim=hidden_dim, dropout=dropout)
    if name == "transformer":
        d_model = max(32, hidden_dim)
        return TransformerAlpha(n_features, d_model=d_model, nhead=4, num_layers=2, dropout=dropout)
    if name == "patchtst":
        d_model = max(32, hidden_dim)
        patch_len = 5 if lookback <= 30 else 10
        stride = max(1, patch_len // 2)
        return PatchTSTAlpha(
            lookback, n_features, patch_len=patch_len, stride=stride, d_model=d_model, nhead=4, num_layers=2, dropout=dropout
        )
    raise ValueError(f"Unknown model: {name}")
