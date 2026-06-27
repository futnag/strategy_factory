"""深層 条件付き SDF（候補② Stage2・docs/46）。GRU 潜在マクロ状態 → no-arbitrage SDF。

Chen-Pelger-Zhu の最小忠実版：GRU が**マクロ水準の系列**から潜在経済状態 h_t を抽出し（recurrent＝
動的状態）、小MLP が条件付き接線ローディング b_t = g(h_t) を出す。SDF M_t = 1 − b_t·F_t（F=特性管理
ポートフォリオ）。学習は no-arbitrage の GMM 損失 Σ_j E[M_t F_{j,t}]^2 を最小化（GAN 無しの固定モーメント版）。

**負のコントロール**＝GRU を Linear（最新マクロ水準のみ）に置換＝「最新の増分では動的状態を表せない」を実証する。
データが小さい（月次~100点）ため**極小ネット＋weight decay＋少epoch**で過学習を抑える。torch 必須（optional [dl]）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _torch():
    try:
        import torch
        return torch
    except ImportError as e:  # pragma: no cover
        raise ImportError("torch が必要です: pip install torch（または pip install \".[dl]\"）") from e


def _build_model(torch, n_macro: int, K: int, state_dim: int = 4, hidden: int = 8,
                 use_gru: bool = True):
    nn = torch.nn

    class SDFNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.use_gru = use_gru
            if use_gru:
                self.enc = nn.GRU(n_macro, state_dim, batch_first=True)
            else:                                   # 負コントロール：最新水準のみ（recurrence無し）
                self.enc = nn.Sequential(nn.Linear(n_macro, state_dim), nn.Tanh())
            self.head = nn.Sequential(nn.Linear(state_dim, hidden), nn.Tanh(),
                                      nn.Linear(hidden, K))

        def forward(self, macro_seq):               # macro_seq: (T, n_macro)
            if self.use_gru:
                out, _ = self.enc(macro_seq.unsqueeze(0))   # (1,T,state)・因果的
                h = out.squeeze(0)
            else:
                h = self.enc(macro_seq)             # (T,state) 各 t で最新水準のみ
            return self.head(h)                     # (T,K) = b_t
    return SDFNet()


def train_sdf(F_arr: np.ndarray, macro_seq: np.ndarray, *, use_gru: bool = True,
              epochs: int = 400, lr: float = 0.01, weight_decay: float = 0.1,
              state_dim: int = 4, hidden: int = 8, seed: int = 0):
    """no-arbitrage GMM 損失で SDF ネットを学習。F_arr/macro_seq は学習窓の連続系列（NaN許容）。"""
    torch = _torch()
    torch.manual_seed(int(seed))
    T, K = F_arr.shape
    n_macro = macro_seq.shape[1]
    valid = np.all(np.isfinite(F_arr), axis=1).astype(np.float32)
    Ft = torch.tensor(np.nan_to_num(F_arr, nan=0.0), dtype=torch.float32)
    seqt = torch.tensor(np.nan_to_num(macro_seq, nan=0.0), dtype=torch.float32)
    vt = torch.tensor(valid, dtype=torch.float32)
    model = _build_model(torch, n_macro, K, state_dim, hidden, use_gru)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    wnorm = vt / vt.sum().clamp(min=1.0)
    for _ in range(int(epochs)):
        opt.zero_grad()
        b = model(seqt)                             # (T,K)
        M = 1.0 - (b * Ft).sum(dim=1)               # (T,)
        moments = ((M * wnorm).unsqueeze(1) * Ft).sum(dim=0)   # E[M F_j]（有効tのみ加重）
        loss = (moments ** 2).sum() + 1e-3 * (b * b).mean()
        loss.backward()
        opt.step()
    return model


def predict_b(model, macro_seq: np.ndarray) -> np.ndarray:
    """学習済モデルでマクロ系列 → b_t（T×K・因果的＝各 t は ≤t のみ参照）。"""
    torch = _torch()
    with torch.no_grad():
        seqt = torch.tensor(np.nan_to_num(macro_seq, nan=0.0), dtype=torch.float32)
        return model(seqt).numpy()


def walk_forward_deep_weights(F: pd.DataFrame, zdict: dict, macro_df: pd.DataFrame, *,
                              use_gru: bool = True, refit: int = 12, min_train: int = 36,
                              **train_kw) -> pd.DataFrame:
    """walk-forward：refit ヶ月ごとに ≤i で学習→次 refit ヶ月の b_t を予測→SDF株式ウェイト。

    GRU は因果的なので、フル系列を渡しても b_t は ≤t のみ参照（先読み無し）。モデル係数は ≤i で固定。
    返り値 W（Date×Code・各行ダラーニュートラル・グロス1）。
    """
    chars = list(F.columns)
    codes = zdict[chars[0]].columns
    dates = F.index
    Farr = F.to_numpy()
    Marr = macro_df.reindex(dates).ffill().to_numpy()
    rows = {}
    i = min_train
    while i < len(dates):
        end = min(i + refit, len(dates))
        model = train_sdf(Farr[:i], Marr[:i], use_gru=use_gru, **train_kw)
        b_all = predict_b(model, Marr[:end])        # (end,K)・b_t は ≤t 参照、係数は ≤i
        for ti in range(i, end):
            t = dates[ti]
            b = b_all[ti]
            w = None
            for k, ch in enumerate(chars):
                if t in zdict[ch].index:
                    term = float(b[k]) * zdict[ch].loc[t]
                    w = term if w is None else w.add(term, fill_value=0.0)
            if w is None:
                continue
            w = w.dropna()
            w = w - w.mean()
            g = float(w.abs().sum())
            if g > 0:
                rows[t] = w / g
        i = end
    return pd.DataFrame(rows).T.reindex(columns=codes) if rows \
        else pd.DataFrame(columns=codes)
