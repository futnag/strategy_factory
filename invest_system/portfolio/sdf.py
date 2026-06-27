"""条件付き no-arbitrage SDF（候補② Stage 1・docs/46 事前登録）。

特性管理ポートフォリオ F_t = Σ_i z_{i,t} r_{i,t+1} を作り、KNS（Kozak-Nagel-Santosh）の
**縮小接線** b = (Σ_F + γI)^{-1} μ_F で no-arbitrage SDF を推定する。条件付きは「特性 × 潜在状態」の
相互作用ファクター（F_c × S_m）を加えて接線を取る＝有効ローディング b_eff,c が状態の関数になる。

すべて numpy/sklearn 不要（線形代数のみ）・PIT（各 t で実現済み F のみで b を推定）。深層SDF（torch）は後段。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def managed_portfolios(zdict: dict, fwd: pd.DataFrame) -> pd.DataFrame:
    """{char: Date×Code z}（PIT・標準化済）＋ fwd（Date×Code 次期リターン）→ F（Date×char）。

    各 t で横断的にデミーン（ダラーニュートラル）した z を次期リターンに掛けた等加重ファクター
    リターン F_c[t] = Σ_i (z_{i,t}−z̄_t) r_{i,t+1} / N_t。
    """
    cols = list(zdict)
    n = fwd.notna().sum(axis=1).replace(0, np.nan)
    out = {}
    for c in cols:
        z = zdict[c].reindex(index=fwd.index, columns=fwd.columns)
        zc = z.sub(z.mean(axis=1), axis=0)
        out[c] = (zc * fwd).sum(axis=1) / n
    return pd.DataFrame(out, index=fwd.index)[cols]


def ridge_tangency(F: pd.DataFrame, c: float = 1.0):
    """KNS 縮小接線 b = (Σ + γI)^{-1} μ。γ = c × 平均分散（比例縮小・チューニング無し）。"""
    X = np.asarray(F.dropna(), dtype=float)
    if X.ndim != 2 or X.shape[0] < X.shape[1] + 2:
        return None
    mu = X.mean(axis=0)
    S = np.cov(X, rowvar=False)
    S = np.atleast_2d(S)
    gamma = c * float(np.mean(np.diag(S)))
    try:
        return np.linalg.solve(S + gamma * np.eye(S.shape[0]), mu)
    except np.linalg.LinAlgError:
        return None


def _aug_terms(chars: list, state: pd.DataFrame | None) -> list:
    """augmented ファクターの項リスト [(char, None) ... , (char, state_col) ...]。"""
    terms = [(c, None) for c in chars]
    if state is not None:
        terms += [(c, m) for c in chars for m in state.columns]
    return terms


def augment_factors(F: pd.DataFrame, state: pd.DataFrame | None):
    """F（Date×char）→ [F, F_c×S_m]（条件付き相互作用）。返り値 (Faug, terms)。"""
    chars = list(F.columns)
    terms = _aug_terms(chars, state)
    data = {}
    for (c, m) in terms:
        if m is None:
            data[c] = F[c]
        else:
            data[f"{c}*{m}"] = F[c] * state[m].reindex(F.index)
    return pd.DataFrame(data, index=F.index), terms


def effective_loadings(b, terms, chars: list, s_row: pd.Series) -> dict:
    """augmented 係数 b ＋ 状態行 → 各 char の有効ローディング b_eff,c = b0_c + Σ_m B_cm S_m。"""
    beff = {c: 0.0 for c in chars}
    for coef, (c, m) in zip(b, terms):
        beff[c] += float(coef) * (1.0 if m is None else float(s_row.get(m, 0.0)))
    return beff


def walk_forward_weights(F: pd.DataFrame, zdict: dict, state: pd.DataFrame | None = None,
                         c: float = 1.0, min_train: int = 36) -> pd.DataFrame:
    """walk-forward に各 t で（t 以前の実現 F に）縮小接線を当て、SDF 株式ウェイトを返す。

    state=None で無条件、state=Date×M で条件付き（最新マクロ水準を渡せば負のコントロール）。
    返り値 W（Date×Code・各行ダラーニュートラル・グロス1）。PIT：b は F.iloc[:i]（t 未実現を除く）で推定、
    状態は state.loc[t]（≤t 既知）、ウェイトは z_{i,t} を使い t→t+1 を保有。
    """
    chars = list(F.columns)
    Faug, terms = augment_factors(F, state)
    codes = zdict[chars[0]].columns
    dates = F.index
    rows = {}
    for i, t in enumerate(dates):
        if i < min_train:
            continue
        b = ridge_tangency(Faug.iloc[:i], c=c)
        if b is None:
            continue
        s_row = (state.loc[t] if (state is not None and t in state.index)
                 else pd.Series(dtype=float))
        beff = effective_loadings(b, terms, chars, s_row)
        w = None
        for ch in chars:
            if t not in zdict[ch].index:
                continue
            term = beff[ch] * zdict[ch].loc[t]
            w = term if w is None else w.add(term, fill_value=0.0)
        if w is None:
            continue
        w = w.dropna()
        w = w - w.mean()                      # ダラーニュートラル
        g = float(w.abs().sum())
        if g > 0:
            rows[t] = w / g                   # グロス1
    return pd.DataFrame(rows).T.reindex(columns=codes) if rows \
        else pd.DataFrame(columns=codes)
