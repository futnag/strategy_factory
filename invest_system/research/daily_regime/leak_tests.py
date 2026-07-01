"""daily_regime: リーク検出ハーネス（step 0–1・釘①）。

「壊れていれば落ちる」ことを先に証明するための道具：
  (1) null 生成器（ブロックシャッフル／位相ランダム化＝**自己相関を保持**）。
  (2) プラセボ・エッジ検定（null 下でエッジが説明されてしまうか）。
  (3) 構造 PIT スライス検査。
  (4) **意図的リーク注入版**（テストで『赤』を確認する用）。

素朴な IID シャッフルは自己相関を壊し検定力を失うため使わない（接続設計 §6・§2.9）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import LookAheadError


# === null 生成（自己相関を保持） =========================================
def block_shuffle(x: pd.Series, block: int, *, seed: int = 0) -> pd.Series:
    """連続ブロック単位でシャッフル（ブロック内の自己相関を保持）。"""
    rng = np.random.default_rng(seed)
    v = np.asarray(x, dtype=float)
    n = len(v)
    nb = int(np.ceil(n / block))
    blocks = [v[i * block:(i + 1) * block] for i in range(nb)]
    order = rng.permutation(nb)
    out = np.concatenate([blocks[i] for i in order])[:n]
    return pd.Series(out, index=x.index)


def phase_randomize(x: pd.Series, *, seed: int = 0) -> pd.Series:
    """位相ランダム化サロゲート（パワースペクトル＝自己相関を保持）。"""
    rng = np.random.default_rng(seed)
    v = np.asarray(x, dtype=float)
    mu = np.nanmean(v)
    v0 = np.nan_to_num(v - mu)
    n = len(v0)
    F = np.fft.rfft(v0)
    mag = np.abs(F)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=len(F))
    phases[0] = 0.0                       # DC は実数
    if n % 2 == 0:
        phases[-1] = 0.0                  # Nyquist も実数
    sur = np.fft.irfft(mag * np.exp(1j * phases), n=n)
    return pd.Series(sur + mu, index=x.index)


# === プラセボ・エッジ検定 ================================================
def edge_stat(signal: pd.Series, fwd_ret: pd.Series) -> float:
    """符号エッジ = mean(sign(signal) · fwd_ret)。"""
    s, r = signal.align(fwd_ret, join="inner")
    m = s.notna() & r.notna()
    if not m.any():
        return 0.0
    return float((np.sign(s[m]) * r[m]).mean())


def placebo_pvalue(
    signal: pd.Series, fwd_ret: pd.Series, *,
    null: str = "block", block: int = 20, n: int = 200, seed: int = 0,
) -> float:
    """null 下のエッジ分布に対する両側 p 値（自己相関保持シャッフル）。小さいほど『本物のエッジ』。"""
    obs = edge_stat(signal, fwd_ret)
    rng = np.random.default_rng(seed)
    null_stats = np.empty(n)
    for i in range(n):
        s = int(rng.integers(0, 2**31 - 1))
        sh = block_shuffle(signal, block, seed=s) if null == "block" else phase_randomize(signal, seed=s)
        null_stats[i] = edge_stat(sh, fwd_ret)
    return float((np.abs(null_stats) >= abs(obs)).mean())


# === 構造 PIT スライス検査 ===============================================
def assert_pit_slicing(view, field: str, t) -> None:
    """AsOfView.asof(t).frame(field) が ≤t 行しか持たないことをアサート。"""
    f = view.asof(t).frame(field)
    if len(f) and pd.Timestamp(f.index.max()) > pd.Timestamp(t):
        raise LookAheadError(f"{field}: asof({t}) に未来行 {f.index.max()}")


# === 意図的リーク注入（テストで『赤』を確認する用） ======================
def leaky_anchor_daily_panel(anchor, daily_index, value_col, anchor_id) -> pd.Series:
    """⚠リーク版：available_from でなく **week_end_date** で可視化（公表ラグを無視）。"""
    a = anchor[anchor["anchor_id"].astype(str) == str(anchor_id)]
    src = pd.DataFrame({
        "k": pd.to_datetime(a["week_end_date"].values),
        "v": a[value_col].astype(float).values,
    }).sort_values("k")
    daily = pd.DataFrame({"d": pd.to_datetime(pd.DatetimeIndex(daily_index))}).sort_values("d")
    out = pd.merge_asof(daily, src, left_on="d", right_on="k", direction="backward")
    return pd.Series(out["v"].values, index=pd.DatetimeIndex(out["d"].values))


def leaky_pool_index(membership: pd.DataFrame, tier: str, asof) -> pd.MultiIndex:
    """⚠リーク版：as-of(asof) のティアを過去へ**遡及適用**して (date,code) を作る。"""
    asof = pd.Timestamp(asof)
    final = membership.loc[asof]
    codes = [c for c in membership.columns if final.get(c) == tier]
    dates = membership.loc[:asof].index
    return pd.MultiIndex.from_product([dates, codes], names=["date", "code"])


def leaky_future_signal(fwd_ret: pd.Series) -> pd.Series:
    """⚠リーク版：未来リターンそのものをシグナル化（完全な先読み）。"""
    return fwd_ret.copy()
