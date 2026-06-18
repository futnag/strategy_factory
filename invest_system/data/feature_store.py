"""Features (Gold) 層：Silver wide から派生特徴量を materialize（再計算・PIT安全）。

価格由来の普遍特徴（returns / log_returns / realized vol / momentum / reversal）と、市場
レジーム（vol 三分位・トレンド）を `data/features/*.parquet` に書き出す。すべて adj_close から
**因果的に計算＝先読みなし**（特徴量 f[t] は ≤t のみ参照）。特徴量は進化するため append でなく
**再計算**（差分 append の Silver とは扱いが異なる）。

分数階差分は研究依存(d)＋高コストのため bulk materialize に含めない（`features/frac_diff.py` を
オンデマンド適用）。レジームは KB §8-3 の構造変化検知の v1 ヒューリスティック（vol 分位＋トレンド）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .store import load_wide


def _feat_dir(base) -> Path:
    return Path(base) / "features"


def _write(df: pd.DataFrame, name: str, base) -> None:
    d = _feat_dir(base)
    d.mkdir(parents=True, exist_ok=True)
    df.to_parquet(d / f"{name}.parquet")


def load_feature(name: str, base: str = "data", start=None, end=None) -> pd.DataFrame:
    """Features 層の派生特徴を読む（wide or market-level）。"""
    fp = _feat_dir(base) / f"{name}.parquet"
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_parquet(fp)
    df.index = pd.to_datetime(df.index)
    if start is not None:
        df = df.loc[pd.Timestamp(start):]
    if end is not None:
        df = df.loc[:pd.Timestamp(end)]
    return df


def tradability_mask(base: str = "data") -> pd.DataFrame:
    """約定可能性マスク（True=約定可能）。C3(arXiv:2507.07107) の mask-first 用。

    引け張り付き（`frictions.limit_lock_flags`＝UL/LL × 引け値）・出来高ゼロの
    （銘柄, 日）を False にする。Silver の close/high/low/upper_limit/lower_limit/
    volume から構築。必要フィールドが未 materialize なら全 True（マスク無効）。
    """
    from ..equities.frictions import limit_lock_flags

    close = load_wide("close", base=base)
    if close.empty:
        return pd.DataFrame()
    high, low = load_wide("high", base=base), load_wide("low", base=base)
    ul = load_wide("upper_limit", base=base)
    ll = load_wide("lower_limit", base=base)
    vo = load_wide("volume", base=base)
    if high.empty or low.empty or ul.empty or ll.empty:
        return pd.DataFrame(True, index=close.index, columns=close.columns)
    no_buy, no_sell = limit_lock_flags(close, high, low, ul, ll,
                                       volume=(None if vo.empty else vo))
    return ~(no_buy | no_sell)


def build_price_features(base: str = "data", vol_window: int = 20,
                         mom_lookback: int = 252, mom_skip: int = 21,
                         rev_window: int = 5,
                         mask_non_tradable: bool = False) -> dict:
    """adj_close（Silver）→ returns/log_returns/vol/momentum/reversal を wide で materialize。

    すべて ≤t のみ参照：returns[t]=adjC[t]/adjC[t-1]-1、momentum=adjC[t-skip]/adjC[t-lb]-1
    （直近月除外）、reversal=-(adjC[t]/adjC[t-w]-1)、vol=直近窓の実現ボラ(年率)。

    mask_non_tradable: True で C3 の mask-first＝**約定不能日（引け張り付き・出来高ゼロ）
    の価格を NaN にしてから**全特徴量を計算する（上流汚染の遮断）。既定 False（現状維持＝
    過去研究の再現性保持）。docs/03 §6.21 の監査で旗艦構成（月次・流動性上位300）への
    影響は無視できる規模と測定済みだが、日次・小型・イベント系の価格研究は True を前提と
    すること。
    """
    px = load_wide("adj_close", base=base)
    if px.empty:
        return {}
    px = px.sort_index()
    if mask_non_tradable:
        tm = tradability_mask(base=base)
        if not tm.empty:
            px = px.where(tm.reindex_like(px).fillna(True))
    ret = px.pct_change()
    out = {
        "returns": ret,
        "log_returns": np.log(px).diff(),
        f"vol_{vol_window}": ret.rolling(
            vol_window, min_periods=max(5, vol_window // 2)).std() * np.sqrt(252),
        "momentum_12_1": px.shift(mom_skip) / px.shift(mom_lookback) - 1.0,
        f"reversal_{rev_window}": -(px / px.shift(rev_window) - 1.0),
    }
    for name, df in out.items():
        _write(df.astype("float32"), name, base)
    return {k: [int(v.shape[0]), int(v.shape[1])] for k, v in out.items()}


def build_regime(base: str = "data", vol_window: int = 60, trend_window: int = 200,
                 min_periods: int = 252) -> dict:
    """市場レジーム（等加重マーケットの vol 分位＋トレンド）を PIT で materialize。

    vol_regime は「市場実現ボラの **拡張窓 percentile**（≤t 分布）」の三分位（0=低/1=中/2=高）＝
    先読みなし。trend_up は等加重マーケット水準が trailing MA を上回るか。market-level（1行/日）。
    """
    ret = load_feature("returns", base=base)
    if ret.empty:
        px = load_wide("adj_close", base=base)
        if px.empty:
            return {}
        ret = px.sort_index().pct_change()
    mkt = ret.mean(axis=1)                                  # 等加重マーケット日次リターン
    mkt_vol = mkt.rolling(vol_window, min_periods=vol_window // 2).std() * np.sqrt(252)
    vol_pct = mkt_vol.expanding(min_periods=min_periods).rank(pct=True)  # ≤t percentile
    vol_regime = pd.cut(vol_pct, [0.0, 1 / 3, 2 / 3, 1.0], labels=[0, 1, 2],
                        include_lowest=True).astype("float64")
    level = (1.0 + mkt.fillna(0.0)).cumprod()
    trend_up = (level > level.rolling(
        trend_window, min_periods=trend_window // 2).mean()).astype("float64")
    reg = pd.DataFrame({"mkt_ret": mkt, "mkt_vol": mkt_vol, "vol_pct": vol_pct,
                        "vol_regime": vol_regime, "trend_up": trend_up})
    _write(reg, "regime", base)
    return {"regime": [int(reg.shape[0]), int(reg.shape[1])]}


def _equal_weight_market(adj_close: pd.DataFrame) -> pd.Series:
    """等加重マーケット日次リターン（build_regime と同一規約・ivol/beta の市場系列）。"""
    return adj_close.pct_change().mean(axis=1)


def _load_sector() -> pd.Series:
    """S33 業種（index=Code・equities_master）。未取得なら空 Series。業種モメンタム/中立化用。"""
    try:
        from ..data.sources import jquants as jq
        listed = jq.fetch_listed_info()
    except Exception:  # noqa: BLE001
        return pd.Series(dtype=object)
    if listed.empty or "S33" not in listed.columns:
        return pd.Series(dtype=object)
    return listed.assign(Code=listed["Code"].astype(str)).set_index("Code")["S33"]


def _monthly_turnover(va: pd.DataFrame, base: str, month_ends, window: int = 60):
    """回転率（月次）：trailing60日平均Va / 月末時価総額に負号。

    時価総額=生株価×(発行済−自己株式)。株数は fins_summary の as-of（J-Quants・EDINET非依存）。
    株価/株数が無ければ None（turnover をスキップ）。
    """
    from ..equities import fundamentals as fu
    raw_close = load_wide("close", base=base)
    if raw_close.empty or not month_ends:
        return None
    sh = fu.fundamentals_panel(list(month_ends), ["ShOutFY", "TrShFY"])
    if "ShOutFY" not in sh:
        return None
    shares = sh["ShOutFY"].sub(sh.get("TrShFY", 0.0), fill_value=0.0)
    shares = shares.where(shares > 0)
    close_me = raw_close.resample("ME").last().reindex(index=month_ends)
    mcap_me = (close_me * shares).where(lambda x: x > 0)
    avg_va_me = (va.rolling(window, min_periods=max(2, int(window * 0.8))).mean()
                 .resample("ME").last().reindex(index=month_ends))
    common = mcap_me.columns.intersection(avg_va_me.columns)
    return -(avg_va_me[common] / mcap_me[common])


def build_price_liquidity_features(base: str = "data", monthly: bool = True) -> dict:
    """価格系・流動性系（GKX Phase 1）を月次 PIT で materialize（float32・wide）。

    日次で計算（trailing 窓・先読みなし）→ 月末（暦月の最終営業日値）に as-of サンプリング
    （monthly=True）。市場=等加重・業種=S33・回転率の時価総額=生株価×as-of株数。raw を書き出し、
    [-1,1]ランク版/セクター中立版は price_factor_view() で既存ユーティリティから生成する。
    """
    from ..equities import price_factors as pf
    adj = load_wide("adj_close", base=base)
    if adj.empty:
        return {}
    adj = adj.sort_index()
    va = load_wide("turnover", base=base)            # Va=売買代金（Silver の turnover フィールド）
    sector = _load_sector()
    fac = pf.compute_all(adj, va=(None if va.empty else va),
                         market=_equal_weight_market(adj), mcap=None,
                         sector=(None if sector.empty else sector))
    out: dict[str, list] = {}
    month_ends = list(adj.resample("ME").last().index) if monthly else None
    for name, df in fac.items():
        m = df.resample("ME").last() if monthly else df
        _write(m.astype("float32"), name, base)
        out[name] = [int(m.shape[0]), int(m.shape[1])]
    if not va.empty and monthly:                     # 回転率は月末時価総額と合成して別途
        tov = _monthly_turnover(va, base, month_ends)
        if tov is not None and not tov.empty:
            _write(tov.astype("float32"), "turnover", base)
            out["turnover"] = [int(tov.shape[0]), int(tov.shape[1])]
    return out


def price_factor_view(name: str, view: str = "rank", base: str = "data") -> pd.DataFrame:
    """材化済み price/liquidity ファクターの 3 ビュー（raw/zscore/rank/sector_neutral）を生成。

    既存ユーティリティ（factors.cross_sectional_zscore/cross_sectional_rank/sector_neutralize）を
    再利用。raw は材化値そのまま。sector_neutral は S33 内デミーン。
    """
    from ..equities import factors as fac
    df = load_feature(name, base=base)
    if df.empty or view == "raw":
        return df
    if view == "zscore":
        return fac.cross_sectional_zscore(df)
    if view == "rank":
        return fac.cross_sectional_rank(df)
    if view == "sector_neutral":
        return fac.sector_neutralize(df, _load_sector())
    raise ValueError(f"unknown view: {view}")


def materialize_features(base: str = "data") -> dict:
    """Features 層の標準 materialize（価格特徴 → レジーム → 価格/流動性ファクター）。再計算で冪等。"""
    rep = {"price": build_price_features(base=base)}
    rep["regime"] = build_regime(base=base)
    rep["price_liquidity"] = build_price_liquidity_features(base=base)
    return rep
