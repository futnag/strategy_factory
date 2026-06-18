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


def build_holdings_features(base: str = "data") -> dict:
    """信用/空売りの**公表日アンカー**月次PIT特徴量を材化（float32・wide・GKX A-1）。

    holdings_factors の long（available_date 付き）を `fundamentals.point_in_time`
    （date_col="available_date", lag_days=0＝公表日 ≤t で as-of）に流す。基準日でなく公表日で
    乗るため未来漏れなし。符号：margin_imbalance/short_to_long/margin_balance_change/days_to_cover
    は raw、short_interest は**負号**（高SI→低リターン）、sector_short_ratio は S33→銘柄に展開。
    """
    from ..equities import fundamentals as fu
    from ..equities import holdings_factors as hf
    from ..equities import margin as mg
    adj = load_wide("adj_close", base=base)
    if adj.empty:
        return {}
    month_ends = list(adj.resample("ME").last().index)
    cal = hf.trading_days(base)
    out: dict[str, list] = {}

    wl = hf.weekly_margin_long(mg.load_weekly_margin(), cal)   # 週次信用 +2 営業日
    if not wl.empty:
        flds = ["margin_imbalance", "short_to_long", "margin_balance_change"]
        pw = fu.point_in_time(wl, month_ends, flds, date_col="available_date",
                              code_col="Code", lag_days=0)
        for name in flds:
            _write(pw[name].astype("float32"), name, base)
            out[name] = [int(pw[name].shape[0]), int(pw[name].shape[1])]

    sl = hf.short_position_long(mg.load_short_positions())     # 大量空売り DiscDate
    if not sl.empty:
        ps = fu.point_in_time(sl, month_ends, ["short_interest", "short_shares"],
                              date_col="available_date", code_col="Code", lag_days=0)
        si = (-ps["short_interest"]).astype("float32")         # 負号（空売りアノマリー）
        _write(si, "short_interest", base)
        out["short_interest"] = [int(si.shape[0]), int(si.shape[1])]
        vol = load_wide("volume", base=base)                   # days-to-cover（残株数/平均出来高）
        if not vol.empty:
            avgvol = (vol.rolling(20, min_periods=10).mean()
                      .resample("ME").last().reindex(index=month_ends))
            ss = ps["short_shares"]
            common = avgvol.columns.intersection(ss.columns)
            dtc = (ss[common] / avgvol[common].where(avgvol[common] > 0)).astype("float32")
            _write(dtc, "days_to_cover", base)
            out["days_to_cover"] = [int(dtc.shape[0]), int(dtc.shape[1])]

    srl = hf.sector_short_long(mg.load_short_ratio(), cal)     # 業種別空売り +1 営業日
    sec_map = _load_sector()
    if not srl.empty and not sec_map.empty:
        sec_wide = fu.point_in_time(srl, month_ends, ["sector_short_ratio"],
                                    date_col="available_date", code_col="S33",
                                    lag_days=0)["sector_short_ratio"]
        codes = [str(c) for c in adj.columns]
        sec_for_code = sec_map.reindex(codes)                  # Code→S33（業種スナップ・docs明記）
        bcast = sec_wide.reindex(columns=sec_for_code.values)
        bcast.columns = codes
        _write(bcast.astype("float32"), "sector_short_ratio", base)
        out["sector_short_ratio"] = [int(bcast.shape[0]), int(bcast.shape[1])]
    return out


def _apply_cols(func, *wides: pd.DataFrame) -> pd.DataFrame:
    """Series 関数を共通列に列方向適用（多入力対応）。微細構造の列ループ用。"""
    cols = wides[0].columns
    for w in wides[1:]:
        cols = cols.intersection(w.columns)
    data = {c: func(*[w[c] for w in wides]) for c in cols}
    return pd.DataFrame(data, index=wides[0].index)


def build_microstructure_features(base: str = "data") -> dict:
    """微細構造（Parkinson/Garman-Klass/Roll/Corwin-Schultz/VPIN/RSI）を月次PIT材化（GKX A-2）。

    `microstructure.py` の Series 関数を**全銘柄に適用**（dtype 非依存の関数は wide 直接、roll/vpin は
    列ループ）→ 月末 as-of・float32。**入力は調整済 OHLC**（adj_high/low/open/close）＝分割日の
    ジャンプでスプレッド/レンジ推定が壊れない。volume は adj 版が無いため raw を使用（docs/16 明記）。
    符号：parkinson/garman_klass は**負号**（低ボラ）、roll/corwin は**正号**（非流動性）、vpin/rsi は raw。
    amihud は price_factors で材化済みのため追加しない。
    """
    from ..features import microstructure as ms
    ah, al = load_wide("adj_high", base=base), load_wide("adj_low", base=base)
    ao, ac = load_wide("adj_open", base=base), load_wide("adj_close", base=base)
    vo = load_wide("volume", base=base)
    if ac.empty or ah.empty or al.empty:
        return {}
    daily = {
        "parkinson_vol": -ms.parkinson_vol(ah, al, 20),               # 低ボラ＝負号
        "garman_klass_vol": -ms.garman_klass_vol(ao, ah, al, ac, 20),
        "corwin_schultz_spread": ms.corwin_schultz_spread(ah, al),    # 非流動性＝正号
        "rsi": ms.rsi(ac, 14),                                        # raw [0,100]
        "roll_spread": _apply_cols(lambda s: ms.roll_spread(s, 20), ac),   # 正号
    }
    if not vo.empty:
        daily["vpin"] = _apply_cols(lambda c, v: ms.vpin(c, v, 50), ac, vo)  # raw [0,1]
    out: dict[str, list] = {}
    for name, df in daily.items():
        m = df.resample("ME").last().astype("float32")
        _write(m, name, base)
        out[name] = [int(m.shape[0]), int(m.shape[1])]
    return out


def build_fundamental_features_v2(base: str = "data") -> dict:
    """fins_summary 新ファンダ（SUE/予想改訂/成長/安定度/持続可能成長/52週高値/季節性/Dimsonβ）
    を月次PIT材化（float32・wide・GKX B・EDINET非依存）。

    開示レベル特徴は `fundamentals.point_in_time`（DiscDate アンカー・lag1）で月末 as-of。SUE と
    予想改訂は raw サプライズを**月末終値で除して** surprise yield 化（赤字でも頑健・winsor は
    zscore/rank ビューが担う）。52週高値/Dimsonβは日次→月末、季節性は月次。
    """
    from ..equities import fundamental_factors as ff
    from ..equities import fundamentals as fu
    adj = load_wide("adj_close", base=base)
    raw_close = load_wide("close", base=base)
    if adj.empty or raw_close.empty:
        return {}
    month_ends = list(adj.resample("ME").last().index)
    close_me = raw_close.resample("ME").last().reindex(index=month_ends)
    cmask = close_me.where(close_me > 0)
    out: dict[str, list] = {}

    def emit(df: pd.DataFrame, name: str) -> None:
        m = df.reindex(index=month_ends).astype("float32")
        _write(m, name, base)
        out[name] = [int(m.shape[0]), int(m.shape[1])]

    out_fy, out_rev = ff.disclosure_features(fu.load_fundamentals())
    fy_flds = ["sue_recent_raw", "sue_initial_raw", "sales_growth", "profit_growth",
               "equity_growth", "roe_stability", "margin_stability", "sustainable_growth"]
    pfy = fu.point_in_time(out_fy, month_ends, fy_flds, date_col="DiscDate",
                           code_col="Code", lag_days=1)
    prev = fu.point_in_time(out_rev, month_ends, ["forecast_revision_raw"],
                            date_col="DiscDate", code_col="Code", lag_days=1)
    emit(pfy["sue_recent_raw"] / cmask, "sue_recent")        # surprise yield
    emit(pfy["sue_initial_raw"] / cmask, "sue_initial")
    emit(prev["forecast_revision_raw"] / cmask, "forecast_revision")
    for f in ("sales_growth", "profit_growth", "equity_growth", "roe_stability",
              "margin_stability", "sustainable_growth"):
        emit(pfy[f], f)
    emit(ff.high_52w(adj).resample("ME").last(), "high_52w")
    emit(ff.seasonality(adj), "seasonality")
    emit(ff.dimson_beta(adj).resample("ME").last(), "dimson_beta")
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
    # 拡充 A/B は完全な Silver OHLCV が揃う base でのみ材化（最小フィクスチャでは no-op）。
    full = all(not load_wide(f, base=base).empty
               for f in ("close", "adj_high", "adj_low", "volume"))
    if full:
        rep["holdings"] = build_holdings_features(base=base)
        rep["microstructure"] = build_microstructure_features(base=base)
        rep["fundamental_v2"] = build_fundamental_features_v2(base=base)
    return rep
