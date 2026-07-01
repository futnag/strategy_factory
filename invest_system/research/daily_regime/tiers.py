"""daily_regime: 流動性ティア PIT 所属（契約2の生成・接続設計 §2.2）。

二段（Amihud 主・ADV 副ゲート）：
  ① ADV 床未満（執行困難）→ gate_illiquid に隔離。
  ② 生存銘柄を**流動性水準**の横断分位で T1..Tn（T1=最流動 … Tn=最非流動）。

帰属キーは「当日生Amihud」でなく**トレーリング中央値**（平滑化した流動性水準）。生Amihud は
ストレス時に跳ねるため、生値で括ると平常流動な銘柄がストレス当日に非流動ティアへ誤再ランクされ、
プールが構造的流動性ピアでなくなる／filtered 確率が跳ね sticky/min-dwell を損なう／帰属が検知対象
ストレスと共動する（統計的交絡）。中央値で水準化し、**refit 時のみ再計算・refit 間は固定**＝
スティッキーなプールにする（プールは安定、HMM が速い一過性ストレスを検知する分業）。これは
先読みでなく交絡なので leak_tests は捕まえない（設計レベルの修正）。

PIT：中央値・ADV はいずれも trailing（≤t）。横断分位は refit 日 r の同時点断面のみ（≤r）。
所属は date×code パネルとして各時点の as-of ラベルを保持し、pooling は tier_pool_index で
「過去点 s は s の as-of ティアで」帰属させる（遡及再付番禁止）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import GATE_TIER, LookAheadError, validate_tiers


def _daily_amihud(adj_close: pd.DataFrame, turnover: pd.DataFrame) -> pd.DataFrame:
    """日次 Amihud 比率 |ret| / 売買代金（窓平均なし・生）。"""
    ret = adj_close.pct_change(fill_method=None).abs()
    return ret / turnover.where(turnover > 0)


def assign_tiers(
    adj_close: pd.DataFrame,
    turnover: pd.DataFrame,
    *,
    n_tiers: int = 3,
    adv_floor: float = 5e7,
    smooth_window: int = 60,
    refit_every: int = 20,
    level_kind: str = "amihud_median",
) -> pd.DataFrame:
    """平滑化＋スティッキーな PIT ティア所属（long: date,code,tier,assigned_through）。

    帰属はトレーリング中央値（流動性水準）の横断分位で、**refit 日のみ**算出し refit 間は固定。
    """
    adj_close, turnover = adj_close.align(turnover, join="inner")
    mp = max(2, int(smooth_window * 0.8))
    if level_kind == "amihud_median":                 # 低水準＝流動 → 昇順ランクが T1
        level = _daily_amihud(adj_close, turnover).rolling(smooth_window, min_periods=mp).median()
        ascending = True
    elif level_kind == "turnover_median":             # 高水準＝流動 → 降順ランクが T1
        level = turnover.rolling(smooth_window, min_periods=mp).median()
        ascending = False
    else:
        raise ValueError(f"未知の level_kind: {level_kind}")
    smooth_adv = turnover.rolling(smooth_window, min_periods=mp).median()

    idx = adj_close.index
    refit_dates = idx[::refit_every]
    assign = {}
    edges = np.linspace(0.0, 1.0, n_tiers + 1)
    for r in refit_dates:
        lvl_r, adv_r = level.loc[r], smooth_adv.loc[r]
        gated_in = adv_r >= adv_floor                 # ADV 床（平滑・≤r）
        elig = lvl_r.where(gated_in & lvl_r.notna())
        rank = elig.rank(pct=True, ascending=ascending)
        tier = pd.Series(index=lvl_r.index, dtype=object)
        for i in range(n_tiers):
            lo, hi = edges[i], edges[i + 1]
            m = (rank >= lo) & (rank <= hi) if i == 0 else (rank > lo) & (rank <= hi)
            tier[m] = f"T{i + 1}"
        tier[((~gated_in) & adv_r.notna())] = GATE_TIER
        assign[r] = tier

    refit_df = pd.DataFrame(assign).T
    refit_df.index = pd.DatetimeIndex(refit_df.index)
    refit_df = refit_df.sort_index()
    daily_mem = refit_df.reindex(idx, method="ffill")  # refit 間は固定（sticky）

    refit_index = pd.DatetimeIndex(refit_dates)
    pos = np.clip(refit_index.searchsorted(idx, side="right") - 1, 0, len(refit_index) - 1)
    gov = pd.Series(refit_index[pos], index=idx)        # 各日を支配する refit 日（≤date）

    long = daily_mem.stack(future_stack=True).dropna().rename("tier").reset_index()
    long.columns = ["date", "code", "tier"]
    long["code"] = long["code"].astype(str)
    long["assigned_through"] = long["date"].map(gov)
    return validate_tiers(long, n_tiers=n_tiers)


def membership_panel(long: pd.DataFrame) -> pd.DataFrame:
    """long → date×code の tier パネル（各時点の as-of ラベル保持）。"""
    return long.pivot(index="date", columns="code", values="tier")


def refit_turnover(membership: pd.DataFrame) -> pd.Series:
    """refit 境界ごとのティア入替率（移動した銘柄割合）を監査出力。

    スティッキーなので変化は refit 境界でのみ起きる。入替率が高ければ帰属が速すぎる兆候。
    """
    prev = membership.shift(1)
    changed = (membership != prev) & membership.notna() & prev.notna()
    denom = (membership.notna() & prev.notna()).sum(axis=1)
    frac = changed.sum(axis=1) / denom.where(denom > 0)
    return frac[frac.fillna(0.0) > 0.0]


def tier_pool_index(membership: pd.DataFrame, tier: str, asof) -> pd.MultiIndex:
    """ティア tier に **as-of 各時点で**属する (date,code) を ≤asof で返す（pooling が消費）。

    遡及再付番禁止：過去点 s は membership.loc[s] の as-of ラベルで判定する
    （membership.loc[asof] を過去へ適用しない）。
    """
    asof = pd.Timestamp(asof)
    sub = membership.loc[:asof]
    stacked = (sub == tier).stack()
    return stacked[stacked].index


def assert_pool_index_pit(membership: pd.DataFrame, tier: str, asof, index) -> None:
    """pool index が PIT・as-of 帰属であることを実行時アサート（破れば LookAheadError）。"""
    asof = pd.Timestamp(asof)
    for d, c in index:
        d = pd.Timestamp(d)
        if d > asof:
            raise LookAheadError(f"pool index に未来点 ({d} > asof {asof})")
        if membership.loc[d, c] != tier:
            raise LookAheadError(
                f"遡及再付番: ({d},{c}) の as-of ティア={membership.loc[d, c]} ≠ {tier}"
            )
