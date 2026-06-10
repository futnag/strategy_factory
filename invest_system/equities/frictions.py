"""日本市場の執行フリクション（値幅制限・空売り可否）— L1/L5 の現実性（DP15）。

米国株の教科書に無い日本固有の執行制約を、バックテストの損益計算へ反映するための
純関数群（ネットワーク不要・テスト可能）。エンジン側の控除は `research/engine.py`
（no_buy/no_sell・short_borrow_bps）が担い、本モジュールはフラグ/マスクの構築を担う。

値幅制限（ストップ高/安）：J-Quants 日次の UL/LL は「当日ストップ高/安を記録したか」の
0/1 フラグ。引け執行のバックテストでは「**引けが制限値幅に張り付いたまま終えた日**」のみ
執行不能とみなす（日中に制限へ触れても引けで剥がれていれば引け執行は可能）：
  no_buy[t,c]  = UL==1 かつ close>=high … ストップ高引け＝買い越し注文は約定しない
  no_sell[t,c] = LL==1 かつ close<=low  … ストップ安引け＝売り越し注文は約定しない
  出来高 0（終日約定なし）→ 両側不能。
売りはストップ高でも約定可（買い需要超過）・買いはストップ安でも約定可、という
板の非対称をそのまま符号化している。close 欠損はブロックしない（上場廃止/未上場と
一時停止を区別できないため、退出はエンジン既存の処理＝目標ウェイト消滅に委ねる）。

空売り可否（貸借銘柄）：制度信用の売建は**貸借銘柄のみ**可能。週次信用残高
（margin_weekly）の IssType（1=信用銘柄, 2=貸借銘柄, 3=その他）を PIT 整合
（公表ラグ込み・Date≤t−lag のみ参照）した bool マスクを返す。一般信用は証券会社
依存のためここでは扱わない（保守側＝制度で売れる銘柄のみ True）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def limit_lock_flags(close: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame,
                     upper_flag: pd.DataFrame, lower_flag: pd.DataFrame,
                     volume: pd.DataFrame | None = None
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """値幅制限の「引け張り付き」判定 → (no_buy, no_sell) の bool パネル。

    入力はいずれも wide（index=日付, col=Code）・**無調整**の O/H/L/C と UL/LL フラグ
    （`store.load_wide("close"/"high"/"low"/"upper_limit"/"lower_limit")` または
    月次スナップショットの `assemble_panel(snaps, "C"/"H"/"L"/"UL"/"LL")`）。
    調整は同日比較のため不要。volume を与えると出来高 0（終日約定なし）も両側不能に含める。
    NaN は False（情報なし＝ブロックしない。close 欠損は no-trade として両側 True）。
    """
    ub = upper_flag.reindex_like(close).fillna(0).astype(float) > 0
    lb = lower_flag.reindex_like(close).fillna(0).astype(float) > 0
    hi = high.reindex_like(close)
    lo = low.reindex_like(close)
    pinned_up = ub & close.notna() & hi.notna() & (close >= hi)
    pinned_dn = lb & close.notna() & lo.notna() & (close <= lo)
    no_trade = pd.DataFrame(False, index=close.index, columns=close.columns)
    if volume is not None:
        vo = volume.reindex_like(close)
        no_trade = vo == 0           # NaN==0 は False＝情報なしはブロックしない
    no_buy = (pinned_up | no_trade).astype(bool)
    no_sell = (pinned_dn | no_trade).astype(bool)
    return no_buy, no_sell


def shortable_mask(margin_weekly: pd.DataFrame, dates,
                   *, lag_days: int = 4, tolerance_days: int = 60,
                   code_col: str = "Code", date_col: str = "Date"
                   ) -> pd.DataFrame:
    """貸借銘柄（制度信用で売建可能）の PIT マスク（index=dates, col=Code, bool）。

    margin_weekly は `margin.load_weekly_margin()` の long（Date, Code, IssType, ...）。
    各日 t では「**t−lag_days 以前**の週次レコード」のみ参照する（週末（金曜）申込日
    基準の残高は通常翌週第2営業日（火曜）公表＝既定 lag_days=4 で公表ラグを保守的に
    吸収）。**銘柄ごとに**直近レコードを LOCF し、tolerance_days より古い銘柄は False
    （上場廃止・貸借区分喪失を自然に失効）。判定は IssType==2（貸借銘柄）。IssType
    欠損行は制度売残 ShrtStdVol>0 で代替（売残が立っている＝制度で売れた実績）。
    """
    dates = pd.DatetimeIndex(dates)
    cols = [code_col, date_col]
    if margin_weekly.empty or not set(cols).issubset(margin_weekly.columns):
        return pd.DataFrame(False, index=dates, columns=[])
    df = margin_weekly[[c for c in (*cols, "IssType", "ShrtStdVol")
                        if c in margin_weekly.columns]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df[code_col] = df[code_col].astype(str)
    if "IssType" in df.columns:
        ok = pd.to_numeric(df["IssType"], errors="coerce") == 2
        if "ShrtStdVol" in df.columns:          # IssType 欠損行のみ代替判定
            alt = pd.to_numeric(df["ShrtStdVol"], errors="coerce") > 0
            ok = ok.where(df["IssType"].notna(), alt)
    else:
        ok = pd.to_numeric(df["ShrtStdVol"], errors="coerce") > 0
    df["_shortable"] = ok.astype(float)
    df = df.sort_values(date_col).drop_duplicates([date_col, code_col], keep="last")
    wide = df.pivot(index=date_col, columns=code_col, values="_shortable")
    # 銘柄ごとの LOCF と最終観測日（行レベルの ffill では「他銘柄だけ更新された週」に
    # 自銘柄の直近レコードが落ちるため、列単位で as-of する）
    vals = wide.ffill()
    obs = pd.DataFrame(
        np.where(wide.notna().to_numpy(),
                 wide.index.to_numpy()[:, None], np.datetime64("NaT")),
        index=wide.index, columns=wide.columns).ffill()
    lookup = dates - pd.Timedelta(days=lag_days)          # ≤ t−lag のみ＝先読みなし
    vals_at = vals.reindex(lookup, method="ffill")
    obs_at = obs.reindex(lookup, method="ffill")
    age = pd.DataFrame(lookup.to_numpy()[:, None] - obs_at.to_numpy(),
                       index=dates, columns=wide.columns)
    fresh = age <= pd.Timedelta(days=tolerance_days)      # NaT 比較は False
    vals_at.index = dates
    return ((vals_at > 0) & fresh).astype(bool)


def short_notional_coverage(weights: pd.Series, shortable_row: pd.Series) -> float:
    """ある時点のウェイトのうち、ショート想定元本が貸借銘柄で占める割合（診断用）。

    weights: 戦略の目標ウェイト（負=ショート）。shortable_row: 同時点の shortable_mask 行。
    ショートが無い時点は 1.0（制約に抵触しない）。
    """
    short = weights[weights < 0]
    if short.empty:
        return 1.0
    ok = shortable_row.reindex(short.index).fillna(False).astype(bool)
    return float(short[ok].abs().sum() / short.abs().sum())
