"""C1（TOB リスクアーブ）診断ロジック：M&A Online イベント表 × EDINET 当初価格 × 株価。

docs/09 の事前登録ドラフト（scope `tob_arb`）を支える純関数群。M&A Online（2016+・主表）を
スパインに、EDINET tob_deals の当初価格で **`_displayed` リークを根治**する（PIT 地雷 #2）：
- **エントリー信号**＝当初価格 / 公表前日終値（届出時に既知）。`_displayed`（=バンプ後の
  最終価格）は信号に使わない。EDINET 突合できた案件は真の当初価格で上書き、未突合は
  表示価格を当初の代理（バンプ実測 ~5%＝大半で当初=最終なので近似十分・差は flag で監視）。
- **結果リターン**：成立＝最終価格で出口（バンプ後価格を実際に受け取る＝リークでない）、
  不成立＝実終了日近傍の市場価格で出口。

レジーム分離（PIT 地雷 #3）：不成立率は 2020 前後で約 5 倍（docs/09 §2.1）。サブ期間
2016–2019 / 2020–2026 を分けて扱い、単純プールしない。
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


def mark_competing(deals: pd.DataFrame, window_days: int = 180,
                   key: str = "target_code", date_col: str = "announce_date"
                   ) -> pd.Series:
    """同一対象に発表が近接する複数 TOB を競合フラグ化（ソレキア型・docs/09 §2.3）。

    競合案件は「敗者 buyer 価格では損／勝者価格では益」＝別レジームなので scope で分離する。
    """
    out = pd.Series(False, index=deals.index)
    if deals.empty or key not in deals or date_col not in deals:
        return out
    d = deals[[key, date_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    win = pd.Timedelta(days=window_days)
    for k, g in d.dropna().groupby(key):
        if len(g) < 2:
            continue
        if (g[date_col].max() - g[date_col].min()) <= win:
            out.loc[g.index] = True
    return out


def subperiod(year: int) -> Optional[str]:
    """レジーム・サブ期間ラベル（docs/09 §2.1）。2016 未満は検証窓外＝None。"""
    if year < 2016:
        return None
    return "2016-2019" if year <= 2019 else "2020-2026"


def deal_metrics(result: Optional[str], initial_price: Optional[float],
                 final_price: Optional[float], prev_close: Optional[float],
                 entry_open: Optional[float], exit_mkt: Optional[float]) -> dict:
    """1 案件のプレミアム・裁定スプレッド・実現リターン（純計算・PIT 規律順守）。

    premium=当初価格/公表前日終値−1（市場が織り込む前の理論幅）。
    arb_spread=当初価格/T+1始値−1（実際に建てた値からの裁定余地・信号）。
    ret=成立: 最終価格/T+1始値−1 ／ 不成立: 実終了日市場価格/T+1始値−1。
    """
    m = {"premium": None, "arb_spread": None, "ret": None}
    if not initial_price or not entry_open or entry_open <= 0:
        if prev_close and initial_price:
            m["premium"] = initial_price / prev_close - 1.0
        return m
    if prev_close and prev_close > 0:
        m["premium"] = initial_price / prev_close - 1.0
    m["arb_spread"] = initial_price / entry_open - 1.0
    fp = final_price if (final_price and final_price > 0) else initial_price
    if result == "成立":
        m["ret"] = fp / entry_open - 1.0
    elif result == "不成立" and exit_mkt and exit_mkt > 0:
        m["ret"] = exit_mkt / entry_open - 1.0
    return m


def live_weights(schedule: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
    """asof 時点でライブな案件（entry_date <= asof < exit_date）への等加重ウェイト。

    schedule 列：sec（J-Quants コード）・entry_date・exit_date。同時進行 N 件に 1/N、
    残りは現金（合計 <100% になり得る＝キャッシュドラッグを反映・確定事項/設計判断 Q4）。
    判定エンジンの Strategy.target_weights から呼ぶ純関数（PIT 安全＝asof 以前のみ参照）。
    """
    if schedule.empty:
        return pd.Series(dtype="float64")
    live = schedule[(schedule["entry_date"] <= asof) & (asof < schedule["exit_date"])]
    codes = [c for c in live["sec"].dropna().unique()]
    if not codes:
        return pd.Series(dtype="float64")
    w = 1.0 / len(codes)
    return pd.Series({c: w for c in codes}, dtype="float64")


def spread_bucket(arb_spread: Optional[float]) -> Optional[str]:
    """裁定スプレッドのバケット（高スプレッド帯のセレクション診断・docs/09 §3）。"""
    if arb_spread is None or pd.isna(arb_spread):
        return None
    if arb_spread < 0.01:
        return "<1%"
    if arb_spread < 0.03:
        return "1-3%"
    if arb_spread < 0.07:
        return "3-7%"
    return ">=7%"
