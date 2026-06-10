"""Phase 2 ペーパー台帳と照合の会計部品（純関数・docs/02 D5）。

設計：照合は**ステートレス**＝毎回 intended/orders（generate が出力）から約定を
シミュレートして equity curve を再構成する（可変の台帳状態を持たない＝壊れない）。
実弾（Phase 2b）では `fills_actual_<月>.csv` が存在すればその約定価格を優先し、
ペーパー（T+1始値）との差がそのまま実測スリッページになる。

会計は**円建て・リターンベース**（株数×価格でなく、投下円×調整後リターン）で行う。
株式分割が保有中に起きても adj 系列の相対リターンは正しいため、分割で株数が変わる
実務との差は照合誤差として現れない。キルスイッチ水準（D5 事前登録）も本モジュールが
単一の正とする。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# D5 事前登録のキルスイッチ水準（合成DD・バックテスト分布から導出）
ALERT_DD = -0.08      # 警報
DERISK_DD = -0.12     # デリスク50%
STOP_DD = -0.15       # 全停止＋ポストモーテム


def next_open_fills(keys, after: pd.Timestamp, open_panel: pd.DataFrame,
                    max_days: int = 5) -> pd.DataFrame:
    """決定日 after の翌取引日以降、最初に始値が存在する日で約定（T+1始値・DP17）。

    open_panel: 日次始値 wide（調整後を推奨）。各 key について after より**後**の
    最初の非欠損始値を max_days 行以内で探す（祝日・売買停止の繰延）。見つからない
    場合は NaN（未約定として照合レポートに現れる）。
    Returns: DataFrame[key, fill_date, fill_price]。
    """
    idx = open_panel.index
    pos0 = int(idx.searchsorted(pd.Timestamp(after), side="right"))
    rows = []
    for k in keys:
        fd, fp = pd.NaT, np.nan
        if k in open_panel.columns:
            seg = open_panel[k].iloc[pos0:pos0 + max_days]
            valid = seg.dropna()
            if len(valid):
                fd, fp = valid.index[0], float(valid.iloc[0])
        rows.append((k, fd, fp))
    return pd.DataFrame(rows, columns=["key", "fill_date", "fill_price"])


def yen_positions_pnl(notional: pd.Series, rel_returns: pd.Series) -> float:
    """円建てポジション（符号付き想定元本）×期間相対リターン → 円損益。

    rel_returns が欠損の銘柄は損益 0（未約定/上場廃止の保守処理＝照合で可視化）。
    """
    r = rel_returns.reindex(notional.index)
    return float((notional * r).fillna(0.0).sum())


def drawdown_status(returns: pd.Series) -> tuple[pd.Series, float, str]:
    """月次リターン系列 → (DD系列, 現在DD, キルスイッチ判定文字列)。

    DD は**当初元本（1.0）を含むランニングマックス**から測る＝初月の損失も DD として
    数える（運用開始直後に −9% なら ALERT が鳴るべき）。
    """
    r = returns.dropna()
    if r.empty:
        return pd.Series(dtype="float64"), 0.0, "OK"
    cum = (1.0 + r).cumprod()
    runmax = np.maximum.accumulate(np.concatenate([[1.0], cum.to_numpy()]))[1:]
    dd = cum / runmax - 1.0
    cur = float(dd.iloc[-1])
    if cur <= STOP_DD:
        status = f"STOP（全停止 {STOP_DD:.0%} 超過）"
    elif cur <= DERISK_DD:
        status = f"DERISK（50%縮小 {DERISK_DD:.0%} 超過）"
    elif cur <= ALERT_DD:
        status = f"ALERT（警報 {ALERT_DD:.0%} 超過）"
    else:
        status = "OK"
    return dd, cur, status


def apply_actual_fills(fills: pd.DataFrame, actual: pd.DataFrame | None
                       ) -> tuple[pd.DataFrame, pd.Series]:
    """実約定 CSV（key, fill_price[, fill_date]）でペーパー約定を上書きし、
    スリッページ（実約定/ペーパー − 1）を返す。actual=None なら上書きなし。
    """
    if actual is None or actual.empty or "key" not in actual.columns:
        return fills, pd.Series(dtype="float64")
    out = fills.set_index("key")
    act = actual.set_index("key")
    both = out.index.intersection(act.index)
    slip = (pd.to_numeric(act.loc[both, "fill_price"], errors="coerce")
            / out.loc[both, "fill_price"] - 1.0).dropna()
    out.loc[both, "fill_price"] = pd.to_numeric(act.loc[both, "fill_price"],
                                                errors="coerce")
    if "fill_date" in act.columns:
        upd = pd.to_datetime(act.loc[both, "fill_date"], errors="coerce")
        out.loc[both, "fill_date"] = upd.where(upd.notna(),
                                               out.loc[both, "fill_date"])
    return out.reset_index(), slip
