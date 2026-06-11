"""クロスセクション・ロングショート・バックテスト（月次・ダラーニュートラル）。

各リバランス日 t でファクター値の上位/下位 quantile を等加重でロング/ショートし、
t→t+1 のフォワードリターンを実現させる。取引コストはターゲット比率の変化（売買
回転）に基づき控除する（正直なネット評価）。純関数でネットワーク不要。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _target_weights(factor_row: pd.Series, quantile: float,
                    min_names: int):
    """1日断面のロングショート目標比率（long合計+1, short合計-1, グロス2）。

    選定は**ファクター値のみ**で行う。「翌期リターンが存在すること」を条件に
    入れると、形成時点で翌期の上場廃止・取引停止を知っている先読み（生存者
    バイアス＝悪い銘柄がロング側から消える上方バイアス）になるため使わない。
    """
    fr = factor_row.dropna()
    if len(fr) < max(min_names, 2):
        return None
    k = max(1, int(len(fr) * quantile))
    order = fr.sort_values()
    short = order.index[:k]      # ファクター低位＝期待プレミアムの逆 → ショート
    longs = order.index[-k:]     # ファクター高位 → ロング
    w = pd.Series(0.0, index=fr.index)
    w[longs] = 1.0 / len(longs)
    w[short] = -1.0 / len(short)
    return w


def long_short_returns(factor: pd.DataFrame, fwd_ret: pd.DataFrame,
                       quantile: float = 0.2, costs_bps: float = 0.0,
                       min_names: int = 20) -> pd.Series:
    """月次ロングショート・ネットリターン系列を返す（index=建玉日 t）。

    factor   : wide（index=日付, columns=Code）。符号は「高い=ロング」前提。
    fwd_ret  : wide。行 t は t→t+1 の実現リターン（forward_returns の出力）。
    quantile : 各サイドの分位（0.2=上下20%）。
    costs_bps: 片道売買コスト[bps]。回転 sum|Δw| に対して課金（両サイド・両端）。

    選定後にリターンが欠損する銘柄（翌期の上場廃止・取引停止等）は当期損益への
    寄与 0 ＝「退出を価格不変で清算する」近似（真の退出リターンは本パネルでは
    観測不能）。選定から除外はしない（_target_weights の先読み排除を参照）。
    """
    idx = factor.index.intersection(fwd_ret.index)
    prev_w: pd.Series | None = None
    rows = []
    for dt in idx:
        w = _target_weights(factor.loc[dt], quantile, min_names)
        if w is None:
            rows.append((dt, np.nan))
            prev_w = None
            continue
        gross = float((w * fwd_ret.loc[dt].reindex(w.index)).sum())
        # 取引コスト：前回比の回転（sum|Δw|）に片道bpsを乗じる
        if costs_bps:
            all_names = w.index if prev_w is None else w.index.union(prev_w.index)
            cur = w.reindex(all_names).fillna(0.0)
            pre = (prev_w.reindex(all_names).fillna(0.0) if prev_w is not None
                   else pd.Series(0.0, index=all_names))
            turnover = float((cur - pre).abs().sum())
            cost = (costs_bps / 1e4) * turnover
        else:
            cost = 0.0
        rows.append((dt, gross - cost))
        prev_w = w
    return pd.Series({d: v for d, v in rows}).sort_index()
