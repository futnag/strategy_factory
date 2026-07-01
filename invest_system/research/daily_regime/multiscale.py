"""daily_regime: マルチタイムスケール結合（方式A・事前分布注入）— step 3。

週足（セクター/指数）方向性アンカーの filtered 確率を、pooled 日足検知器の各ステップの状態
**事前**に注入する（接続設計 §5.2・アーキ1）。bear→日足 stressed 事前↑。

⚠ step 3 の解釈枠：週足アンカーは step 5 まで**モメンタム・スタブ**。本 step は「結合機構が
   機械的に正しく動くか（注入が効く・PIT でリーク無し・挙動が妥当）」の**配管検証**であって、
   「マルチスケールがベースラインを超えるか」の中核仮説の証拠ではない（中核判定は step 5 で
   本物の方向性HMM 投入後）。step 2 ベースラインを暫定扱いにしたのと同じ規律。

PIT：tilt は asof_bridge.anchor_daily_panel（available_from ラグ＝直前完了週のみ）経由。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .asof_bridge import anchor_daily_panel
from .pooling import walk_forward_pooled_t_hmm


def weekly_log_tilt(anchor: pd.DataFrame, daily_index: pd.DatetimeIndex, anchor_id: str,
                    *, n_states: int = 2, lam: float = 1.0) -> np.ndarray:
    """週足アンカー（bear/bull filtered）→ 日足 (T,K) の log-tilt（available_from ラグ・PIT）。

    bear>bull（弱気）のとき stressed（最終状態）の事前を lam·(p_bear−p_bull) だけ引き上げ、calm
    （第0状態）を同量引き下げる。中間状態は 0。lam=0 で tilt=0（注入なし＝ベースライン）。
    """
    p_bear = anchor_daily_panel(anchor, daily_index, "p_bear", anchor_id).reindex(daily_index)
    p_bull = anchor_daily_panel(anchor, daily_index, "p_bull", anchor_id).reindex(daily_index)
    diff = (p_bear - p_bull).fillna(0.0).to_numpy(float)        # available_from 前は NaN→0（注入なし）
    tilt = np.zeros((len(daily_index), n_states))
    tilt[:, -1] = lam * diff                                    # stressed 事前↑（弱気時）
    tilt[:, 0] = -lam * diff                                    # calm 事前↓
    return tilt


def walk_forward_multiscale(
    target_code: str,
    feat_panel: dict[str, pd.DataFrame],
    membership: pd.DataFrame,
    anchor: pd.DataFrame,
    anchor_id: str,
    *,
    lam_inject: float = 1.0,
    n_states: int = 2,
    **kwargs,
) -> pd.DataFrame:
    """方式A：pooled 日足検知器に週足アンカーを事前注入した walk-forward（lam_inject=0 で恒等）。"""
    idx = feat_panel[target_code].index
    tilt = weekly_log_tilt(anchor, idx, anchor_id, n_states=n_states, lam=lam_inject)
    return walk_forward_pooled_t_hmm(
        target_code, feat_panel, membership, n_states=n_states, log_tilt=tilt, **kwargs)
