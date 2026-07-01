"""daily_regime: 2つのPIT系（週足 walk-forward 系 / AsOf 系）の橋渡し（釘②）。

レジーム/アンカー/ティアの PIT 系列を AsOfView 消費可能な日足パネルへ変換し、シーム契約の
不変条件を**実行時アサーション**で強制する（破れば LookAheadError）。AsOfView は asof(t) で
`.loc[:t]` スライスするため、登録後は構造的に ≤t しか見えない。smoothed は登録禁止。
"""
from __future__ import annotations

import pandas as pd

from invest_system.research.data_view import AsOfView

from .contracts import LookAheadError, validate_anchor


def assert_no_future_rows(panel: pd.DataFrame, asof) -> None:
    """パネルが asof 以前の行しか持たないことをアサート。"""
    asof = pd.Timestamp(asof)
    if len(panel) and pd.Timestamp(panel.index.max()) > asof:
        raise LookAheadError(f"パネルに未来行 {panel.index.max()} > asof {asof}")


def _canonical_daily(
    anchor: pd.DataFrame, daily_index, value_col: str, anchor_id: str
) -> pd.Series:
    """契約1 → 日足系列：**available_from** で as-of 結合（直前完了週の値のみ可視）。"""
    a = anchor[anchor["anchor_id"].astype(str) == str(anchor_id)]
    src = pd.DataFrame({
        "available_from": pd.to_datetime(a["available_from"].values),
        "v": a[value_col].astype(float).values,
    }).sort_values("available_from")
    daily = pd.DataFrame(
        {"d": pd.to_datetime(pd.DatetimeIndex(daily_index))}
    ).sort_values("d")
    out = pd.merge_asof(
        daily, src, left_on="d", right_on="available_from", direction="backward"
    )
    return pd.Series(out["v"].values, index=pd.DatetimeIndex(out["d"].values), name=value_col)


def anchor_daily_panel(
    anchor: pd.DataFrame, daily_index, value_col: str, anchor_id: str
) -> pd.Series:
    """契約1 の正準ビルダ：各日 d に available_from ≤ d の最新アンカー値を割り当てる。

    week_end_date でなく available_from を可視開始にすることで §1-4 ラグを構造的に保証する。
    """
    validate_anchor(anchor)
    return _canonical_daily(anchor, daily_index, value_col, anchor_id)


def assert_anchor_daily_pit(
    anchor: pd.DataFrame, daily_series: pd.Series, value_col: str, anchor_id: str
) -> None:
    """日足アンカーが available_from ラグ仕様に一致することをアサート（破れば LookAheadError）。

    仕様（available_from 結合）を独立に再計算して照合する。week_end_date 漏れ（公表ラグ無視）の
    リーク版はラグ日でズレるため、ここで赤になる。
    """
    expected = _canonical_daily(anchor, daily_series.index, value_col, anchor_id)
    g = daily_series.reindex(expected.index)
    both_nan = expected.isna() & g.isna()
    eq = (expected == g) | both_nan
    if not bool(eq.all()):
        first = eq.index[~eq.to_numpy()][0]
        raise LookAheadError(
            f"アンカー日足が available_from ラグ仕様と不一致"
            f"（week_end_date 漏れの疑い）@ {first}"
        )


def build_regime_view(
    base_panels: dict[str, pd.DataFrame], regime_frames: dict[str, pd.DataFrame]
) -> AsOfView:
    """既存価格パネル＋レジーム/アンカー/ティア日足パネルを AsOfView に登録。

    すべて ≤t で消費される（AsOfView.asof は `.loc[:d]`）。`*_smoothed` は評価系から
    遮断するため登録禁止（§2.1 filtered/smoothed 分離）。
    """
    panels = dict(base_panels)
    for name, frame in regime_frames.items():
        if name.endswith("_smoothed"):
            raise LookAheadError(f"smoothed フレームは登録禁止（評価系から遮断）: {name}")
        panels[name] = frame
    return AsOfView(panels)
