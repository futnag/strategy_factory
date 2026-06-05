"""財務サマリーのポイントインタイム整合（先読みバイアスの排除）。

各リバランス日 t に対し、各銘柄で「開示日 DiscDate ≤ t − lag_days」を満たす
最新の開示値のみを採用する。決算は場中(14:00等)に開示され得るため lag_days≥1 を
既定とし、開示当日の価格には反映させない保守的設計とする。

これは因果推論（pillar C）以前の最重要規律：未来情報の混入を断つ。
"""
from __future__ import annotations

import pandas as pd


def point_in_time(fund_long: pd.DataFrame, rebal_dates, fields: list[str],
                  date_col: str = "DiscDate", code_col: str = "Code",
                  lag_days: int = 1) -> dict[str, pd.DataFrame]:
    """銘柄×開示の長形式 → フィールド別の as-of wide パネル。

    fund_long  : 行=開示, 列に date_col, code_col, 各 field を含む
    rebal_dates: リバランス日の列（Timestamp 群）
    返り値      : {field: DataFrame(index=rebal_dates, columns=code)}（as-of値）
    """
    rebal = pd.DatetimeIndex(sorted(pd.to_datetime(list(rebal_dates)))).normalize()
    present = [f for f in fields if f in fund_long.columns]
    if fund_long.empty or not present:
        return {f: pd.DataFrame(index=rebal, dtype="float64") for f in present}

    df = fund_long.dropna(subset=[date_col]).copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    # as-of 突合のための左キー（リバランス日からラグを引いた締切日）
    left = pd.DataFrame({"asof": rebal})
    left["cutoff"] = left["asof"] - pd.Timedelta(days=lag_days)
    left = left.sort_values("cutoff")

    # 列を1本ずつ挿入するとDataFrameが断片化する（pandas警告＋低速）。
    # フィールド別に {code: series} を貯め、最後に一括構築する。
    acc: dict[str, dict[str, pd.Series]] = {f: {} for f in present}
    for code, g in df.groupby(code_col):
        g = g.sort_values(date_col)
        # 同一開示日が複数なら最後（訂正等）を採用
        g = g[~g[date_col].duplicated(keep="last")]
        m = pd.merge_asof(left, g, left_on="cutoff", right_on=date_col,
                          direction="backward").set_index("asof")
        for f in present:
            if f in m.columns:
                acc[f][str(code)] = pd.to_numeric(m[f], errors="coerce").reindex(rebal)
    return {f: (pd.DataFrame(acc[f], index=rebal) if acc[f]
               else pd.DataFrame(index=rebal, dtype="float64")) for f in present}
