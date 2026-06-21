"""Phase 4 評価ハーネス（**開発中は throwaway 診断**・K 不変）。

handoff §5,§7：OOS R²（GKX 流）・月次 IC/rank-IC・分位 L/S スプレッド・回転率・手数料後・
廃止封筒・線形 vs ML 増分。**判定（DSR・レジストリ登録）はここではしない**＝探索診断のみ
（judge_grid/registry を触らない＝K 不変）。判定は run_phase4_judgment.py が一度だけ行う。

L/S は OOS 予測の月次クロスセクション上位/下位デシルの等加重スプレッド。y は超過トータル
リターン（廃止補完済）。長−短で断面平均は相殺されるので超過/生は同値。手数料は回転率×bps。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..validation.dsr import sharpe_ratio


def oos_r2(oos_pred: pd.Series, y: pd.Series) -> float:
    """GKX 流 OOS R² = 1 − Σ(y−ŷ)² / Σ y²（ゼロ基準＝個別超過リターン予測の説明力）。"""
    df = pd.concat([oos_pred.rename("p"), y.rename("y")], axis=1).dropna()
    if df.empty or float((df["y"] ** 2).sum()) == 0.0:
        return float("nan")
    sse = float(((df["y"] - df["p"]) ** 2).sum())
    sst = float((df["y"] ** 2).sum())
    return 1.0 - sse / sst


def monthly_ic(oos_pred: pd.Series, y: pd.Series, method: str = "spearman") -> dict:
    """月次クロスセクション IC（既定 rank-IC）。mean IC・IR・有効月数を返す。"""
    df = pd.concat([oos_pred.rename("p"), y.rename("y")], axis=1).dropna()
    ics = []
    for _, g in df.groupby(level="date"):
        if len(g) >= 10:
            ic = g["p"].corr(g["y"], method=method)
            if pd.notna(ic):
                ics.append(ic)
    if not ics:
        return {"mean_ic": float("nan"), "ir": float("nan"), "n_months": 0}
    arr = np.array(ics)
    ir = float(arr.mean() / arr.std(ddof=1)) if len(arr) > 1 and arr.std(ddof=1) > 0 else float("nan")
    return {"mean_ic": float(arr.mean()), "ir": ir, "n_months": len(arr)}


def ls_portfolio_returns(oos_pred: pd.Series, y: pd.Series, decile: float = 0.1,
                         costs_bps: float = 15.0, min_names: int = 20
                         ) -> pd.DataFrame:
    """月次 L/S（上位/下位デシル等加重）。返り値 index=month, cols=[gross,net,turnover,n]。

    各月：ŷ で銘柄を並べ上位 decile をロング(+1/nL)・下位 decile をショート(−1/nS)。
    gross=Σ w·y、turnover=Σ|Δw|（前月との差）、net=gross − turnover×bps。手数料は片道 bps。
    """
    df = pd.concat([oos_pred.rename("p"), y.rename("y")], axis=1).dropna()
    rows = []
    prev = pd.Series(dtype="float64")
    for dt, g in df.groupby(level="date"):
        g = g.droplevel("date")
        if len(g) < max(min_names, 10):
            continue
        k = max(1, int(len(g) * decile))
        order = g["p"].sort_values()
        short, long = order.index[:k], order.index[-k:]
        w = pd.Series(0.0, index=g.index)
        w[long] = 1.0 / len(long)
        w[short] = -1.0 / len(short)
        gross = float((w * g["y"]).sum())
        names = w.index.union(prev.index)
        turn = float((w.reindex(names).fillna(0.0) - prev.reindex(names).fillna(0.0)).abs().sum())
        net = gross - turn * costs_bps / 1e4
        rows.append((dt, gross, net, turn, int((w != 0).sum())))
        prev = w[w != 0]
    return pd.DataFrame(rows, columns=["date", "gross", "net", "turnover", "n"]
                        ).set_index("date")


@dataclass
class FamilyEval:
    family: str
    oos_r2: float
    mean_ic: float
    ir: float
    sharpe_net: float           # per-period（月次）L/S ネット Sharpe
    sharpe_gross: float
    ann_sharpe_net: float       # 年率（×√12）表示用
    turnover: float             # 平均月次回転率
    n_months: int


def evaluate_family(family: str, oos_pred: pd.Series, y: pd.Series,
                    decile: float = 0.1, costs_bps: float = 15.0) -> FamilyEval:
    """1 推定器の OOS 診断束（R²・IC・L/S ネット Sharpe・回転率）。throwaway。"""
    r2 = oos_r2(oos_pred, y)
    icd = monthly_ic(oos_pred, y)
    port = ls_portfolio_returns(oos_pred, y, decile, costs_bps)
    net, gross = port["net"], port["gross"]
    sn = sharpe_ratio(net) if len(net) >= 2 and net.std(ddof=1) > 0 else float("nan")
    sg = sharpe_ratio(gross) if len(gross) >= 2 and gross.std(ddof=1) > 0 else float("nan")
    return FamilyEval(family=family, oos_r2=r2, mean_ic=icd["mean_ic"], ir=icd["ir"],
                      sharpe_net=sn, sharpe_gross=sg,
                      ann_sharpe_net=sn * np.sqrt(12) if pd.notna(sn) else float("nan"),
                      turnover=float(port["turnover"].mean()) if len(port) else float("nan"),
                      n_months=len(port))


def linear_vs_ml_increment(evals: dict[str, FamilyEval]) -> dict:
    """線形ベースライン最良 vs ML 最良の増分（OOS R²・ネット Sharpe）。ML が線形を超えるか。"""
    lin = [e for k, e in evals.items() if k in ("ols", "lasso", "elasticnet")]
    ml = [e for k, e in evals.items() if k in ("rf", "hgbrt", "mlp")]
    best = lambda xs, a: max((getattr(e, a) for e in xs if pd.notna(getattr(e, a))),
                             default=float("nan"))
    return {"linear_best_r2": best(lin, "oos_r2"), "ml_best_r2": best(ml, "oos_r2"),
            "linear_best_sharpe": best(lin, "sharpe_net"),
            "ml_best_sharpe": best(ml, "sharpe_net"),
            "ml_beats_linear_r2": best(ml, "oos_r2") > best(lin, "oos_r2"),
            "ml_beats_linear_sharpe": best(ml, "sharpe_net") > best(lin, "sharpe_net")}
