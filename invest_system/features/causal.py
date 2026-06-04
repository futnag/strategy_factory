"""因果フィルタ（L4 / 柱C）：コライダー（合流点）を除去する。

統合ナレッジベース §7 / DP8 の実装。
資料群（「ファクターの蜃気楼」「運用モデル仕様策定指針」）が最も強調する論点：
- 交絡因子（cause, X←Z→Y）は制御すべき／合流点（collider, X→Z←Y）は制御してはならない。
- 合流点を回帰に含めると偽相関を生み、推定βが符号反転しうる（買いが売りシグナルに）。
- 合流点はリターンの「結果」なので、観測時点でリターンは実現済み＝収益化不能。

方向判定にはペアワイズ LiNGAM（非ガウス性を利用、Hyvärinen-Smith 2013）を純 numpy
で実装する。金融リターンはファットテール（非ガウス）でこの前提に適合する。
注：PC / GES など条件付き独立性ベースの探索は causal-learn 等で将来拡張する。
ガウス分布のデータでは方向同定は不能（LiNGAM の前提）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Hyvärinen (1998) 微分エントロピー近似の定数
_K1 = 79.047
_K2 = 7.4129
_GAMMA = 0.37457
_H_GAUSS = (1.0 + np.log(2.0 * np.pi)) / 2.0


def collider_bias_beta(beta: float, delta: float, gamma: float) -> float:
    """合流点を回帰に含めた時の推定 β の期待値（閉形式）。

    構造方程式： y = x·β + ε,  z = y·γ + x·δ + ζ（独立・単位分散ノイズ）。
    y を [x, z] に回帰した時の x の係数は (β − δγ) / (1 + γ²) になる。
    δγ が大きいと β̂ は符号反転する（資料の「符号逆転」）。

    （ブリーフィングが引用する近似形 β − δγ/(1+γ²) に対し、本式は上記 SEM の厳密値。）
    """
    return (beta - delta * gamma) / (1.0 + gamma ** 2)


def _standardize(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    s = u.std()
    if s == 0:
        raise ValueError("zero variance")
    return (u - u.mean()) / s


def _entropy(u: np.ndarray) -> float:
    """標準化変数 u の微分エントロピー近似（Hyvärinen 1998 / DirectLiNGAM）。"""
    return (_H_GAUSS
            - _K1 * (np.mean(np.log(np.cosh(u))) - _GAMMA) ** 2
            - _K2 * (np.mean(u * np.exp(-u ** 2 / 2.0))) ** 2)


def direction_score(x, y) -> float:
    """ペアワイズ LiNGAM 方向スコア。>0 なら x→y（x が原因）, <0 なら y→x。

    真の因果方向では「原因＋独立残差」の総エントロピーが小さい（尤度が高い）。
    Hyvärinen-Smith (2013) の対尤度比。非ガウス性が必要。
    """
    xs = _standardize(x)
    ys = _standardize(y)
    rho = float(np.mean(xs * ys))
    if abs(rho) >= 1.0:
        return 0.0
    res_y = ys - rho * xs          # y を x で回帰した残差（方向 x→y の独立成分）
    res_x = xs - rho * ys          # x を y で回帰した残差（方向 y→x の独立成分）
    res_y = res_y / res_y.std()
    res_x = res_x / res_x.std()
    cost_x_to_y = _entropy(xs) + _entropy(res_y)
    cost_y_to_x = _entropy(ys) + _entropy(res_x)
    return float(cost_y_to_x - cost_x_to_y)    # >0 → x→y（cost が小さい方が真）


def classify_features(X: pd.DataFrame, y, threshold: float = 0.0) -> pd.DataFrame:
    """各特徴量と目的変数 y の因果方向を判定。

    Returns
    -------
    DataFrame  index=特徴量名, columns=[score, role]
        role = 'cause'（x→y, 採用）/ 'effect'（y→x = collider, 除外）。
        score>threshold で cause と判定。
    """
    yv = np.asarray(y, dtype=float)
    rows = {}
    for col in X.columns:
        s = direction_score(X[col].to_numpy(dtype=float), yv)
        rows[col] = (s, "cause" if s > threshold else "effect")
    return pd.DataFrame.from_dict(rows, orient="index", columns=["score", "role"])


def causal_filter(X: pd.DataFrame, y, threshold: float = 0.0):
    """合流点（y の下流 = effect）を除去し、原因（cause）特徴量だけ残す。

    Returns (X_filtered, classification_df)。
    """
    cls = classify_features(X, y, threshold)
    keep = [c for c in X.columns if cls.loc[c, "role"] == "cause"]
    return X[keep], cls
