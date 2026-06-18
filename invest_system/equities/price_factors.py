"""価格系・流動性系のクロスセクション・ファクター（GKX Phase 1）。

GKX（Gu, Kelly & Xiu 2020）が最も支配的と示した価格系（多ホライズン・モメンタム／短期・長期
リバーサル／各種ボラ／特異ボラ／ベータ／MAX）と流動性系（Amihud／回転率／売買代金／ゼロ
リターン日）を、**既存の J-Quants 日次データだけ**（EDINET 非依存）で供給する。`factors.py` の
肥大化回避のため分離し、標準化（`cross_sectional_zscore`/`cross_sectional_rank`）・セクター中立化
（`sector_neutralize`）は factors.py のものを再利用する（本モジュールは raw を返す）。

規律（Phase 2 と同一）:
- **PIT・先読み厳禁**：各 f[t] は ≤t の価格・出来高のみ。trailing 窓は過去のみ参照（pandas の
  rolling/shift は過去方向）。各特徴量に「未来改変→≤t 不変」のテストを課す。
- **符号は「大きいほどロング側＝期待プレミアム方向」に統一**（低ボラ・低ベータ・低回転・MAX・
  歪度は負号、Amihud・ゼロ日は正号）。docstring に出典＋符号根拠を併記。
- **窓は §4 標準で事前固定**（チューニング探索はしない）：短期=1M、ボラ=60/252、ベータ/特異ボラ
  =252、流動性=60（営業日）。1 ヶ月≈21 営業日。
- 不完全窓は NaN（min_periods で要求）。返り値は wide パネル（index=日付, col=Code）。

入力は日次：adj_close（調整済・モメンタム/ボラ用）、va（売買代金 Va・流動性用）、market（市場
日次リターン＝既定は等加重 ret.mean(axis=1)＝feature_store.build_regime と同一規約）、
mcap（時価総額・回転率用）、sector（S33・業種モメンタム用）。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

MONTH = 21          # 1 ヶ月の営業日数（事前固定）
_ANN = np.sqrt(252.0)


def _ret(adj_close: pd.DataFrame) -> pd.DataFrame:
    return adj_close.pct_change()


def _mp(window: int) -> int:
    """完全窓に近い最小観測数（不完全窓は NaN）。"""
    return max(2, int(window * 0.8))


# === モメンタム／リバーサル ===============================================
def mom_1m_reversal(adj_close: pd.DataFrame) -> pd.DataFrame:
    """短期リバーサル：過去 1 ヶ月リターンに負号（Jegadeesh 1990）。高い=直近の負け=ロング側。"""
    return -(adj_close / adj_close.shift(MONTH) - 1.0)


def mom_3m(adj_close: pd.DataFrame) -> pd.DataFrame:
    """3 ヶ月モメンタム（直近 1 ヶ月スキップ）。t は ≤t-1M の価格のみ＝先読みなし。"""
    return adj_close.shift(MONTH) / adj_close.shift(3 * MONTH) - 1.0


def mom_6m(adj_close: pd.DataFrame) -> pd.DataFrame:
    """6 ヶ月モメンタム（直近 1 ヶ月スキップ。Jegadeesh-Titman 1993）。"""
    return adj_close.shift(MONTH) / adj_close.shift(6 * MONTH) - 1.0


def mom_36m_reversal(adj_close: pd.DataFrame) -> pd.DataFrame:
    """長期リバーサル：過去 36 ヶ月リターンに負号（De Bondt-Thaler 1985）。"""
    return -(adj_close / adj_close.shift(36 * MONTH) - 1.0)


def industry_momentum(adj_close: pd.DataFrame, sector: pd.Series,
                      lookback: int = 12 * MONTH, skip: int = MONTH) -> pd.DataFrame:
    """業種モメンタム（Moskowitz-Grinblatt 1999）：S33 業種の等加重 12-1 リターンを構成銘柄へ。

    各業種の等加重日次リターンから業種指数を作り、その 12-1 モメンタムを当該業種の全銘柄に
    ブロードキャスト。高い=強い業種=ロング側。業種未知の銘柄は NaN。
    """
    ret = _ret(adj_close)
    sec = sector.reindex([str(c) for c in adj_close.columns])
    sec.index = adj_close.columns
    out = pd.DataFrame(np.nan, index=adj_close.index, columns=adj_close.columns)
    for s, cols in sec.fillna("NA").groupby(sec.fillna("NA")):
        if s == "NA":
            continue
        sret = ret[list(cols.index)].mean(axis=1)        # 業種 等加重日次リターン
        lvl = (1.0 + sret.fillna(0.0)).cumprod()
        mom = lvl.shift(skip) / lvl.shift(lookback) - 1.0
        out[list(cols.index)] = pd.concat([mom] * len(cols.index), axis=1).values
    return out


# === ボラティリティ／リスク ===============================================
def realized_vol(adj_close: pd.DataFrame, window: int) -> pd.DataFrame:
    """実現ボラ（年率）に負号（低ボラ・アノマリー：Ang ら 2006／Baker-Bradley-Wurgler 2011）。

    高い（=負号後）＝低ボラ＝ロング側。t の値は ≤t のリターンのみ。
    """
    vol = _ret(adj_close).rolling(window, min_periods=_mp(window)).std() * _ANN
    return -vol


def _rolling_beta_residvar(ret: pd.DataFrame, market: pd.Series, window: int):
    """trailing 市場モデル（単回帰）の β と残差分散を**ベクトル化**で算出（≤t のみ）。

    var(resid)=var(ret)-β²·var(mkt)（OLS 残差の性質）を使い、銘柄ループを避ける。
    """
    mp = _mp(window)
    m = market.reindex(ret.index)
    mm = m.rolling(window, min_periods=mp).mean()
    var_m = (m * m).rolling(window, min_periods=mp).mean() - mm * mm
    rm = ret.rolling(window, min_periods=mp).mean()
    cov = (ret.mul(m, axis=0).rolling(window, min_periods=mp).mean()
           .sub(rm.mul(mm, axis=0)))
    beta = cov.div(var_m.replace(0.0, np.nan), axis=0)
    var_r = (ret * ret).rolling(window, min_periods=mp).mean() - rm * rm
    resid_var = (var_r.sub(beta.pow(2).mul(var_m, axis=0))).clip(lower=0.0)
    return beta, resid_var


def idiosyncratic_vol(adj_close: pd.DataFrame, market: Optional[pd.Series] = None,
                      window: int = 252) -> pd.DataFrame:
    """特異ボラ（年率）に負号（Ang-Hodrick-Xing-Zhang 2006：高 ivol→低リターン）。

    市場（既定=等加重）への trailing 単回帰残差の標準偏差。残差分散は var(ret)-β²var(mkt) で
    閉形式（自己条件付けを避けるための per-t trailing 推定）。高い（負号後）=低 ivol=ロング側。
    """
    ret = _ret(adj_close)
    mkt = market if market is not None else ret.mean(axis=1)
    _, resid_var = _rolling_beta_residvar(ret, mkt, window)
    return -np.sqrt(resid_var) * _ANN


def market_beta(adj_close: pd.DataFrame, market: Optional[pd.Series] = None,
                window: int = 252) -> pd.DataFrame:
    """市場ベータに負号（BAB：Frazzini-Pedersen 2014。低ベータ＝リスク調整後アウトパフォーム）。

    高い（負号後）=低ベータ=ロング側。trailing 252 営業日の単回帰ベータ（≤t のみ）。
    """
    ret = _ret(adj_close)
    mkt = market if market is not None else ret.mean(axis=1)
    beta, _ = _rolling_beta_residvar(ret, mkt, window)
    return -beta


def max_return(adj_close: pd.DataFrame, window: int = MONTH) -> pd.DataFrame:
    """MAX：過去 1 ヶ月の日次最大リターンに負号（Bali-Cakici-Whitelaw 2011：宝くじ志向→低リターン）。"""
    return -_ret(adj_close).rolling(window, min_periods=_mp(window)).max()


def return_skew(adj_close: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """日次リターンの trailing 歪度に負号（特異歪度プレミアム：高歪度（宝くじ）→低リターン）。"""
    return -_ret(adj_close).rolling(window, min_periods=_mp(window)).skew()


# === 流動性 ===============================================================
def amihud_illiquidity(adj_close: pd.DataFrame, va: pd.DataFrame,
                       window: int = 60) -> pd.DataFrame:
    """Amihud(2002) 非流動性：mean(|日次リターン| / 売買代金 Va) の trailing 平均。正符号＝高い=非流動=ロング側。

    小型・低流動性に偏在（取引コスト依存）＝docs/15 に注意。Va は円・|ret|/Va は微小だが
    クロスセクションのランク/z で吸収される（絶対スケールは無関係）。
    """
    illiq = (_ret(adj_close).abs() / va.where(va > 0))
    return illiq.rolling(window, min_periods=_mp(window)).mean()


def turnover(va: pd.DataFrame, mcap: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """回転率：trailing 平均 Va / 時価総額に負号（高回転→低リターン：Datar-Naik-Radcliffe 1998）。

    高い（負号後）=低回転=ロング側。mcap は生株価×（発行済−自己株）。
    """
    avg_va = va.rolling(window, min_periods=_mp(window)).mean()
    return -(avg_va / mcap.where(mcap > 0))


def dollar_volume(va: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """売買代金（流動性/サイズ代理）：trailing 平均 Va の対数に負号（大型/高流動→低リターン）。

    高い（負号後）=低流動/小型=ロング側（サイズ因子と整合）。
    """
    avg_va = va.rolling(window, min_periods=_mp(window)).mean()
    return -np.log(avg_va.where(avg_va > 0))


def zero_return_days(adj_close: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """ゼロリターン日比率（Lesmond-Ogden-Trzcinka 1999 の非流動性代理）。正符号＝高い=非流動=ロング側。

    取引が薄く価格が動かない日の比率。リターン NaN（未上場/欠損）は分母から除外。
    """
    ret = _ret(adj_close)
    is_zero = (ret == 0.0).where(ret.notna())            # NaN は計上しない
    return is_zero.rolling(window, min_periods=_mp(window)).mean()


# === まとめ（材化の入口） =================================================
def compute_all(adj_close: pd.DataFrame, va: Optional[pd.DataFrame] = None,
                market: Optional[pd.Series] = None, mcap: Optional[pd.DataFrame] = None,
                sector: Optional[pd.Series] = None) -> dict[str, pd.DataFrame]:
    """与えた入力から計算可能な §4 ファクターの日次 wide を dict で返す（符号付き・raw）。

    adj_close は必須。va が在れば流動性系、mcap が在れば回転率、sector が在れば業種モメンタムを
    追加。market 既定は等加重 ret.mean(axis=1)。標準化・中立化は呼び出し側で factors の関数を重ねる。
    """
    ret = _ret(adj_close)
    mkt = market if market is not None else ret.mean(axis=1)
    out: dict[str, pd.DataFrame] = {
        "mom_1m_reversal": mom_1m_reversal(adj_close),
        "mom_3m": mom_3m(adj_close),
        "mom_6m": mom_6m(adj_close),
        "mom_36m_reversal": mom_36m_reversal(adj_close),
        "rvol_60": realized_vol(adj_close, 60),
        "rvol_252": realized_vol(adj_close, 252),
        "ivol": idiosyncratic_vol(adj_close, mkt, 252),
        "beta": market_beta(adj_close, mkt, 252),
        "max_ret": max_return(adj_close, MONTH),
        "ret_skew": return_skew(adj_close, 252),
    }
    if sector is not None:
        out["industry_momentum"] = industry_momentum(adj_close, sector)
    if va is not None:
        out["amihud_illiq"] = amihud_illiquidity(adj_close, va, 60)
        out["dollar_volume"] = dollar_volume(va, 60)
        out["zero_ret_days"] = zero_return_days(adj_close, 60)
        if mcap is not None:
            out["turnover"] = turnover(va, mcap, 60)
    return out
