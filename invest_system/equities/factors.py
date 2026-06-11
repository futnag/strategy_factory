"""クロスセクション・ファクターの算出・標準化・交絡中立化。

バリュー/クオリティ/サイズはすべて「生株価 C ＋ 開示時点の財務」で内部整合的に
計算する（分割調整後 AdjC はリターン/モメンタム専用）。フロー変数は四半期累計の
歪みを避けるため予想通年値（FEPS/FSales/FNP/FOP）を優先する。

交絡（セクター・サイズ）の中立化は pillar C の核心：ファクターの見かけの効果から
業種・規模という共通要因を除き、固有の効果を取り出す。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def market_cap(raw_price: pd.DataFrame, shares_out: pd.DataFrame,
               treasury: pd.DataFrame | None = None) -> pd.DataFrame:
    """時価総額 = 生株価 × (発行済株式数 − 自己株式数)。ラベルで自動整合。

    片側欠損（発行済 NaN × 自己株あり）は fill_value=0 で株数が負になり時価総額の
    符号が反転するため、株数 ≤0 は NaN（後段の z 化で自然に除外）。
    """
    sh = shares_out
    if treasury is not None:
        sh = shares_out.sub(treasury, fill_value=0.0)
    return raw_price * sh.where(sh > 0)


def value_quality_size_factors(pit: dict[str, pd.DataFrame], raw_price: pd.DataFrame,
                               adj_price: pd.DataFrame | None = None,
                               mom_lookback: int = 12, mom_skip: int = 1
                               ) -> dict[str, pd.DataFrame]:
    """PIT財務（フィールド別wide）＋価格から各ファクターのwideパネルを生成。

    値の向きは「大きいほど割安/高品質/小型」になるよう符号付け（=ロング側が
    期待プレミアム方向）。欠損は NaN のまま（後段のz化で自然に除外）。
    """
    def f(name: str) -> pd.DataFrame:
        return pit.get(name, pd.DataFrame(index=raw_price.index))

    shares = f("ShOutFY").sub(f("TrShFY"), fill_value=0.0)
    shares = shares.where(shares > 0)        # 片側欠損で負になる開示は NaN（market_cap と同じ）
    mcap = raw_price * shares

    out: dict[str, pd.DataFrame] = {}
    # --- バリュー（高いほど割安）---
    out["earnings_yield"] = f("FEPS") / raw_price          # 予想E/P
    out["book_to_market"] = f("Eq") / mcap                  # B/M
    out["cf_yield"] = f("CFO") / mcap                       # 営業CF利回り
    out["sales_yield"] = f("FSales") / mcap                 # 予想売上利回り
    out["div_yield"] = f("FDivAnn") / raw_price             # 予想配当利回り
    # --- クオリティ（高いほど良質）---
    eq = f("Eq").replace(0.0, np.nan)
    ta = f("TA").replace(0.0, np.nan)
    fsales = f("FSales").replace(0.0, np.nan)
    out["roe"] = f("FNP") / eq                              # 予想ROE
    out["roa"] = f("FNP") / ta                              # 予想ROA
    out["op_margin"] = f("FOP") / fsales                    # 予想営業利益率
    out["equity_ratio"] = f("EqAR")                         # 自己資本比率
    out["accruals"] = (f("CFO") - f("NP")) / ta             # 低アクルーアル(CFO>NP実利益)=高品質
    # --- サイズ（小型ほど大きい値＝小型プレミアム方向）---
    out["size"] = -np.log(mcap.where(mcap > 0))
    # --- モメンタム（12-1, 調整後価格）---
    if adj_price is not None:
        out["momentum"] = adj_price.shift(mom_skip) / adj_price.shift(mom_lookback) - 1.0

    # ユニバース列に整列
    return {k: v.reindex(index=raw_price.index, columns=raw_price.columns)
            for k, v in out.items()}


def low_volatility(price: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """低ボラティリティ・ファクター：各時点で過去 window 期間のリターン実現ボラに負号。

    低ボラ・アノマリー（高ボラ株が長期にリスク調整後アンダーパフォーム）。price は任意頻度
    の調整済価格パネル（月末パネルなら window=12 で約1年ボラ）。値が大きい＝低ボラ＝ロング側。
    rolling は過去のみ参照するため各時点 t のボラは t までのリターンで算出＝先読みなし。
    """
    ret = price.pct_change()
    vol = ret.rolling(window, min_periods=max(2, window // 2)).std()
    return -vol


def residual_momentum(ret: pd.DataFrame, market: pd.Series, *, window: int = 36,
                      mom_len: int = 12, skip: int = 1,
                      standardize: bool = True) -> pd.DataFrame:
    """残差モメンタム（iMOM）。Blitz-Huij-Martens 2011 / Chaves 2016（docs/04 P3-H）。

    各 (t, 銘柄) で過去 window ヶ月（t-window〜t-1）の月次リターンを市場リターンへ
    単回帰（市場モデル。Chaves は単回帰だけで主要な便益が出ると報告）し、残差のうち
    「直近 skip ヶ月を除く直近 mom_len ヶ月窓」（月 t-mom_len〜t-1-skip＝mom_len−skip 本）
    の合計を、standardize=True ならその標本標準偏差（ddof=1）で割って返す。
    伝統的モメンタムから市場ベータ起因の系統成分（リバーサルしやすい）を除き、
    銘柄固有のアンダーリアクション成分を取り出す仮説の実装。

    PIT：行 t は ≤t-1 の実現リターンのみ使用（先読みなし）。窓内に欠損のある銘柄は
    NaN（除外ルールの事前固定＝完全な window ヶ月を要求。上場間もない銘柄は自然に
    除外される）。ret: 月次リターン wide（index=月末, col=Code）。market: 同 index の
    市場リターン（等加重等）。market 側に欠損のある行は全銘柄 NaN。
    """
    if window <= mom_len or mom_len <= skip:
        raise ValueError("require window > mom_len > skip")
    out = pd.DataFrame(np.nan, index=ret.index, columns=ret.columns, dtype=float)
    R = ret.to_numpy(dtype=float)
    M = market.reindex(ret.index).to_numpy(dtype=float)
    for it in range(window, len(ret.index)):
        mw = M[it - window:it]                        # 月 t-window 〜 t-1
        if np.isnan(mw).any():
            continue
        Rw = R[it - window:it]
        ok = ~np.isnan(Rw).any(axis=0)                # 完全な窓を要求（事前固定）
        if not ok.any():
            continue
        Y = Rw[:, ok]
        mc = mw - mw.mean()
        denom = float((mc * mc).sum())
        if denom <= 0:
            continue
        beta = (mc @ (Y - Y.mean(axis=0))) / denom
        alpha = Y.mean(axis=0) - beta * mw.mean()
        resid = Y - (alpha[None, :] + np.outer(mw, beta))
        seg = resid[window - mom_len: window - skip]  # 月 t-mom_len 〜 t-1-skip
        sig = seg.sum(axis=0)
        if standardize:
            sd = seg.std(axis=0, ddof=1)
            sig = np.where(sd > 0, sig / sd, np.nan)
        out.iloc[it, np.flatnonzero(ok)] = sig
    return out


def cross_sectional_zscore(df: pd.DataFrame, winsor: float = 3.0) -> pd.DataFrame:
    """各日付（行）で銘柄横断にzスコア化。winsor で±σにクリップ（外れ値抑制）。"""
    mu = df.mean(axis=1)
    sd = df.std(axis=1, ddof=0)
    z = df.sub(mu, axis=0).div(sd.replace(0.0, np.nan), axis=0)
    if winsor is not None and winsor > 0:
        z = z.clip(lower=-winsor, upper=winsor)
    return z


def sector_neutralize(df: pd.DataFrame, sector: pd.Series) -> pd.DataFrame:
    """各日付で同一セクター内の平均を引き、業種共通成分を除去（交絡制御）。

    sector: index=Code, 値=セクターコード(S33等)。df の列に整合させて使用。
    """
    sec = sector.reindex([str(c) for c in df.columns])
    sec.index = df.columns
    groups = sec.fillna("NA")
    # 列方向に groupby して各行から群平均を引く
    demeaned = df.copy()
    for _, cols in groups.groupby(groups):
        block = df[cols.index]
        demeaned[cols.index] = block.sub(block.mean(axis=1), axis=0)
    return demeaned


def cross_sectional_residualize(target: pd.DataFrame,
                                controls: list[pd.DataFrame]) -> pd.DataFrame:
    """各日付で target を controls に銘柄横断回帰し、残差（直交成分）を返す。

    controls の線形結合で説明できる部分を除去し、target 固有の独立情報のみ残す。
    あるファクターが既知ファクター（モメンタム/サイズ/バリュー等）の代理に過ぎない
    かを検定する用途：残差化後もシグナルが残れば独立、消えれば代理。
    """
    cols = target.columns
    out = pd.DataFrame(np.nan, index=target.index, columns=cols, dtype=float)
    for t in target.index:
        y = target.loc[t].to_numpy(dtype=float)
        xs = [c.reindex(columns=cols).reindex(index=[t]).to_numpy(dtype=float).ravel()
              for c in controls]
        X = np.column_stack([np.ones_like(y)] + xs)        # 切片＋controls
        mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if int(mask.sum()) < X.shape[1] + 2:
            continue
        beta, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
        resid = np.full_like(y, np.nan)
        resid[mask] = y[mask] - X[mask] @ beta
        out.loc[t] = resid
    return out
