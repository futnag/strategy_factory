"""GKX 型クロスセクション ML の設計行列（Phase 4）。

全凍結特徴量を 1 つの月次パネル X に統合し、ラベル y（1M 先・トータル・超過・廃止補完）を
組む。**新規特徴量は作らない**（特徴量セットは凍結＝handoff §1,§4）。重複は docs/18 準拠で解消
（accruals は EDINET 版のみ／CF/P は 1 本＝cf_to_price／12-1 モメンタムは value 版 momentum 1 本）。

標準化（一元化済みを使用）：毎月クロスセクションで
  winsorize_cross_sectional(1/99%) → rank_and_fill([-1,1]・欠損中立0) → sector_neutralize
PIT：materialized 特徴量・value/quality/size・EDINET は全て as-of（DiscDate アンカー）、
materialized は月末スナップショット。macro は external.asof_align で公表ラグ。
ラベル：close[t]→close[t+1]・配当込み・本番ユニバース断面平均控除・last_price 廃止補完。

特性 × マクロ交互作用：**線形モデルにのみ明示的な Kronecker 列**を作る（add_interactions）。
木/NN には生の特性＋マクロを渡し交互作用はモデルに学習させる（次元爆発回避＝handoff §4）。
マクロ系列は raw のまま返す（標準化は推定器の Pipeline が train fold で行う＝先読み無し）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from ..data import external as ext
from ..data import feature_store as fs
from ..data import store
from . import factors as fac
from . import flows
from . import panel as pn
from .edinet_factors import edinet_factor_panels
from .factors import value_quality_size_factors
from .fundamentals import fundamentals_panel
from .universe import universe_members

# --- 凍結特徴量セット（重複解消済み・docs/18） -----------------------------
# materialized（feature_store.load_feature・月末 PIT 済）
MATERIALIZED: list[str] = [
    # Phase1 価格/流動性
    "mom_1m_reversal", "mom_3m", "mom_6m", "mom_36m_reversal", "rvol_60", "rvol_252",
    "ivol", "beta", "max_ret", "ret_skew", "industry_momentum", "amihud_illiq",
    "dollar_volume", "zero_ret_days", "turnover",
    # 拡充A：微細構造
    "parkinson_vol", "garman_klass_vol", "corwin_schultz_spread", "rsi",
    "roll_spread", "vpin",
    # 拡充A：信用/空売り（公表日アンカー材化済）
    "margin_imbalance", "short_to_long", "margin_balance_change", "short_interest",
    "days_to_cover", "sector_short_ratio",
    # 拡充B：SUE/改訂/成長/安定/52週/季節性/Dimsonβ
    "sue_recent", "sue_initial", "forecast_revision", "sales_growth", "profit_growth",
    "equity_growth", "roe_stability", "margin_stability", "sustainable_growth",
    "high_52w", "seasonality", "dimson_beta",
]
# value/quality/size（factors.value_quality_size_factors で生成）。重複解消：
#   accruals は EDINET 版に一本化＝ここでは除外／cf_yield≡cf_to_price も EDINET 版に一本化。
VQS: list[str] = ["earnings_yield", "book_to_market", "sales_yield", "div_yield",
                  "roe", "roa", "op_margin", "equity_ratio", "size", "momentum"]
# EDINET 三表（edinet_factor_panels で生成・accruals/cf_to_price はこちらを採用）
EDINET: list[str] = ["fcf_yield", "fcf_yield_3y", "roic", "asset_growth", "accruals",
                     "gross_profitability", "ebitda_margin", "leverage",
                     "net_share_issuance", "rd_intensity", "cf_to_price"]
ALL_FEATURES: list[str] = MATERIALIZED + VQS + EDINET

_VQS_FIELDS = ["FEPS", "Eq", "CFO", "FSales", "FDivAnn", "FNP", "TA", "FOP",
               "EqAR", "NP", "ShOutFY", "TrShFY"]
# 採用するマクロ状態変数（PIT・docs/19 で確定）。credit spread は全データに無く除外。
MACRO_KEYS = ["jp_10y", "term_spread", "vix", "n225_iv", "foreign_flow"]
_N225_IV = Path("data/supplemental/n225_iv.parquet")


def production_universe(rebal: Optional[pd.DatetimeIndex] = None,
                       min_price: float = 100.0, min_mcap: float = 1e10,
                       min_adv: float = 5e7, common_only: bool = True,
                       preset: Optional[str] = None) -> pd.DataFrame:
    """本番ユニバース mask（株価¥100・時価総額床・ADV60¥50M・普通株）を月末で構築（PIT）。

    rebal 省略時は materialized 月末（feature_store の beta 等）を採用。判定と診断で同一の
    ユニバース定義を共有するため、ここに一元化（docs/19・liquid_universe_mask の絶対しきい値版）。
    preset を指定すると universe.LIQUID_UNIVERSE_PRESETS のしきい値を採用（Phase 4b 等）。
    """
    from .universe import LIQUID_UNIVERSE_PRESETS, filter_common_stocks, liquid_universe_mask
    if preset is not None:
        cfg = LIQUID_UNIVERSE_PRESETS[preset]
        min_price, min_mcap, min_adv = cfg["min_price"], cfg["min_mcap"], cfg["min_adv"]
    close = store.load_wide("C").resample("ME").last()
    if rebal is None:
        rebal = fs.load_feature("beta").index
    rebal = pd.DatetimeIndex(rebal)
    cl = close.reindex(index=rebal)
    adv = store.load_wide("Va").rolling(60, min_periods=20).mean().resample("ME").last(
    ).reindex(index=rebal, columns=cl.columns)
    sh = fundamentals_panel(rebal, ["ShOutFY", "TrShFY"])
    shares = (sh["ShOutFY"] - sh.get("TrShFY", 0.0).reindex_like(sh["ShOutFY"]).fillna(0.0)
              ).reindex(index=rebal, columns=cl.columns)
    mcap = (shares * cl).where(lambda x: x > 0)
    mask = liquid_universe_mask(cl, mcap, adv, min_price, min_mcap, min_adv)
    if common_only:
        from pathlib import Path as _P
        mp = _P("data/jquants/equities_master.parquet")
        if mp.exists():
            common = set(filter_common_stocks(pd.read_parquet(mp))["Code"].astype(str))
            for c in mask.columns:
                if str(c) not in common:
                    mask[c] = False
    return mask


@dataclass
class DesignMatrix:
    X: pd.DataFrame              # 特性（標準化済）long, index=(date, Code)
    y: pd.Series                # ラベル（1M 先・トータル・超過・廃止補完）index=(date, Code)
    macro: pd.DataFrame         # マクロ状態（raw・PIT）index=month
    months: pd.DatetimeIndex    # 月末リバランス日（昇順・一意＝CV のグループ軸）
    t1: pd.Series               # 月→ラベル終了月（purge/embargo 用）index=months
    char_cols: list[str]        # 特性列名
    macro_cols: list[str]       # マクロ列名
    universe: pd.DataFrame      # 本番ユニバース bool mask（index=month, col=Code）
    meta: dict = field(default_factory=dict)


# --- 月次価格パネル（PIT・materialized 月末と整合） -------------------------
def _monthly_price_panels(rebal: pd.DatetimeIndex, codes: list[str]):
    adjc = store.load_wide("AdjC").resample("ME").last().reindex(index=rebal, columns=codes)
    close = store.load_wide("C").resample("ME").last().reindex(index=rebal, columns=codes)
    sh = fundamentals_panel(rebal, ["ShOutFY", "TrShFY"], codes=codes)
    shares = (sh["ShOutFY"] - sh.get("TrShFY", 0.0).reindex_like(sh["ShOutFY"]).fillna(0.0)
              ).reindex(index=rebal, columns=codes)
    mcap = (shares * close).where(lambda x: x > 0)
    dps = pn.dividend_per_share_asof(rebal, codes=codes)
    return adjc, close, mcap, dps


# --- 特徴量の生パネル群（重複解消済み） ------------------------------------
def feature_panels(rebal: pd.DatetimeIndex, codes: list[str],
                   mcap: Optional[pd.DataFrame] = None,
                   close: Optional[pd.DataFrame] = None,
                   adjc: Optional[pd.DataFrame] = None) -> dict[str, pd.DataFrame]:
    """凍結特徴量の raw wide を {name: DataFrame(index=rebal, col=codes)} で返す。"""
    if mcap is None or close is None or adjc is None:
        adjc, close, mcap, _ = _monthly_price_panels(rebal, codes)
    out: dict[str, pd.DataFrame] = {}
    for name in MATERIALIZED:                      # 材化済（月末 PIT）
        w = fs.load_feature(name)
        out[name] = (w.reindex(index=rebal, columns=codes) if not w.empty
                     else pd.DataFrame(index=rebal, columns=codes, dtype="float64"))
    pit = fundamentals_panel(rebal, _VQS_FIELDS, codes=codes)   # value/quality/size
    vqs = value_quality_size_factors(pit, raw_price=close, adj_price=adjc)
    for name in VQS:
        if name in vqs:
            out[name] = vqs[name].reindex(index=rebal, columns=codes)
    edp = edinet_factor_panels(rebal, codes=codes, mcap=mcap)   # EDINET 三表
    for name in EDINET:
        if name in edp:
            out[name] = edp[name].reindex(index=rebal, columns=codes)
    return out


# --- 標準化（winsorize→rank_and_fill→sector_neutralize） -------------------
def standardize_panels(panels: dict[str, pd.DataFrame],
                       sector: Optional[pd.Series] = None,
                       lower: float = 0.01, upper: float = 0.99
                       ) -> dict[str, pd.DataFrame]:
    """各特徴量を毎月クロスセクションで winsorize→[-1,1]rank→中立0→（任意）セクター中立。"""
    out = {}
    for name, w in panels.items():
        z = fac.rank_and_fill(fac.winsorize_cross_sectional(w, lower, upper))
        if sector is not None:
            z = fac.sector_neutralize(z, sector)
        out[name] = z
    return out


# --- マクロ状態変数（PIT・raw） --------------------------------------------
def _n225_iv_series(col: str = "n225_iv_atm") -> pd.Series:
    if not _N225_IV.exists():
        return pd.Series(dtype="float64")
    s = pd.read_parquet(_N225_IV)
    c = col if col in s.columns else ("n225_iv_med" if "n225_iv_med" in s.columns
                                      else s.columns[0])
    return s[c].dropna()


def macro_state(rebal: pd.DatetimeIndex, lag_days: int = 1) -> pd.DataFrame:
    """採用マクロ状態変数を PIT（公表ラグ）で月末に整列。raw（標準化は推定器側）。

    金利系（jp_10y・us_10y 等）は monthly→ffill で publication 遅れがあるため lag_days を課す。
    term_spread = jp_10y − jp_policy。vix は日次。n225_iv は options_225 集約（要キャッシュ）。
    foreign_flow は投資部門別の海外ネット強度（PubDate アンカー）。credit spread は無く除外。
    """
    base = ext.load_macro(["jp_10y", "jp_policy", "us_10y", "vix"])
    aligned = ext.asof_align(base, rebal, lag_days=lag_days)
    out = pd.DataFrame(index=rebal)
    out["jp_10y"] = aligned.get("jp_10y")
    out["term_spread"] = aligned.get("jp_10y") - aligned.get("jp_policy")
    out["vix"] = aligned.get("vix")
    iv = _n225_iv_series()                          # N225 IV（公表=取引日, lag 1）
    out["n225_iv"] = (ext.asof_align(iv.to_frame("n225_iv"), rebal, lag_days=lag_days)["n225_iv"]
                      if not iv.empty else np.nan)
    try:                                           # 海外フロー強度（PubDate アンカー）
        fl = flows.load_investor_types()
        nf = flows.net_flow_intensity(fl, investor="foreign", section="TokyoNagoya")
        if "PubDate" in fl.columns and len(nf):
            pub = (fl.dropna(subset=["PubDate"]).drop_duplicates("EnDate")
                   .set_index("EnDate")["PubDate"])
            nf = nf.copy()
            nf.index = pd.to_datetime(pub.reindex(nf.index).fillna(nf.index.to_series()))
            nf = nf.sort_index()
        out["foreign_flow"] = ext.asof_align(nf.to_frame("foreign_flow"), rebal,
                                             lag_days=lag_days)["foreign_flow"]
    except Exception:  # noqa: BLE001
        out["foreign_flow"] = np.nan
    return out.reindex(columns=MACRO_KEYS)


# --- ラベル（1M 先・トータル・超過・廃止補完） -----------------------------
def build_label(rebal: pd.DatetimeIndex, codes: list[str], universe: pd.DataFrame,
                delist_policy: str = "last_price",
                adjc: Optional[pd.DataFrame] = None,
                close: Optional[pd.DataFrame] = None,
                dps: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """y[t,code]=トータル 1M 先リターン − 本番ユニバース断面平均（廃止 delist_policy 補完）。

    universe 非所属は NaN（学習・評価対象外）。超過の基準＝本番ユニバースの断面等加重平均。
    """
    if adjc is None or close is None or dps is None:
        adjc, close, mcap, dps = _monthly_price_panels(rebal, codes)
    tot = pn.total_forward_returns(adjc, close, dps)            # close[t]→close[t+1]＋配当
    tot = pn.impute_delisting(tot, adjc, policy=delist_policy)  # 廃止月を方針で補完
    uni = universe.reindex(index=rebal, columns=codes).fillna(False)
    tot_u = tot.where(uni)                                      # 本番ユニバースのみ
    excess = tot_u.sub(tot_u.mean(axis=1), axis=0)             # 断面平均控除＝超過
    return excess


# --- long 結合（X・y・t1・groups） -----------------------------------------
def _stack(panels: dict[str, pd.DataFrame], universe: pd.DataFrame) -> pd.DataFrame:
    """{name: wide} を long DataFrame（index=(date,Code), col=name）へ。ユニバース内行のみ。

    pandas 3.0 の stack は NaN を落とさないため、ユニバース外（=NaN）を明示 dropna で除く。
    """
    cols = {}
    for name, w in panels.items():
        s = w.where(universe).stack(future_stack=True)     # 非所属は NaN のまま
        cols[name] = s
    X = pd.DataFrame(cols)
    X.index.names = ["date", "Code"]
    return X.dropna(how="all")                              # 非所属行（全特徴 NaN）を除去


def add_interactions(X_chars: pd.DataFrame, macro_long: pd.DataFrame,
                     char_cols: list[str], macro_cols: list[str]) -> pd.DataFrame:
    """線形モデル用：特性 × マクロの Kronecker 交互作用列を追加（生特性＋マクロ＋積）。

    X_chars・macro_long は同じ (date,Code) index。返り値 = [chars | macro | char×macro]。
    """
    parts = [X_chars[char_cols], macro_long[macro_cols]]
    inter = {}
    for c in char_cols:
        for mq in macro_cols:
            inter[f"{c}__x__{mq}"] = X_chars[c] * macro_long[mq]
    parts.append(pd.DataFrame(inter, index=X_chars.index))
    return pd.concat(parts, axis=1)


def build_design_matrix(universe: pd.DataFrame, *, rebal: Optional[pd.DatetimeIndex] = None,
                        codes: Optional[Iterable] = None, sector: Optional[pd.Series] = None,
                        delist_policy: str = "last_price", lag_days: int = 1,
                        standardize: bool = True) -> DesignMatrix:
    """凍結特徴量＋マクロ＋ラベルを 1 つの設計行列に組む（PIT・重複解消・標準化）。

    universe: 本番 bool mask（index=month, col=Code・universe.liquid_universe_mask 由来）。
    rebal/codes 省略時は universe の index/columns を採用。返り値 DesignMatrix。
    交互作用列は線形用に add_interactions で別途生成（次元爆発回避のため既定は付けない）。
    """
    rebal = pd.DatetimeIndex(rebal) if rebal is not None else universe.index
    # codes 既定＝ユニバースに一度でも所属した銘柄のみ（非所属は使われない＝計算節約）。
    codes = [str(c) for c in (codes if codes is not None
                              else universe_members(universe.reindex(index=rebal)))]
    uni = universe.reindex(index=rebal, columns=codes).fillna(False)
    adjc, close, mcap, dps = _monthly_price_panels(rebal, codes)
    raw = feature_panels(rebal, codes, mcap=mcap, close=close, adjc=adjc)
    # 本番ユニバースに絞ってから標準化＝ランクは「本番ユニバース内」で取る（非所属を巻き込まない）。
    raw_u = {name: w.where(uni) for name, w in raw.items()}
    std = standardize_panels(raw_u, sector) if standardize else raw_u
    char_cols = [c for c in ALL_FEATURES if c in std]
    X = _stack({c: std[c] for c in char_cols}, uni)
    macro = macro_state(rebal, lag_days=lag_days)
    y = build_label(rebal, codes, uni, delist_policy, adjc=adjc, close=close,
                    dps=dps).stack(future_stack=True)
    y = y.dropna()
    y.index.names = ["date", "Code"]
    # X と y を共通 (date,Code) で整列（ユニバース内・ラベル有り）
    common = X.index.intersection(y.index)
    X, y = X.loc[common], y.loc[common]
    months = pd.DatetimeIndex(sorted(X.index.get_level_values("date").unique()))
    t1 = pd.Series(months, index=months).shift(-1)             # 月→翌月（ラベル終了）
    t1.iloc[-1] = months[-1]
    return DesignMatrix(X=X, y=y, macro=macro, months=months, t1=t1,
                        char_cols=char_cols, macro_cols=list(macro.columns),
                        universe=uni, meta={"n_features": len(char_cols),
                                            "delist_policy": delist_policy,
                                            "lag_days": lag_days})


# --- 永続キャッシュ（フル材化は重い＝判定/評価で再利用するため一度だけ作る） --------
def save_design_matrix(D: DesignMatrix, cache_dir: str = "data/phase4") -> None:
    """設計行列を parquet＋json で保存（判定/評価が同一 D を読み直せるように）。"""
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    D.X.to_parquet(d / "X.parquet")
    D.y.to_frame("y").to_parquet(d / "y.parquet")
    D.macro.to_parquet(d / "macro.parquet")
    D.universe.astype(bool).to_parquet(d / "universe.parquet")
    (d / "meta.json").write_text(json.dumps({
        "months": [t.isoformat() for t in D.months],
        "t1": [t.isoformat() for t in D.t1],
        "char_cols": D.char_cols, "macro_cols": D.macro_cols, "meta": D.meta,
    }, ensure_ascii=False), encoding="utf-8")


def load_design_matrix(cache_dir: str = "data/phase4") -> DesignMatrix:
    """save_design_matrix で書いた設計行列を読み戻す。"""
    d = Path(cache_dir)
    X = pd.read_parquet(d / "X.parquet")
    y = pd.read_parquet(d / "y.parquet")["y"]
    macro = pd.read_parquet(d / "macro.parquet")
    universe = pd.read_parquet(d / "universe.parquet")
    j = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    months = pd.DatetimeIndex(pd.to_datetime(j["months"]))
    t1 = pd.Series(pd.to_datetime(j["t1"]), index=months)
    return DesignMatrix(X=X, y=y, macro=macro, months=months, t1=t1,
                        char_cols=j["char_cols"], macro_cols=j["macro_cols"],
                        universe=universe, meta=j.get("meta", {}))
