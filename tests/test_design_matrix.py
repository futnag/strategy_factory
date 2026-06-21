"""Phase 4 設計行列の純関数検証（PIT・重複解消・交互作用次元・廃止/超過ラベル）。

ネットワーク・実データ不要（合成フィクスチャ）。重い build_design_matrix（materialized/
EDINET/macro 依存）は本体スクリプトの実データスモークで確認し、ここでは純粋部品を固定する。
"""
import numpy as np
import pandas as pd
import pytest

from invest_system.equities import design_matrix as dm


# --- 重複解消（凍結セットの不変条件・docs/18） ------------------------------
def test_frozen_feature_set_dedup():
    feats = dm.ALL_FEATURES
    assert len(feats) == len(set(feats))            # 重複名なし
    assert feats.count("accruals") == 1             # accruals は 1 本（EDINET 版）
    assert "accruals" in dm.EDINET and "accruals" not in dm.VQS
    assert "cf_to_price" in dm.EDINET and "cf_yield" not in feats   # CF/P は 1 本に統一
    # 12-1 モメンタムは value 版 momentum 1 本（materialized の momentum_12_1 は入れない）
    assert "momentum" in dm.VQS and "momentum_12_1" not in feats


# --- 交互作用の次元・命名 --------------------------------------------------
def test_add_interactions_dimension_and_names():
    idx = pd.MultiIndex.from_product([pd.to_datetime(["2025-01-31"]), ["A", "B"]],
                                     names=["date", "Code"])
    chars = pd.DataFrame({"f1": [0.5, -0.5], "f2": [1.0, -1.0]}, index=idx)
    macro = pd.DataFrame({"m1": [2.0, 2.0], "m2": [3.0, 3.0]}, index=idx)
    out = dm.add_interactions(chars, macro, ["f1", "f2"], ["m1", "m2"])
    # 列数 = chars(2) + macro(2) + 積(2×2=4) = 8
    assert out.shape[1] == 2 + 2 + 4
    assert "f1__x__m1" in out.columns
    assert out.loc[(idx[0]), "f1__x__m1"] == pytest.approx(0.5 * 2.0)   # 積が正しい


# --- 標準化：winsorize→rank→中立0→セクター中立 ---------------------------
def test_standardize_panels_rank_neutral_no_nan():
    idx = pd.to_datetime(["2025-01-31"])
    raw = {"f": pd.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0], "D": [np.nan]},
                             index=idx)}
    std = dm.standardize_panels(raw, sector=None)["f"]
    assert std.loc[idx[0], "D"] == 0.0              # 欠損は中立 0
    assert std.loc[idx[0], "A"] == pytest.approx(-1.0)  # 最小→ -1
    assert std.loc[idx[0], "C"] == pytest.approx(1.0)   # 最大→ +1
    assert not std.isna().any().any()


# --- ラベル：トータル・超過・廃止補完・ユニバース外 NaN --------------------
def _label_panels():
    idx = pd.to_datetime(["2025-01-31", "2025-02-28", "2025-03-31"])
    codes = ["A", "B", "C"]
    adjc = pd.DataFrame({"A": [100., 110., 121.], "B": [100., 100., 100.],
                         "C": [100., 90., np.nan]}, index=idx)   # C は 2月で廃止
    close = adjc.copy()
    dps = pd.DataFrame(0.0, index=idx, columns=codes)
    return idx, codes, adjc, close, dps


def test_build_label_excess_and_delisting_last_price():
    idx, codes, adjc, close, dps = _label_panels()
    uni = pd.DataFrame(True, index=idx, columns=codes)
    y = dm.build_label(idx, codes, uni, delist_policy="last_price",
                       adjc=adjc, close=close, dps=dps)
    # Jan: tot A=+0.10,B=0,C=-0.10 → 超過の断面平均=0
    assert abs(y.loc[idx[0]].mean()) < 1e-12
    assert y.loc[idx[0], "A"] == pytest.approx(0.10)
    # C は 2月廃止：last_price で 0 補完＝NaN で落ちない（生存者バイアス排除）
    assert not np.isnan(y.loc[idx[1], "C"])


def test_build_label_delisting_envelope_differs():
    idx, codes, adjc, close, dps = _label_panels()
    uni = pd.DataFrame(True, index=idx, columns=codes)
    lp = dm.build_label(idx, codes, uni, "last_price", adjc=adjc, close=close, dps=dps)
    m100 = dm.build_label(idx, codes, uni, "all_minus100", adjc=adjc, close=close, dps=dps)
    # 廃止月 C の超過は last_price と全−100% で異なる（封筒の幅）
    assert lp.loc[idx[1], "C"] != pytest.approx(m100.loc[idx[1], "C"])


def test_build_label_universe_excludes_nonmembers():
    idx, codes, adjc, close, dps = _label_panels()
    uni = pd.DataFrame(True, index=idx, columns=codes)
    uni.loc[idx[0], "B"] = False                    # Jan は B を非所属に
    y = dm.build_label(idx, codes, uni, "last_price", adjc=adjc, close=close, dps=dps)
    assert np.isnan(y.loc[idx[0], "B"])             # 非所属は NaN（対象外）
    # 残り A,C のみで超過平均=0
    assert abs(y.loc[idx[0]].dropna().mean()) < 1e-12
