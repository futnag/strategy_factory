"""EDINET 三表ファンダ：パース／要素IDマッピング／PIT統合／特徴量（ネットワーク不要）。

DL 済みの zip には依存せず、合成フィクスチャで純関数を検証する（CI 安全・決定的）。
handoff §6 受け入れ基準：①先読み不変②被覆/基準別null④移行断絶NaN⑤コードクロスウォーク。
"""
import zipfile

import numpy as np
import pandas as pd

from invest_system.data.sources import edinet as ed
from invest_system.data.sources import edinet_taxonomy as tax
from invest_system.equities import edinet_factors as efa
from invest_system.equities import edinet_fundamentals as ef
from invest_system.equities import factors as fac
from invest_system.equities import fundamentals as fu


# --- フィクスチャ -----------------------------------------------------------
def _facts(rows: list[tuple]) -> pd.DataFrame:
    """(element_id, context_id, value) の facts（read_xbrl_csv 相当）を組む。"""
    df = pd.DataFrame(rows, columns=["element_id", "context_id", "value"])
    for c in ["item_name", "rel_year", "consolidated", "period_type", "unit_id", "unit"]:
        df[c] = ""
    df["namespace"] = df["element_id"].str.split(":").str[0]
    df["value_num"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def _make_type5_zip(path, rows: list[tuple], with_audit: bool = True) -> None:
    """UTF-16・タブ区切りの本報告 CSV を含む type=5 ZIP を作る（監査 jpaud も混ぜる）。"""
    header = ["要素ID", "項目名", "コンテキストID", "相対年度", "連結・個別",
              "期間・時点", "ユニットID", "単位", "値"]
    lines = ["\t".join(header)]
    for eid, ctx, val in rows:
        lines.append("\t".join([eid, "名", ctx, "当期", "連結", "-", "-", "-", str(val)]))
    body = ("\r\n".join(lines)).encode("utf-16")
    with zipfile.ZipFile(path, "w") as z:
        if with_audit:                       # 監査報告（除外されるべき小さい CSV）
            z.writestr("XBRL_TO_CSV/jpaud-aai-001_E1_2025-03-31.csv",
                       "監査\t情報".encode("utf-16"))
        z.writestr("XBRL_TO_CSV/jpcrp030000-asr-001_E1_2025-03-31.csv", body)


# --- read_xbrl_csv ----------------------------------------------------------
def test_read_xbrl_csv_picks_report_and_parses(tmp_path):
    z = tmp_path / "S1_5.zip"
    _make_type5_zip(z, [("jpigp_cor:RevenueIFRS", "CurrentYearDuration", "100"),
                        ("jpigp_cor:AssetsIFRS", "CurrentYearInstant", "500")])
    df = ed.read_xbrl_csv(z)
    assert list(df.columns)[:9] == ed._XBRL_CSV_COLS      # 9 列に正規化
    assert "jpaud" not in " ".join(df["element_id"])      # 監査 CSV は除外
    assert df["namespace"].iloc[0] == "jpigp_cor"
    assert df["value_num"].iloc[0] == 100.0


# --- 会計基準判定 -----------------------------------------------------------
def test_detect_basis():
    ifrs = _facts([("jpigp_cor:AssetsIFRS", "CurrentYearInstant", "1")])
    jgaap = _facts([("jppfs_cor:Assets", "CurrentYearInstant", "1")])
    us = _facts([("jpcrp030000-asr_E1:Foo", "CurrentYearInstant", "1")])
    assert tax.detect_basis(ifrs) == tax.IFRS
    assert tax.detect_basis(jgaap) == tax.JGAAP
    assert tax.detect_basis(us) == tax.USGAAP


def test_select_prefers_consolidated_over_nonconsolidated():
    # 連結（メンバー無し）と親会社単体（_NonConsolidatedMember）が併存 → 連結を採る
    facts = _facts([
        ("jppfs_cor:Assets", "CurrentYearInstant", "900"),
        ("jppfs_cor:Assets", "CurrentYearInstant_NonConsolidatedMember", "300"),
    ])
    val, eid = tax.select_value(facts, "total_assets", tax.JGAAP)
    assert val == 900.0


def test_select_suffix_fallback_for_custom_revenue():
    # 標準タグが無く銘柄固有拡張のみ → 接尾辞フォールバックで採用（トヨタ型）
    facts = _facts([("jpcrp030000-asr_E1:TotalNetRevenuesIFRS",
                     "CurrentYearDuration", "37154298")])
    val, eid = tax.select_value(facts, "net_sales", tax.IFRS)
    assert val == 37154298.0
    assert eid.endswith("TotalNetRevenuesIFRS")


# --- 有利子負債（集約／個別合算／無借金／欠落） -----------------------------
def test_interest_debt_ifrs_aggregate():
    facts = _facts([
        ("jpigp_cor:AssetsIFRS", "CurrentYearInstant", "100"),
        ("jpigp_cor:InterestBearingLiabilitiesCLIFRS", "CurrentYearInstant", "30"),
        ("jpigp_cor:InterestBearingLiabilitiesNCLIFRS", "CurrentYearInstant", "20"),
    ])
    assert tax.interest_debt(facts, tax.IFRS) == 50.0


def test_interest_debt_ifrs_granular_sum():
    # 集約タグ無し → 個別タグ（Borrowings + LongTermDebt + Lease）を合算（日立型）
    facts = _facts([
        ("jpigp_cor:AssetsIFRS", "CurrentYearInstant", "100"),
        ("jpigp_cor:BorrowingsCLIFRS", "CurrentYearInstant", "7"),
        ("jpigp_cor:LongTermDebtNCLIFRS", "CurrentYearInstant", "76"),
    ])
    assert tax.interest_debt(facts, tax.IFRS) == 83.0


def test_interest_debt_jgaap_debtfree_zero_but_ifrs_missing_none():
    # JGAAP：BS 在り＆負債科目ゼロ → 無借金 0（任天堂型）
    jgaap = _facts([("jppfs_cor:Assets", "CurrentYearInstant", "100")])
    assert tax.interest_debt(jgaap, tax.JGAAP) == 0.0
    # IFRS：集約も個別も無い → None（偽ゼロを出さない）
    ifrs = _facts([("jpigp_cor:AssetsIFRS", "CurrentYearInstant", "100")])
    assert tax.interest_debt(ifrs, tax.IFRS) is None


def test_extract_canonical_has_basis_and_debt():
    facts = _facts([
        ("jpigp_cor:AssetsIFRS", "CurrentYearInstant", "100"),
        ("jpigp_cor:RevenueIFRS", "CurrentYearDuration", "50"),
        ("jpigp_cor:InterestBearingLiabilitiesCLIFRS", "CurrentYearInstant", "10"),
    ])
    out = tax.extract_canonical(facts)
    assert out["basis"] == tax.IFRS
    assert out["net_sales"] == 50.0
    assert out["interest_debt"] == 10.0


# --- PIT 先読み不変（提出日アンカー） --------------------------------------
def _two_year_long():
    return pd.DataFrame([
        {"Code": "72030", "DiscDate": "2024-06-20", "period_end": "2024-03-31",
         "basis": "IFRS", "total_assets": 110.0},
        {"Code": "72030", "DiscDate": "2025-06-20", "period_end": "2025-03-31",
         "basis": "IFRS", "total_assets": 121.0},
    ])


def test_pit_asof_and_lookahead_invariance():
    long = _two_year_long()
    reb = pd.to_datetime(["2025-01-31", "2025-12-31"])
    pan = fu.point_in_time(long, reb, ["total_assets"], date_col="DiscDate",
                           code_col="Code", lag_days=1)["total_assets"]
    assert pan.loc["2025-01-31", "72030"] == 110.0     # FY2025 開示はまだ見えない
    assert pan.loc["2025-12-31", "72030"] == 121.0
    # 未来（2025-06-20）の値を改変しても ≤t（2025-01-31）の as-of は不変
    mut = long.copy()
    mut.loc[mut["DiscDate"] == "2025-06-20", "total_assets"] = 9999.0
    pan2 = fu.point_in_time(mut, reb, ["total_assets"], date_col="DiscDate",
                            code_col="Code", lag_days=1)["total_assets"]
    assert pan2.loc["2025-01-31", "72030"] == 110.0    # 先読み不変
    assert pan2.loc["2025-12-31", "72030"] == 9999.0


# --- 派生特徴量：YoY と移行/非連続 NaN -------------------------------------
def _row(code, year, basis, **kw):
    base = dict(Code=code, period_end=f"{year}-03-31", basis=basis,
                total_assets=100.0, shares_outstanding=10.0, cfo=10.0, cfi=-4.0,
                profit=8.0, gross_profit=30.0, operating_income=12.0, depreciation=5.0,
                net_sales=200.0, income_taxes=3.0, pretax_income=11.0,
                interest_debt=20.0, net_assets=60.0, rd_expense=6.0, equity=55.0)
    base.update(kw)
    return base


def test_derive_features_yoy_and_transition_nan():
    long = pd.DataFrame([
        # A: 3 年連続 IFRS（YoY 発火）
        _row("A", 2023, "IFRS", total_assets=100.0, shares_outstanding=10.0),
        _row("A", 2024, "IFRS", total_assets=110.0, shares_outstanding=10.0),
        _row("A", 2025, "IFRS", total_assets=121.0, shares_outstanding=11.0),
        # B: 2024 JGAAP → 2025 IFRS（移行年は YoY を NaN）
        _row("B", 2024, "JGAAP", total_assets=100.0),
        _row("B", 2025, "IFRS", total_assets=130.0),
        # C: 2023 → 2025（1 年欠落＝非連続 → YoY NaN）
        _row("C", 2023, "IFRS", total_assets=100.0),
        _row("C", 2025, "IFRS", total_assets=140.0),
    ])
    d = efa.derive_disclosure_features(long).set_index(["Code", "period_end"])
    # A: 連続年は計算される
    assert abs(d.loc[("A", "2024-03-31"), "asset_growth"] - 0.10) < 1e-9
    assert abs(d.loc[("A", "2025-03-31"), "net_share_issuance"] - 0.10) < 1e-9
    assert d.loc[("A", "2023-03-31"), "asset_growth"] != d.loc[("A", "2023-03-31"), "asset_growth"]  # NaN（前期なし）
    # B: 移行年は NaN
    assert np.isnan(d.loc[("B", "2025-03-31"), "asset_growth"])
    # C: 非連続年は NaN
    assert np.isnan(d.loc[("C", "2025-03-31"), "asset_growth"])
    # 比率：FCF=CFO+CFI、レバレッジ=debt/純資産、粗利益性=粗利益/総資産
    a25 = d.loc[("A", "2025-03-31")]
    assert a25["fcf"] == 6.0
    assert abs(a25["leverage"] - 20.0 / 60.0) < 1e-9
    assert abs(a25["gross_profitability"] - 30.0 / 121.0) < 1e-9


def test_add_basis_transition_flag():
    long = pd.DataFrame([_row("B", 2024, "JGAAP"), _row("B", 2025, "IFRS")])
    tr = ef.add_basis_transition(long).set_index("period_end")
    assert bool(tr.loc["2024-03-31", "basis_changed"]) is False    # 初年度は False
    assert bool(tr.loc["2025-03-31", "basis_changed"]) is True


# --- cross_sectional_rank（[-1,1] ランク） ---------------------------------
def test_cross_sectional_rank():
    df = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0], "d": [4.0]},
                      index=pd.to_datetime(["2025-01-31"]))
    r = fac.cross_sectional_rank(df).iloc[0]
    assert abs(r["a"] - (-1.0)) < 1e-9 and abs(r["d"] - 1.0) < 1e-9
    assert abs(r["b"] - (-1 / 3)) < 1e-9
    # 欠損は保持、有効 1 銘柄の行は NaN（順序が定義できない）
    df2 = pd.DataFrame({"a": [5.0], "b": [np.nan]}, index=df.index)
    r2 = fac.cross_sectional_rank(df2).iloc[0]
    assert np.isnan(r2["b"]) and np.isnan(r2["a"])


# --- コードクロスウォーク（secCode=Code 5 桁）＋ build 統合（オフライン） ----
def test_build_edinet_long_crosswalk(tmp_path, monkeypatch):
    monkeypatch.setattr(ed, "_CACHE", tmp_path)
    (tmp_path / "list").mkdir()
    (tmp_path / "docs").mkdir()
    # 一覧ミラー（有報 1 件・secCode=72030・提出日 2025-06-20）
    rec = [{"docID": "S100TEST", "secCode": "72030", "docTypeCode": "120",
            "csvFlag": "1", "submitDateTime": "2025-06-20 15:00",
            "periodEnd": "2025-03-31", "ordinanceCode": "010"}]
    ed._save_list(ed.parse_documents(rec), tmp_path / "list" / "20250620.parquet")
    # 本体（IFRS の数値）
    _make_type5_zip(tmp_path / "docs" / "S100TEST_5.zip", [
        ("jpigp_cor:RevenueIFRS", "CurrentYearDuration", "200"),
        ("jpigp_cor:AssetsIFRS", "CurrentYearInstant", "1000"),
    ])
    long = ef.build_edinet_long(rebuild=True,
                                cache_path=tmp_path / "fund_long.parquet")
    assert len(long) == 1
    r = long.iloc[0]
    assert r["Code"] == "72030"          # secCode をそのまま Code に（5 桁恒等）
    assert r["basis"] == "IFRS"
    assert r["net_sales"] == 200.0 and r["total_assets"] == 1000.0
    assert pd.Timestamp(r["DiscDate"]) == pd.Timestamp("2025-06-20 15:00")
