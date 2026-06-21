"""EDINET 三表ファンダ：パース／要素IDマッピング／PIT統合／特徴量（ネットワーク不要）。

DL 済みの zip には依存せず、合成フィクスチャで純関数を検証する（CI 安全・決定的）。
handoff §6 受け入れ基準：①先読み不変②被覆/基準別null④移行断絶NaN⑤コードクロスウォーク。
"""
import zipfile

import numpy as np
import pandas as pd
import pytest

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


def test_net_share_issuance_split_adjusted():
    # 2:1 分割で生株数 10→20。split_cf(t-1)=0.5, split_cf(t)=1.0 → 調整株数 20→20 → 純発行≈0。
    long = pd.DataFrame([_row("A", 2024, "IFRS", shares_outstanding=10.0),
                         _row("A", 2025, "IFRS", shares_outstanding=20.0)])
    long["split_cf"] = [0.5, 1.0]
    d = efa.derive_disclosure_features(long).set_index("period_end")
    assert abs(d.loc["2025-03-31", "net_share_issuance"]) < 1e-9       # 純粋な分割は 0
    # split_cf 無し（raw）なら +100% 発行のアーティファクトが出る（修正前の挙動）。
    d2 = efa.derive_disclosure_features(long.drop(columns=["split_cf"])
                                        ).set_index("period_end")
    assert abs(d2.loc["2025-03-31", "net_share_issuance"] - 1.0) < 1e-9


def test_net_share_issuance_isolates_real_issuance_from_split():
    # 2:1 分割 ＋ 実 10% 発行：生 10→22、調整 20→22 → 純発行 +10%（分割だけ相殺）。
    long = pd.DataFrame([_row("A", 2024, "IFRS", shares_outstanding=10.0),
                         _row("A", 2025, "IFRS", shares_outstanding=22.0)])
    long["split_cf"] = [0.5, 1.0]
    d = efa.derive_disclosure_features(long).set_index("period_end")
    assert abs(d.loc["2025-03-31", "net_share_issuance"] - 0.10) < 1e-9


def test_attach_split_cf_asof_lookup(monkeypatch):
    # AdjC/C を as-of(≤period_end) で引いて split_cf を付与（store 価格 wide をモック）。
    import invest_system.data.store as store
    dates = pd.to_datetime(["2024-03-29", "2025-03-31"])
    panels = {"AdjC": pd.DataFrame({"A": [50.0, 100.0]}, index=dates),
              "C": pd.DataFrame({"A": [100.0, 100.0]}, index=dates)}
    monkeypatch.setattr(store, "load_wide", lambda field, base=None: panels[field])
    long = pd.DataFrame([_row("A", 2024, "IFRS"), _row("A", 2025, "IFRS")])
    cf = efa.attach_split_cf(long).set_index("period_end")["split_cf"]
    assert abs(cf.loc["2024-03-31"] - 0.5) < 1e-9     # 2024-03-29 の 50/100=0.5 を as-of
    assert abs(cf.loc["2025-03-31"] - 1.0) < 1e-9


def test_attach_split_cf_no_price_is_noop(monkeypatch):
    # 価格 wide が無ければ long をそのまま返す（raw 株数＝後方互換）。
    import invest_system.data.store as store
    monkeypatch.setattr(store, "load_wide", lambda field, base=None: pd.DataFrame())
    long = pd.DataFrame([_row("A", 2024, "IFRS"), _row("A", 2025, "IFRS")])
    out = efa.attach_split_cf(long)
    assert "split_cf" not in out.columns and len(out) == 2


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


# --- チェックポイント／再開／冪等／原子性／後方互換（材化の堅牢化） ----------
def _corpus(tmp_path, monkeypatch, n: int = 6):
    """N 件の有報 zip ＋ by-date 一覧ミラーを tmp に作り、ed._CACHE を差し替える。

    偶数 index だけ EquityIFRS（=net_assets）を持たせ、一部の正準フィールドが None の行を
    混ぜる＝チェックポイント刻みで dtype が割れないこと（一括 vs 分割同一性）を実検証する。
    返り値は長形式キャッシュのパス（tmp/fund_long.parquet）。
    """
    monkeypatch.setattr(ed, "_CACHE", tmp_path)
    (tmp_path / "list").mkdir()
    (tmp_path / "docs").mkdir()
    recs = []
    for i in range(n):
        did = f"S10{i:05d}"
        recs.append({"docID": did, "secCode": f"{72030 + i}", "docTypeCode": "120",
                     "csvFlag": "1", "submitDateTime": f"2025-06-{10 + i:02d} 15:00",
                     "periodEnd": f"{2020 + i}-03-31", "ordinanceCode": "010"})
        rows = [("jpigp_cor:RevenueIFRS", "CurrentYearDuration", str(100 + i)),
                ("jpigp_cor:AssetsIFRS", "CurrentYearInstant", str(1000 + i))]
        if i % 2 == 0:                            # 偶数のみ EquityIFRS（=net_assets）
            rows.append(("jpigp_cor:EquityIFRS", "CurrentYearInstant", str(500 + i)))
        _make_type5_zip(tmp_path / "docs" / f"{did}_5.zip", rows)
    ed._save_list(ed.parse_documents(recs), tmp_path / "list" / "20250610.parquet")
    return tmp_path / "fund_long.parquet"


def test_build_bulk_vs_split_identical(tmp_path, monkeypatch):
    # 同じ corpus を「実質一括」と「2 件刻み」で材化 → 最終 long が完全一致（行・列・dtype）。
    _corpus(tmp_path, monkeypatch, n=6)
    bulk = ef.build_edinet_long(rebuild=True, cache_path=tmp_path / "bulk.parquet",
                                checkpoint_every=10**9)
    split = ef.build_edinet_long(rebuild=True, cache_path=tmp_path / "split.parquet",
                                 checkpoint_every=2)
    pd.testing.assert_frame_equal(bulk, split)
    assert set(bulk["docID"]) == {f"S10{i:05d}" for i in range(6)}


def test_build_resume_from_partial(tmp_path, monkeypatch):
    # 途中まで書いたキャッシュ → 増分実行で既処理を再パースせず残りだけ追加・最終は一括と一致。
    _corpus(tmp_path, monkeypatch, n=6)
    full = ef.build_edinet_long(rebuild=True, cache_path=tmp_path / "full.parquet",
                                checkpoint_every=10**9)
    head_ids = sorted(full["docID"])[:3]          # 先頭 3 件だけ書いた「中断状態」を模倣
    cp = tmp_path / "resume.parquet"
    full[full["docID"].isin(head_ids)].reset_index(drop=True).to_parquet(cp)

    calls = []                                    # extract_canonical の呼び出し回数を計測
    orig = tax.extract_canonical
    monkeypatch.setattr(tax, "extract_canonical",
                        lambda f: (calls.append(1), orig(f))[1])
    resumed = ef.build_edinet_long(cache_path=cp, checkpoint_every=2)

    assert len(calls) == 3                         # 既処理 3 件は再パースせず残り 3 件のみ
    fa = full.sort_values("docID").reset_index(drop=True)
    ra = resumed.sort_values("docID").reset_index(drop=True)
    assert list(ra.columns) == list(fa.columns)    # 列一致
    assert set(ra["docID"]) == set(fa["docID"])    # docID 集合一致
    assert len(ra) == len(fa) == 6                 # 行数一致
    for col in ["Code", "basis", "net_sales", "total_assets"]:
        pd.testing.assert_series_equal(ra[col], fa[col], check_dtype=False,
                                       check_names=False)


def test_build_idempotent_rerun(tmp_path, monkeypatch):
    # 完了後の再実行は +0 件・再パースなし・結果不変。
    cp = _corpus(tmp_path, monkeypatch, n=6)
    first = ef.build_edinet_long(cache_path=cp, checkpoint_every=2)
    calls = []
    orig = tax.extract_canonical
    monkeypatch.setattr(tax, "extract_canonical",
                        lambda f: (calls.append(1), orig(f))[1])
    second = ef.build_edinet_long(cache_path=cp, checkpoint_every=2)
    assert calls == []                             # 再パースなし
    assert set(second["docID"]) == set(first["docID"])
    assert len(second) == len(first) == 6
    assert list(second.columns) == list(first.columns)


def test_atomic_write_parquet_replaces_and_leaves_no_temp(tmp_path):
    # temp→os.replace：既存を原子的に置換し、一時ファイルを残さない。
    p = tmp_path / "x.parquet"
    ef._atomic_write_parquet(pd.DataFrame({"a": [1, 2]}), p)
    ef._atomic_write_parquet(pd.DataFrame({"a": [1, 2, 3]}), p)
    assert len(pd.read_parquet(p)) == 3
    assert not list(tmp_path.glob("*.tmp.*"))


def test_build_checkpoint_main_intact_on_write_failure(tmp_path, monkeypatch):
    # 2 回目の保存中に「クラッシュ」しても本体は直近チェックポイント（先頭 2 件）のまま読める。
    cp = _corpus(tmp_path, monkeypatch, n=6)
    state = {"n": 0}
    orig = ef._atomic_write_parquet

    def flaky(df, path):
        state["n"] += 1
        if state["n"] == 2:
            raise RuntimeError("boom")
        orig(df, path)

    monkeypatch.setattr(ef, "_atomic_write_parquet", flaky)
    with pytest.raises(RuntimeError):
        ef.build_edinet_long(rebuild=True, cache_path=cp, checkpoint_every=2)
    got = pd.read_parquet(cp)                       # 本体は破損せず読める
    assert len(got) == 2                            # 1 回目チェックポイントぶんが保全


def test_build_backcompat_signatures(tmp_path, monkeypatch):
    # 既存呼び出し（引数なし／verbose=True）がそのまま動く。
    cp = _corpus(tmp_path, monkeypatch, n=4)
    a = ef.build_edinet_long(cache_path=cp)
    b = ef.build_edinet_long(cache_path=cp, verbose=True)   # 再実行＝+0 件
    assert len(a) == 4 and len(b) == 4
    assert set(a["docID"]) == set(b["docID"])


def test_panel_via_build_backcompat(tmp_path, monkeypatch):
    # edinet_fundamentals_panel() 経由（build を引数なしで呼ぶ）が従来どおり as-of を返す。
    _corpus(tmp_path, monkeypatch, n=4)
    monkeypatch.setattr(ef, "_LONG_CACHE", tmp_path / "fundamentals_long.parquet")
    reb = pd.to_datetime(["2025-12-31"])
    pan = ef.edinet_fundamentals_panel(reb, ["net_sales", "total_assets"])
    assert set(pan) == {"net_sales", "total_assets"}
    assert pan["net_sales"].loc["2025-12-31"].notna().sum() == 4   # 4 銘柄が as-of で可視
