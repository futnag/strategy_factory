"""有報XBRL→所有者別状況 オフライン抽出（parse_ownership_from_xbrl）の検証。合成データ・ネット不要。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from invest_system.equities.edinet_ownership import parse_ownership_from_xbrl  # noqa: E402

U = "jpcrp_cor:NumberOfSharesHeldNumberOfUnits"
P = "jpcrp_cor:PercentageOfShareholdings"


def _df(rows):
    return pd.DataFrame(rows, columns=["element_id", "rel_year", "value"])


def test_basic_units_and_pct_crosscheck():
    df = _df([
        (U + "ForeignInvestorsOtherThanIndividuals", "当期末", "6010"),
        (U + "ForeignIndividualInvestors", "当期末", "106"),
        (U + "IndividualsAndOthers", "当期末", "108692"),
        (U + "Total", "当期末", "130388"),
        (P + "ForeignersOtherThanIndividuals", "当期末", "0.0461"),
        (P + "ForeignIndividuals", "当期末", "0.0008"),
        (P + "IndividualsAndOthers", "当期末", "0.8336"),
    ])
    r = parse_ownership_from_xbrl(df)
    assert r["quality_flag"] == "ok"
    assert abs(r["foreign_pct"] - 4.6906) < 0.01
    assert abs(r["individual_pct"] - 83.36) < 0.01
    assert r["units_total"] == 130388.0


def test_dash_treated_as_absent():
    df = _df([
        (U + "ForeignInvestorsOtherThanIndividuals", "当期末", "6010"),
        (U + "ForeignIndividualInvestors", "当期末", "－"),     # 全角ダッシュ＝該当なし
        (U + "IndividualsAndOthers", "当期末", "108692"),
        (U + "Total", "当期末", "130388"),
    ])
    r = parse_ownership_from_xbrl(df)
    assert abs(r["foreign_pct"] - 100.0 * 6010 / 130388) < 0.01   # 個人外国は0扱い
    assert r["quality_flag"] == "ok"


def test_multi_share_class_summed():
    df = _df([
        (U + "ForeignInvestorsOtherThanIndividuals", "当期末", "4000"),   # 普通株
        (U + "ForeignInvestorsOtherThanIndividuals", "当期末", "2010"),   # 種類株（合算対象）
        (U + "Total", "当期末", "60000"),
        (U + "Total", "当期末", "70388"),
    ])
    r = parse_ownership_from_xbrl(df)
    assert r["units_total"] == 130388.0
    assert abs(r["foreign_pct"] - 100.0 * 6010 / 130388) < 0.01


def test_prior_year_ignored():
    df = _df([
        (U + "ForeignInvestorsOtherThanIndividuals", "当期末", "6010"),
        (U + "Total", "当期末", "130388"),
        (U + "ForeignInvestorsOtherThanIndividuals", "前期末", "999999"),  # 前期＝無視
        (U + "Total", "前期末", "999999"),
    ])
    r = parse_ownership_from_xbrl(df)
    assert abs(r["foreign_pct"] - 100.0 * 6010 / 130388) < 0.01


def test_pct_only_fallback_when_total_missing():
    df = _df([
        (P + "ForeignersOtherThanIndividuals", "当期末", "0.20"),
        (P + "ForeignIndividuals", "当期末", "0.01"),
        (P + "IndividualsAndOthers", "当期末", "0.50"),
    ])
    r = parse_ownership_from_xbrl(df)
    assert r["quality_flag"] == "pct_only"
    assert abs(r["foreign_pct"] - 21.0) < 0.01
    assert abs(r["individual_pct"] - 50.0) < 0.01


def test_empty_or_no_table():
    assert parse_ownership_from_xbrl(pd.DataFrame()) ["foreign_pct"] is None
    df = _df([("jpcrp_cor:NetSales", "当期末", "123")])
    assert parse_ownership_from_xbrl(df)["quality_flag"] == "missing"
