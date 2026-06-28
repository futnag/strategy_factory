"""有報 XBRL（EDINET type=5 CSV）から**所有者別状況**をオフライン抽出（MCP/API 非依存・CI互換）。

実データで確認（docs/50 §8・実例 docID S100X6CU/6264＝外国4.69%・個人83.36%）：所有者別状況は
**専用 element_id** で格納され、カテゴリは要素名に内包される（dimension 解析不要）。

- 単元数: `jpcrp_cor:NumberOfSharesHeldNumberOfUnits{<category>}`
- 割合  : `jpcrp_cor:PercentageOfShareholdings{<category>}`（0–1 の分数。要素名サフィックスが
          単元側と一部異なる＝外国は "Foreigners…"/"ForeignIndividuals"）
- 文脈  : `rel_year` が当期、context は株式クラス member（複数クラスは単元を合算・Total が分母）

`foreign_pct`/`individual_pct` は **単元比（×100＝%）** を主とし、割合要素で交差検証する。
入力は `edinet.read_xbrl_csv(zip)` の DataFrame（列: element_id, rel_year, value 他）。純関数。
"""
from __future__ import annotations

import pandas as pd

# 正準カテゴリ ← element_id サフィックス（**優先順で**判定。Foreign を Individual より先に。
# "ForeignInvestorsOtherThanIndividuals" は "Individuals" を含むので最優先で潰す）。
_CAT_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("foreign_nonindiv", ("ForeignInvestorsOtherThanIndividuals", "ForeignersOtherThanIndividuals")),
    ("foreign_indiv", ("ForeignIndividualInvestors", "ForeignIndividuals")),
    ("individual", ("IndividualsAndOthers",)),
    ("total", ("Total",)),
    ("gov", ("NationalAndLocalGovernments",)),
    ("fin", ("FinancialInstitutions",)),
    ("secco", ("FinancialServiceProviders",)),
    ("othercorp", ("OtherCorporations",)),
]
_UNITS_TAG = "NumberOfSharesHeldNumberOfUnits"
_PCT_TAG = "PercentageOfShareholdings"


def _category(element_id: str) -> str | None:
    for canon, subs in _CAT_RULES:
        if any(s in element_id for s in subs):
            return canon
    return None


def _num(v) -> float | None:
    """値セル → float。'－'/'-'/空/非数値は None。カンマ除去。"""
    s = str(v).strip().replace(",", "")
    if s in ("", "－", "-", "—", "null", "None", "nan", "NaN"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_ownership_from_xbrl(df: pd.DataFrame) -> dict:
    """read_xbrl_csv の DataFrame → 当期の所有者別状況レコード（純関数）。

    返り値: {foreign_pct, individual_pct, units_total, n_units_rows, quality_flag, ...各カテゴリ単元}。
    quality_flag: ok / check（単元と割合が乖離>0.5pt）/ pct_only（単元欠で割合採用）/ missing（不能）。
    """
    out = {"foreign_pct": None, "individual_pct": None, "units_total": None,
           "n_units_rows": 0, "quality_flag": "missing"}
    if df is None or df.empty or "element_id" not in df.columns:
        return out
    eid = df["element_id"].astype(str)
    rel = df["rel_year"].astype(str) if "rel_year" in df.columns else pd.Series("", index=df.index)
    cur = df[(rel.str.contains("当期", na=False)) &
             (eid.str.contains(_UNITS_TAG, na=False) | eid.str.contains(_PCT_TAG, na=False))]
    units: dict[str, float] = {}
    pct: dict[str, float] = {}
    for _, r in cur.iterrows():
        e = str(r["element_id"])
        c = _category(e)
        if c is None:
            continue
        val = _num(r["value"])
        if val is None:
            continue
        if _UNITS_TAG in e:
            units[c] = units.get(c, 0.0) + val          # 複数株式クラスは合算
        elif _PCT_TAG in e:
            pct[c] = pct.get(c, 0.0) + val
    out["n_units_rows"] = len([k for k in units if k != "total"])
    for k, v in units.items():
        out[f"u_{k}"] = v
    total = units.get("total")

    def _pct_from_units():
        fp = 100.0 * (units.get("foreign_nonindiv", 0.0) + units.get("foreign_indiv", 0.0)) / total
        ip = 100.0 * units.get("individual", 0.0) / total
        return fp, ip

    if total and total > 0:
        fp, ip = _pct_from_units()
        out.update(foreign_pct=round(fp, 4), individual_pct=round(ip, 4),
                   units_total=total, quality_flag="ok")
        if pct:  # 割合要素で交差検証（割合は 0–1 分数 → ×100）
            fp_pct = 100.0 * (pct.get("foreign_nonindiv", 0.0) + pct.get("foreign_indiv", 0.0))
            if abs(fp - fp_pct) > 0.5:
                out["quality_flag"] = "check"
    elif pct:  # 単元/Total 欠 → 割合要素にフォールバック
        out.update(foreign_pct=round(100.0 * (pct.get("foreign_nonindiv", 0.0)
                                              + pct.get("foreign_indiv", 0.0)), 4),
                   individual_pct=round(100.0 * pct.get("individual", 0.0), 4),
                   quality_flag="pct_only")
    return out
