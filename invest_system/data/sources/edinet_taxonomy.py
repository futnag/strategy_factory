"""EDINET type=5 CSV の要素ID → 正準フィールドのマッピング（会計基準別）。

有報 type=5 CSV は (要素ID, コンテキストID, 値, …) の長形式（edinet.read_xbrl_csv）。
要素IDの名前空間 prefix が会計基準を表す:
  jppfs_cor … 日本基準（財務諸表本表）
  jpigp_cor … IFRS（要素IDは "IFRS" サフィックス付き）
  jpcrp_cor … 共通開示（主要な経営指標等 SummaryOfBusinessResults・研究開発費 等）
  jpdei_cor … 書類・提出者情報
  jpcrp{nnnnnn}-asr_{EDINETコード} … 銘柄固有の拡張（例: トヨタの売上 TotalNetRevenuesIFRS）

規律（handoff §2,§4）:
- 正準フィールドは **生ライン項目のみ**（加工済み比率は持たない＝factors 側で自前計算）。
- コンテキストは **当期・連結**（CurrentYearDuration/Instant・メンバー無し）を採る。
- 要素IDは年度・企業でゆれるため候補を **優先リスト**で持ち、先に当たったものを採用。
- 銘柄固有拡張に備え、売上等は **接尾辞フォールバック**も用意（J-Quants 突合でバグ検出）。
- IFRS/JGAAP/US-GAAP で取れないフィールドは None（基準別 null は呼び出し側で扱う）。
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .edinet import CURRENT_CONSOLIDATED_CTX

IFRS, JGAAP, USGAAP = "IFRS", "JGAAP", "USGAAP"

# 親会社単体（NonConsolidated）も許す当期コンテキスト（発行株数・資本金など全社レベル項目用）
_NONCONS_CTX = ("CurrentYearDuration_NonConsolidatedMember",
                "CurrentYearInstant_NonConsolidatedMember")

# (正準フィールド) -> (基準) -> 候補要素ID（優先順）。COMMON は基準横断（jpcrp_cor 等）。
FIELD_MAP: dict[str, dict[str, list[str]]] = {
    "net_sales": {
        IFRS: ["jpigp_cor:RevenueIFRS", "jpigp_cor:NetSalesIFRS",
               "jpigp_cor:SalesRevenuesIFRS", "jpigp_cor:OperatingRevenuesIFRS",
               "jpigp_cor:RevenueFromContractsWithCustomersIFRS",
               "jpcrp_cor:RevenueIFRSSummaryOfBusinessResults",
               "jpcrp_cor:NetSalesIFRSSummaryOfBusinessResults"],
        JGAAP: ["jppfs_cor:NetSales", "jppfs_cor:NetSalesOfCompletedConstructionContracts",
                "jppfs_cor:OperatingRevenue1", "jppfs_cor:GrossOperatingRevenue",
                "jpcrp_cor:NetSalesSummaryOfBusinessResults"],
    },
    "operating_income": {
        IFRS: ["jpigp_cor:OperatingProfitLossIFRS"],
        JGAAP: ["jppfs_cor:OperatingIncome",
                "jpcrp_cor:OperatingIncomeSummaryOfBusinessResults"],
    },
    "ordinary_income": {
        JGAAP: ["jppfs_cor:OrdinaryIncome",
                "jpcrp_cor:OrdinaryIncomeLossSummaryOfBusinessResults"],
    },
    "pretax_income": {
        IFRS: ["jpigp_cor:ProfitLossBeforeTaxIFRS",
               "jpcrp_cor:ProfitLossBeforeTaxIFRSSummaryOfBusinessResults"],
        JGAAP: ["jppfs_cor:IncomeBeforeIncomeTaxes", "jppfs_cor:ProfitLossBeforeTax"],
    },
    "profit": {   # 親会社株主に帰属する当期純利益
        IFRS: ["jpigp_cor:ProfitLossAttributableToOwnersOfParentIFRS",
               "jpcrp_cor:ProfitLossAttributableToOwnersOfParentIFRSSummaryOfBusinessResults"],
        JGAAP: ["jppfs_cor:ProfitLossAttributableToOwnersOfParent",
                "jpcrp_cor:ProfitLossAttributableToOwnersOfParentSummaryOfBusinessResults"],
    },
    "profit_total": {  # 当期純利益（非支配持分含む）
        IFRS: ["jpigp_cor:ProfitLossIFRS"],
        JGAAP: ["jppfs_cor:ProfitLoss"],
    },
    "total_assets": {
        IFRS: ["jpigp_cor:AssetsIFRS",
               "jpcrp_cor:TotalAssetsIFRSSummaryOfBusinessResults"],
        JGAAP: ["jppfs_cor:Assets", "jpcrp_cor:TotalAssetsSummaryOfBusinessResults"],
    },
    "net_assets": {   # 純資産合計
        IFRS: ["jpigp_cor:EquityIFRS"],
        JGAAP: ["jppfs_cor:NetAssets", "jpcrp_cor:NetAssetsSummaryOfBusinessResults"],
    },
    "equity": {   # 親会社株主に帰属する持分（自己資本）
        IFRS: ["jpigp_cor:EquityAttributableToOwnersOfParentIFRS",
               "jpcrp_cor:EquityAttributableToOwnersOfParentIFRSSummaryOfBusinessResults"],
        JGAAP: ["jppfs_cor:ShareholdersEquity"],
    },
    "gross_profit": {
        IFRS: ["jpigp_cor:GrossProfitIFRS"],
        JGAAP: ["jppfs_cor:GrossProfit"],
    },
    "cfo": {
        IFRS: ["jpigp_cor:NetCashProvidedByUsedInOperatingActivitiesIFRS",
               "jpcrp_cor:CashFlowsFromUsedInOperatingActivitiesIFRSSummaryOfBusinessResults"],
        JGAAP: ["jppfs_cor:NetCashProvidedByUsedInOperatingActivities",
                "jpcrp_cor:NetCashProvidedByUsedInOperatingActivitiesSummaryOfBusinessResults"],
    },
    "cfi": {
        IFRS: ["jpigp_cor:NetCashProvidedByUsedInInvestingActivitiesIFRS",
               "jpcrp_cor:CashFlowsFromUsedInInvestingActivitiesIFRSSummaryOfBusinessResults"],
        # JGAAP は Investing / Investment の表記ゆれあり（例: 任天堂は Investment）。
        JGAAP: ["jppfs_cor:NetCashProvidedByUsedInInvestingActivities",
                "jppfs_cor:NetCashProvidedByUsedInInvestmentActivities",
                "jpcrp_cor:NetCashProvidedByUsedInInvestingActivitiesSummaryOfBusinessResults"],
    },
    "cff": {
        IFRS: ["jpigp_cor:NetCashProvidedByUsedInFinancingActivitiesIFRS"],
        JGAAP: ["jppfs_cor:NetCashProvidedByUsedInFinancingActivities",
                "jpcrp_cor:NetCashProvidedByUsedInFinancingActivitiesSummaryOfBusinessResults"],
    },
    "cash": {
        IFRS: ["jpigp_cor:CashAndCashEquivalentsIFRS",
               "jpcrp_cor:CashAndCashEquivalentsIFRSSummaryOfBusinessResults"],
        JGAAP: ["jppfs_cor:CashAndCashEquivalents",
                "jpcrp_cor:CashAndCashEquivalentsSummaryOfBusinessResults"],
    },
    "depreciation": {   # CF 計算書の減価償却費（EBITDA 用の加算項目）
        IFRS: ["jpigp_cor:DepreciationAndAmortizationOpeCFIFRS",
               "jpigp_cor:DepreciationAndAmortisationExpenseIFRS"],
        JGAAP: ["jppfs_cor:DepreciationAndAmortizationOpeCF"],
    },
    "income_taxes": {   # 法人税等合計（ROIC の NOPAT 算出に使用）
        IFRS: ["jpigp_cor:IncomeTaxExpenseIFRS"],
        JGAAP: ["jppfs_cor:IncomeTaxes"],
    },
}

# 有利子負債（当期・連結）。IFRS は二段構え：①全部入り集約タグ InterestBearingLiabilities
# があれば最優先（二重計上防止）。②無ければ標準個別タグ（BondsAndBorrowings / Borrowings /
# LongTermDebt / BondsPayable / LeaseLiabilities の CL・NCL）を広く合算する。企業は命名規約を
# 混在させない（BondsAndBorrowings 派 か Borrowings+LongTermDebt 派）ため実務上二重計上しない。
# 銘柄固有拡張の満期内訳（例：日立の CurrentPortionOfLongTermDebt）は拾えず軽度の過少あり。
IFRS_DEBT_AGG = ["jpigp_cor:InterestBearingLiabilitiesCLIFRS",
                 "jpigp_cor:InterestBearingLiabilitiesNCLIFRS"]
IFRS_DEBT_GRANULAR = [
    "jpigp_cor:BondsAndBorrowingsCLIFRS", "jpigp_cor:BondsAndBorrowingsNCLIFRS",
    "jpigp_cor:BorrowingsCLIFRS", "jpigp_cor:BorrowingsNCLIFRS",
    "jpigp_cor:LongTermDebtCLIFRS", "jpigp_cor:LongTermDebtNCLIFRS",
    "jpigp_cor:BondsPayableCLIFRS", "jpigp_cor:BondsPayableNCLIFRS",
    "jpigp_cor:LeaseLiabilitiesCLIFRS", "jpigp_cor:LeaseLiabilitiesNCLIFRS",
]
JGAAP_DEBT_COMPONENTS = [
    "jppfs_cor:ShortTermLoansPayable", "jppfs_cor:CurrentPortionOfLongTermLoansPayable",
    "jppfs_cor:CurrentPortionOfBondsPayable", "jppfs_cor:CommercialPapersLiabilities",
    "jppfs_cor:LeaseObligationsCL", "jppfs_cor:BondsPayable",
    "jppfs_cor:LongTermLoansPayable", "jppfs_cor:LeaseObligationsNCL",
]

# 基準横断（共通開示 jpcrp_cor）。基準に依らず探す。
FIELD_MAP_COMMON: dict[str, list[str]] = {
    "rd_expense": ["jpcrp_cor:ResearchAndDevelopmentExpensesResearchAndDevelopmentActivities"],
}

# 全社レベル（親会社単体コンテキストでも採る）。発行済株式数・資本金。
FIELD_MAP_ENTITY: dict[str, list[str]] = {
    "shares_outstanding": ["jpcrp_cor:TotalNumberOfIssuedSharesSummaryOfBusinessResults"],
}

# 銘柄固有拡張に備えた接尾辞フォールバック（売上はタグ揺れが激しい）。
FIELD_SUFFIX_FALLBACK: dict[str, dict[str, list[str]]] = {
    "net_sales": {
        IFRS: ["NetRevenuesIFRS", "SalesRevenuesIFRS", "OperatingRevenuesIFRS",
               "RevenueIFRS", "NetSalesIFRS"],
    },
}


def detect_basis(facts: pd.DataFrame) -> str:
    """当期・連結コンテキストの要素から会計基準を判定（IFRS/JGAAP/USGAAP）。

    IFRS: jpigp_cor または "IFRS" サフィックスの要素が連結に存在。
    JGAAP: 連結に jppfs_cor 本表要素が存在（親会社単体は NonConsolidated なので混ざらない）。
    それ以外（US-GAAP・本表が銘柄固有拡張のみ）は USGAAP として扱う。
    """
    cc = facts[facts["context_id"].isin(CURRENT_CONSOLIDATED_CTX)]
    eids = cc["element_id"].dropna()
    ns = set(cc["namespace"].dropna())
    if "jpigp_cor" in ns or eids.str.endswith("IFRS").any():
        return IFRS
    if "jppfs_cor" in ns:
        return JGAAP
    return USGAAP


def _by_eid(facts: pd.DataFrame, ctxs: tuple) -> pd.Series:
    """指定コンテキストの (要素ID -> 数値) 。同一要素は先頭値を採る。"""
    sub = facts[facts["context_id"].isin(ctxs)].dropna(subset=["value_num"])
    return sub.groupby("element_id")["value_num"].first()


def select_value(facts: pd.DataFrame, field: str,
                 basis: Optional[str] = None) -> tuple[Optional[float], Optional[str]]:
    """正準フィールドの当期・連結値を優先リストで取得。返り値 (値, 採用要素ID)。

    見つからなければ (None, None)。売上等は接尾辞フォールバック（最大絶対値）を試す。
    """
    basis = basis or detect_basis(facts)
    if field in FIELD_MAP_ENTITY:               # 全社レベルは単体コンテキストも許す
        by = _by_eid(facts, CURRENT_CONSOLIDATED_CTX + _NONCONS_CTX)
        for eid in FIELD_MAP_ENTITY[field]:
            if eid in by.index:
                return float(by[eid]), eid
        return None, None

    by = _by_eid(facts, CURRENT_CONSOLIDATED_CTX)
    candidates = list(FIELD_MAP.get(field, {}).get(basis, []))
    candidates += FIELD_MAP_COMMON.get(field, [])
    for eid in candidates:
        if eid in by.index:
            return float(by[eid]), eid
    for suf in FIELD_SUFFIX_FALLBACK.get(field, {}).get(basis, []):
        hits = by[by.index.str.endswith(suf)]
        if len(hits):
            eid = hits.abs().idxmax()
            return float(by[eid]), eid
    return None, None


def interest_debt(facts: pd.DataFrame, basis: Optional[str] = None) -> Optional[float]:
    """有利子負債合計（当期・連結）。構成要素の和。

    XBRL は値ゼロの科目を省くため、構成要素が一つも無い場合は BS（総資産）が在れば
    実質無借金として 0.0、BS も無ければ None（未取得）を返す。
    """
    basis = basis or detect_basis(facts)
    by = _by_eid(facts, CURRENT_CONSOLIDATED_CTX)
    if basis == IFRS:
        agg = [float(by[e]) for e in IFRS_DEBT_AGG if e in by.index]
        if agg:                           # ①全部入り集約タグ（二重計上防止で単独採用）
            return float(sum(agg))
        gran = [float(by[e]) for e in IFRS_DEBT_GRANULAR if e in by.index]
        return float(sum(gran)) if gran else None    # ②標準個別タグの和。無ければ None
    # JGAAP: 個別科目の和。BS が在って一つも無ければ実質無借金 0（None と区別）。
    vals = [float(by[e]) for e in JGAAP_DEBT_COMPONENTS if e in by.index]
    if vals:
        return float(sum(vals))
    ta, _ = select_value(facts, "total_assets", basis)
    return 0.0 if ta is not None else None


def extract_canonical(facts: pd.DataFrame) -> dict:
    """有報 type=5 facts → 正準フィールドの dict（当期・連結）。基準は 'basis' に。"""
    basis = detect_basis(facts)
    fields = (list(FIELD_MAP) + list(FIELD_MAP_COMMON) + list(FIELD_MAP_ENTITY))
    out: dict = {"basis": basis}
    for f in fields:
        out[f], _ = select_value(facts, f, basis)
    out["interest_debt"] = interest_debt(facts, basis)
    return out
