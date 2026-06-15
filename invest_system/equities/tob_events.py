"""EDINET 公開買付（TOB）案件テーブルの構築（C1 リスクアーブの一次データ）。

2 層構成：
(A) **チェーン集約**（一覧メタデータのみ・ネットワーク不要）：deal_id = 公開買付届出書
    (240) の docID。子書類（250 訂正・260 撤回・270 報告・290/300 意見・310 QA）を
    parentDocID で 240 に集約し案件テーブルを作る。第三者TOB（府令040）のみ＝自社株TOB
    （050）は対象外（docs/07 C1）。
(B) **本体抽出**（書類 CSV・要素 ID は実測確定 docs/08 §7.1）：
    240/250 (jptoo-ton_cor): 価格 PriceOfPurchaseEtcTextBlock、届出当初期間
      OriginalPeriodAtFilingTextBlock、買付予定数 NumberOfShareCertificatesEtc...、
      買付予定割合(数値) RatioOf...ToBePurchased...。250 は訂正対象 docID を
      jpdei_cor:IdentificationOfDocumentSubjectToAmendmentDEI に持つ（parentDocID 欠損の
      補完）。240 と 250 の価格差＝バンプ、期間末日の差＝延長。
    270 (jptoo-tor_cor): 成否 SuccessOrFailureOfTenderOfferTextBlock、実期間
      TenderOfferPeriodTextBlock、買付数 NumberOfShareCertificatesEtcAcquiredByPurchaseEtc。

PIT 規律（docs/08 §8.2・docs/09 §3）：
- エントリーの PIT アンカーは 240 の submitDateTime（保守的な遅い側）。
- **当初価格・届出当初期間（240）はエントリー時点で既知**＝ルックアヘッド無し。
- 最終価格・実終了日・成否（250/270）は**結果側**＝列名 `_final`/`result_*` で区別。
  エントリー条件に結果側を使わない。
- 本体抽出は重い（案件あたり 240＋250×n＋270 をダウンロード）ので、build_deal_chains
  （構造）と enrich_deal（本体・オンデマンド）を分離する。
"""
from __future__ import annotations

import csv
import io
import re
import zipfile
from typing import Optional

import pandas as pd

from ..data.sources import edinet as ed

# --- 実測確定の要素 ID（docs/08 §7.1） -------------------------------------
NS_TON = "jptoo-ton_cor"      # 240 公開買付届出書 / 250 訂正届出書
NS_TOR = "jptoo-tor_cor"      # 270 公開買付報告書
E_PRICE = f"{NS_TON}:PriceOfPurchaseEtcTextBlock"
E_ORIG_PERIOD = f"{NS_TON}:OriginalPeriodAtFilingTextBlock"
E_PERIOD = f"{NS_TON}:PeriodOfPurchaseEtcTextBlock"
E_INTENDED = f"{NS_TON}:NumberOfShareCertificatesEtcIntendedToPurchaseTextBlock"
E_RATIO = (f"{NS_TON}:RatioOfNumberOfVotingRightsRepresentedByShareCertificatesEtc"
           "ToBePurchasedAmongNumberOfVotingRightsOwnedByAllShareholdersEtc"
           "OfSubjectCompany")
E_AMEND_TARGET = "jpdei_cor:IdentificationOfDocumentSubjectToAmendmentDEI"
E_RESULT = f"{NS_TOR}:SuccessOrFailureOfTenderOfferTextBlock"
E_TOR_PERIOD = f"{NS_TOR}:TenderOfferPeriodTextBlock"
E_ACQUIRED = f"{NS_TOR}:NumberOfShareCertificatesEtcAcquiredByPurchaseEtcTextBlock"

# 子書類の種別 → 役割
_CHILD_ROLE = {"250": "amend", "260": "withdraw", "270": "report",
               "290": "opinion", "300": "opinion_amend", "310": "qa"}

_PRICE_RE = re.compile(r"金\s*([0-9,]+)\s*円")
_DATE_RE = re.compile(r"(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日")
_Z2H = str.maketrans("０１２３４５６７８９，．", "0123456789,.")


# --- 純関数：CSV パース＆フィールド抽出（ネットワーク不要） -----------------
def _z2h(s: str) -> str:
    return str(s).translate(_Z2H)


def parse_xbrl_csv(text: str) -> dict:
    """EDINET XBRL→CSV（UTF-16 TSV・9 列）を {要素ID: 値} に。

    csv.reader で引用付き複数行 TextBlock も正しく 1 セルに収める。同一要素 ID が複数
    コンテキストで出る場合は最初の非空値を採用（－/空は後続で上書き可）。
    """
    out: dict = {}
    reader = csv.reader(io.StringIO(text), delimiter="\t")
    next(reader, None)                                  # ヘッダ
    for row in reader:
        if len(row) < 2:
            continue
        eid = row[0].strip()
        val = (row[8] if len(row) >= 9 else row[-1]).strip()
        if not eid:
            continue
        if eid not in out or out[eid] in ("", "－", "-"):
            out[eid] = val
    return out


def extract_price(text: Optional[str]) -> Optional[int]:
    """買付価格テキストブロックから普通株式の円価格（最初の「金…円」）を取り出す。"""
    if not text:
        return None
    m = _PRICE_RE.search(_z2h(text))
    return int(m.group(1).replace(",", "")) if m else None


def extract_dates(text: Optional[str]) -> list[str]:
    """テキスト中の和暦数字日付を ISO 文字列で全て返す（順序保持）。"""
    if not text:
        return []
    return [f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
            for y, mo, d in _DATE_RE.findall(_z2h(text))]


def _period(text: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """「期間{開始}から{終了}まで」の開始・終了を返す。

    届出当初期間ブロックは末尾に公告日など別の日付を含むため、最後の日付ではなく
    **先頭 2 つ**（開始・終了）を採る（実データで公告日の混入を確認・docs/08 §7.1）。
    """
    ds = extract_dates(text)
    return (ds[0] if ds else None, ds[1] if len(ds) >= 2 else None)


_TENDERED_RE = re.compile(r"応募株券等の(?:総数|数の合計)[^\d]*([0-9,]+)\s*株")
_LOWER_RE = re.compile(r"下限[^\d]*([0-9,]+)\s*株")


def _num(m) -> Optional[int]:
    return int(m.group(1).replace(",", "")) if m else None


def _result_from_text(text: Optional[str]) -> Optional[str]:
    """成否を判定。明示語（成立/不成立）優先、無ければ応募総数 vs 下限の数値比較。

    実データの成立文は「成立」を含まず「応募総数(X株)が下限(Y株)以上となり」と数値で
    述べることがあるため、語が無い場合は X≥Y を成立とする（docs/08 §7.1）。
    """
    if not text:
        return None
    z = _z2h(text)
    if "不成立" in z:
        return "不成立"
    if "成立" in z:
        return "成立"
    sub, low = _num(_TENDERED_RE.search(z)), _num(_LOWER_RE.search(z))
    if sub is not None and low is not None:
        return "成立" if sub >= low else "不成立"
    return None


def fields_240(rec: dict) -> dict:
    """240/250 の届出本文から当初条件を抽出（rec=parse_xbrl_csv の出力）。"""
    price_raw = rec.get(E_PRICE)
    orig = rec.get(E_ORIG_PERIOD) or rec.get(E_PERIOD)
    p_start, p_end = _period(orig)
    ratio = rec.get(E_RATIO)
    return {
        "initial_price": extract_price(price_raw),
        "period_start": p_start,
        "period_end": p_end,
        "purchase_ratio_pct": pd.to_numeric(_z2h(ratio), errors="coerce")
        if ratio else None,
        "price_raw": price_raw,
        "intended_raw": rec.get(E_INTENDED),
    }


def fields_250(rec: dict) -> dict:
    """250 訂正の現行価格・期間と訂正対象 docID（parentDocID 欠損の補完）。"""
    _, p_end = _period(rec.get(E_PERIOD) or rec.get(E_ORIG_PERIOD))
    return {
        "amends_doc_id": rec.get(E_AMEND_TARGET),
        "price": extract_price(rec.get(E_PRICE)),
        "period_end": p_end,
    }


def fields_270(rec: dict) -> dict:
    """270 報告から成否・実期間を抽出。"""
    a_start, a_end = _period(rec.get(E_TOR_PERIOD))
    res_raw = rec.get(E_RESULT)
    z = _z2h(res_raw) if res_raw else ""
    return {
        "result": _result_from_text(res_raw),
        "tendered_shares": _num(_TENDERED_RE.search(z)) if z else None,
        "lower_bound_shares": _num(_LOWER_RE.search(z)) if z else None,
        "result_raw": res_raw,
        "actual_period_start": a_start,
        "actual_period_end": a_end,
        "acquired_raw": rec.get(E_ACQUIRED),
    }


# --- (A) チェーン集約（一覧メタデータのみ） --------------------------------
def build_deal_chains(list_df: pd.DataFrame, competing_window_days: int = 180
                      ) -> pd.DataFrame:
    """第三者TOB（府令040）の案件テーブル（1 行 = 1 案件＝240 起点）。

    列：deal_id, announce_dt, acquirer_name/edinet/sec, target_edinet/sec,
    n_amend(250), has_withdrawal(260), has_report(270), n_opinion(290+300),
    competing。target_sec は意見表明(290＝対象者が提出)の secCode から復元
    （240 の secCode は買収者側）。本体抽出は enrich_deal で別途。
    """
    tob = list_df[list_df["ordinanceCode"] == "040"]
    notices = tob[tob["docTypeCode"] == "240"]
    rows = []
    for _, n in notices.iterrows():
        did = n["docID"]
        kids = tob[tob["parentDocID"] == did]
        roles = kids["docTypeCode"].map(_CHILD_ROLE)
        opinions = kids[kids["docTypeCode"].isin(["290", "300"])]
        tsec = opinions["secCode"].dropna()
        rows.append({
            "deal_id": did,
            "announce_dt": n["submitDateTime"],
            "acquirer_name": n.get("filerName"),
            "acquirer_edinet": n.get("edinetCode"),
            "acquirer_sec": n.get("secCode"),
            "target_edinet": n.get("subjectEdinetCode"),
            "target_sec": tsec.iloc[0] if len(tsec) else pd.NA,
            "n_amend": int((roles == "amend").sum()),
            "has_withdrawal": bool((roles == "withdraw").any()),
            "has_report": bool((roles == "report").any()),
            "n_opinion": int(kids["docTypeCode"].isin(["290", "300"]).sum()),
        })
    deals = pd.DataFrame(rows, columns=_DEAL_COLS)       # 空でも列を保つ
    return _mark_competing(deals, competing_window_days)


_DEAL_COLS = ["deal_id", "announce_dt", "acquirer_name", "acquirer_edinet",
              "acquirer_sec", "target_edinet", "target_sec", "n_amend",
              "has_withdrawal", "has_report", "n_opinion"]


def _mark_competing(deals: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """同一対象（target_edinet）に発表が近接する複数案件を競合フラグ（ソレキア型）。"""
    deals = deals.copy()
    deals["competing"] = False
    if "target_edinet" not in deals or deals.empty:
        return deals
    win = pd.Timedelta(days=window_days)
    for tgt, g in deals.groupby("target_edinet"):
        if pd.isna(tgt) or len(g) < 2:
            continue
        dts = pd.to_datetime(g["announce_dt"])
        if (dts.max() - dts.min()) <= win:
            deals.loc[g.index, "competing"] = True
    return deals


# --- (B) 本体抽出（オンデマンド・ネットワーク） ----------------------------
def _decode(raw: bytes) -> str:
    for enc in ("utf-16", "utf-16-le", "cp932", "utf-8-sig", "utf-8"):
        try:
            return raw.decode(enc)
        except Exception:                               # noqa: BLE001
            continue
    return raw.decode("utf-8", "ignore")


def doc_csv_records(doc_id: str, api_key: Optional[str] = None,
                    refresh: bool = False) -> dict:
    """書類本体（type=5 CSV）を取得し、TOB 本文 CSV（jptoo…）を {要素ID:値} で返す。"""
    path = ed.fetch_document(doc_id, doc_type=5, api_key=api_key, refresh=refresh)
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not names:
            return {}
        pick = next((n for n in names if "jptoo" in n.lower()), names[0])
        return parse_xbrl_csv(_decode(z.read(pick)))


def enrich_deal(deal: pd.Series | dict, list_df: pd.DataFrame,
                api_key: Optional[str] = None) -> dict:
    """1 案件の本体を取得し当初/結果/バンプを抽出（ネットワーク・案件あたり数 DL）。

    240 から当初価格・届出当初期間、各 250 から訂正後価格/期間（提出順に走査して
    バンプ・延長を検出）、270 から成否・実終了日。final_price は連鎖の最終価格。
    """
    deal_id = deal["deal_id"] if not isinstance(deal, dict) else deal["deal_id"]
    tob = list_df[list_df["ordinanceCode"] == "040"]
    kids = tob[tob["parentDocID"] == deal_id].sort_values("submitDateTime")

    def safe(doc_id):
        """本体取得・パースに失敗しても 1 案件で全体を落とさない（古い csvFlag=0 等）。"""
        try:
            return doc_csv_records(doc_id, api_key)
        except Exception:                               # noqa: BLE001
            return {}

    rec0 = safe(deal_id)
    base = fields_240(rec0)
    body_ok = bool(rec0)
    initial_price = base["initial_price"]
    final_price = initial_price
    final_period_end = base["period_end"]
    n_bumps = n_extends = 0
    prev_price, prev_end = initial_price, base["period_end"]

    for _, k in kids[kids["docTypeCode"] == "250"].iterrows():
        f = fields_250(safe(k["docID"]))
        if f["price"] is not None and prev_price is not None and f["price"] != prev_price:
            n_bumps += 1
            prev_price = f["price"]
            final_price = f["price"]
        if f["period_end"] and prev_end and f["period_end"] != prev_end:
            n_extends += 1
            prev_end = f["period_end"]
            final_period_end = f["period_end"]

    result = None
    actual_end = None
    reps = kids[kids["docTypeCode"] == "270"]
    if len(reps):
        rf = fields_270(safe(reps.iloc[-1]["docID"]))
        result = rf["result"]
        actual_end = rf["actual_period_end"]

    return {
        "deal_id": deal_id,
        "initial_price": initial_price,
        "final_price": final_price,
        "purchase_ratio_pct": base["purchase_ratio_pct"],
        "period_start": base["period_start"],
        "period_end_initial": base["period_end"],
        "period_end_final": final_period_end or actual_end,
        "n_bumps": n_bumps,
        "n_extensions": n_extends,
        "result": result,
        "withdrawn": bool((kids["docTypeCode"] == "260").any()),
        "body_ok": body_ok,
    }
