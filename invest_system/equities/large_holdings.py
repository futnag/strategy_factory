"""EDINET 大量保有報告（府令060）本体抽出と C2 ユニバース構築（保有目的フィルタ）。

C2 の母集合は **保有目的に「重要提案行為等」を含む大量保有報告**（docs/10 §7・
activists.is_important_proposal）。本体 CSV（type=5・jplvh_cor 名前空間）から保有目的・
保有割合・提出者を抽出し、activists.resolve でグループ／hard・soft タグを付ける。

新規/変更/訂正の峻別（実測確定 2026-06・docs/08 §7.1 と旧 HANDOFF を訂正）:
- **本体の DocumentTitleCoverPage が唯一信頼できる判別子**。docTypeCode 350 は新規・変更の
  両方を含み、formCode も信頼できない（formCode 010000 に「変更報告書No.12」が混入する実例
  あり＝末尾 0000=新規という当初仮説は不成立）。→ `report_class` はタイトルで判定する。
    タイトルに「訂正」を含む = correction / 「変更報告書」を含む = change /
    「大量保有報告書」のみ = new。
- 一般報告(01) と 特例報告(02/03) の別は formCode 先頭で判る（これは信頼できる）。特例報告は
  法第27条の26（基準日方式のまとめ報告）で **重要提案行為等の意図が無い機関投資家専用**＝
  定義上アクティビストは一般報告(01)に入る（実測: 特例 030000 の重要提案ヒット=0/25）。
  C2 新規参入母集団のスキャン対象は一般報告(formCode 先頭 01)に限定してよい。

PIT 規律（PIT 地雷 #1・docs/10 §1.1）:
- エントリーアンカーは一覧の submitDateTime（提出日）。本体の DateWhenFilingRequirementArose
  （報告義務発生日）は使わない＝特例報告で最大 148 日のラグ＝重大リーク。提出ラグは
  「特例報告か」の特徴量として保持する。
"""
from __future__ import annotations

import zipfile
from typing import Optional

import pandas as pd

from . import activists
from ..data.sources import edinet as ed
from .tob_events import _decode, _z2h, extract_dates, parse_xbrl_csv

NS_LVH = "jplvh_cor"
NS_DEI = "jpdei_cor"

# --- 実測確定の要素 ID（jplvh010000-lvh-001 / 第一号様式） -------------------
E_TITLE = f"{NS_LVH}:DocumentTitleCoverPage"                 # 大量保有報告書 / 変更報告書No.x
E_REASON_CHANGE = f"{NS_LVH}:ReasonForFilingChangeReportCoverPage"
E_PURPOSE = f"{NS_LVH}:PurposeOfHolding"                     # 保有目的（重要提案行為等の判定対象）
E_RATIO = f"{NS_LVH}:HoldingRatioOfShareCertificatesEtc"     # 株券等保有割合（0.0902=9.02%）
E_RATIO_PREV = f"{NS_LVH}:HoldingRatioOfShareCertificatesEtcPerLastReport"
E_OBLIG_DATE = f"{NS_LVH}:DateWhenFilingRequirementAroseCoverPage"  # 報告義務発生日（PIT非使用）
E_FILING_DATE = f"{NS_LVH}:FilingDateCoverPage"
E_N_JOINT = f"{NS_LVH}:TotalNumberOfFilersAndJointHoldersCoverPage"  # 提出者＋共同保有者総数
E_SHARES_HELD = f"{NS_LVH}:TotalNumberOfStocksEtcHeld"
E_SHARES_OUT = f"{NS_LVH}:TotalNumberOfOutstandingStocksEtc"
E_FILER_EDINET = f"{NS_DEI}:EDINETCodeDEI"
E_FILER_NAME_JP = f"{NS_DEI}:FilerNameInJapaneseDEI"
E_FILER_NAME_EN = f"{NS_DEI}:FilerNameInEnglishDEI"

_NEW, _CHANGE, _CORRECTION, _UNKNOWN = "new", "change", "correction", "unknown"


def report_class(doc_title: Optional[str]) -> str:
    """本体タイトル → 'new'（大量保有報告書）/'change'（変更報告書）/'correction'（訂正）。

    唯一信頼できる新規/変更/訂正の判別（formCode・docTypeCode は不可）。判定順は訂正→変更→
    新規（訂正報告書のタイトルは「大量保有報告書（…訂正報告書…）」と新規語を含むため先に除く）。
    """
    t = str(doc_title or "")
    if "訂正" in t:
        return _CORRECTION
    if "変更報告書" in t:
        return _CHANGE
    if "大量保有報告書" in t:
        return _NEW
    return _UNKNOWN


def is_special_report(form_code: Optional[str]) -> bool:
    """特例報告（法第27条の26・基準日方式）か。先頭 02/03＝特例（機関投資家・非介入）。

    formCode 先頭の一般(01)/特例(02/03)の別は信頼できる（新規/変更の末尾と違い安定）。
    """
    return str(form_code or "")[:2] in ("02", "03")


# --- 本体 CSV 抽出（純関数 ＋ 取得） ---------------------------------------
def lvh_records(doc_id: str, api_key: Optional[str] = None,
                refresh: bool = False) -> dict:
    """府令060 本体（type=5 CSV）を取得し大量保有 CSV（jplvh…）を {要素ID:値} で返す。"""
    path = ed.fetch_document(doc_id, doc_type=5, api_key=api_key, refresh=refresh)
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not names:
            return {}
        pick = next((n for n in names if "jplvh" in n.lower()), names[0])
        return parse_xbrl_csv(_decode(z.read(pick)))


def _ratio(val: Optional[str]) -> Optional[float]:
    if not val or val in ("－", "-"):
        return None
    return pd.to_numeric(_z2h(str(val)), errors="coerce")


def fields_lvh(rec: dict) -> dict:
    """大量保有本体（lvh_records の出力）から C2 特徴量を抽出。

    purpose=保有目的（重要提案行為等の判定対象）/ holding_ratio=株券等保有割合(小数) /
    filer_edinet・filer_name=名寄せキー（activists.resolve）/ oblig_date=報告義務発生日
    （PIT 非使用・提出ラグ特徴量用）。
    """
    purpose = rec.get(E_PURPOSE) or ""
    odates = extract_dates(rec.get(E_OBLIG_DATE)) or _iso_list(rec.get(E_OBLIG_DATE))
    title = (rec.get(E_TITLE) or "").strip()
    return {
        "doc_title": title,
        "report_class": report_class(title),
        "change_reason": (rec.get(E_REASON_CHANGE) or "").strip().strip("－-"),
        "purpose": purpose,
        "is_important_proposal": activists.is_important_proposal(purpose),
        "holding_ratio": _ratio(rec.get(E_RATIO)),
        "holding_ratio_prev": _ratio(rec.get(E_RATIO_PREV)),
        "n_joint_holders": pd.to_numeric(_z2h(rec.get(E_N_JOINT) or ""),
                                         errors="coerce"),
        "shares_held": pd.to_numeric(_z2h(rec.get(E_SHARES_HELD) or ""),
                                     errors="coerce"),
        "shares_outstanding": pd.to_numeric(_z2h(rec.get(E_SHARES_OUT) or ""),
                                            errors="coerce"),
        "filer_edinet": (rec.get(E_FILER_EDINET) or "").strip(),
        "filer_name_jp": (rec.get(E_FILER_NAME_JP) or "").strip(),
        "filer_name_en": (rec.get(E_FILER_NAME_EN) or "").strip(),
        "oblig_date": odates[0] if odates else None,
    }


def _iso_list(val: Optional[str]) -> list[str]:
    """'2026-06-05' のような既 ISO 文字列をそのまま 1 要素リストに（和暦数字以外の保険）。"""
    if not val:
        return []
    s = str(val).strip()
    return [s] if len(s) >= 8 and s[:4].isdigit() else []


def tag_filer(filer_name: Optional[str], filer_edinet: Optional[str]) -> dict:
    """提出者を activist registry で名寄せ：canonical_group / style（hard|soft）。名簿外は None。"""
    m = activists.resolve(filer_name=filer_name, edinet_code=filer_edinet)
    return {
        "canonical_group": m["canonical_group"] if m else None,
        "style": m["style"] if m else None,
        "in_registry": m is not None,
    }


def enrich_holding(doc_id: str, list_row: Optional[dict] = None,
                   api_key: Optional[str] = None) -> dict:
    """1 件の大量保有報告本体を取得し C2 特徴量＋名寄せタグを付けた 1 レコードを返す。

    list_row（一覧メタ）があれば submitDateTime（PIT アンカー）・issuerEdinetCode・
    secCode・formCode を引き継ぐ。本体取得失敗は body_ok=False で握り潰す（1 件で全体を
    落とさない）。
    """
    out: dict = {"doc_id": doc_id, "body_ok": False}
    if list_row:
        out.update({
            "submit_dt": list_row.get("submitDateTime"),
            "issuer_edinet": list_row.get("issuerEdinetCode"),
            "sec_code": list_row.get("secCode"),
            "form_code": list_row.get("formCode"),
            "special_report": is_special_report(list_row.get("formCode")),
        })
    try:
        rec = lvh_records(doc_id, api_key=api_key)
    except Exception:                                       # noqa: BLE001
        return out
    if not rec:
        return out
    f = fields_lvh(rec)
    out.update(f)
    out.update(tag_filer(f.get("filer_name_jp") or f.get("filer_name_en"),
                         f.get("filer_edinet")))
    out["body_ok"] = True
    return out
