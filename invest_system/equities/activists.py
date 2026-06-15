"""アクティビスト名簿（registry）のローダー＋C2 ユニバース判定（purpose フィルタ）。

v2 方針（docs/10 §6-7・HANDOFF）— **registry はユニバースを決めるものではない**：
- **C2 ユニバースの母集合は「保有目的に『重要提案行為等』を含む大量保有報告」**で機械
  定義する（`is_important_proposal`）。法定開示テキストによる客観・網羅・自己更新的な
  分類で、手編集名簿でファンドを選ぶ in-sample 選択を避ける。
- registry は **名寄せ（filerName/edinetCode → グループ）とスタイルタグ（hard/soft）の
  補助** に格下げ。複数 EDINET コードの dedup（村上系・ダルトン系）も担う。
- データの正は `invest_system/equities/activist_registry.csv`（コミット可＝公開
  リファレンス。index_events.py と同じ扱い。J-Quants 市場データと違い ToS 制約なし）。

CSV 列：group_slug / canonical_group（dedup 後のグループ）/ display_name /
edinet_codes（';' 区切り）/ style（engagement_hard|engagement_soft）/ aliases（';' 区切り
名寄せ用）/ note。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

REGISTRY_PATH = Path(__file__).parent / "activist_registry.csv"
STYLES = ("engagement_hard", "engagement_soft")
IMPORTANT_PROPOSAL = "重要提案行為等"          # C2 ユニバースの母集合キーワード（docs/10 §7）


@lru_cache(maxsize=1)
def registry() -> pd.DataFrame:
    """名簿 DataFrame（1 行 = 1 グループ）。全列 str・欠損は空文字。"""
    return pd.read_csv(REGISTRY_PATH, dtype=str).fillna("")


def _norm(s: str) -> str:
    """名寄せ用の正規化：空白・記号・法人格/共通語を除去して小文字化。"""
    s = str(s)
    for ch in (" ", "　", "・", "（", "）", "(", ")", "、", "，", ",", ".", "－", "-"):
        s = s.replace(ch, "")
    for w in ("株式会社", "合同会社", "有限会社", "投資事業有限責任組合", "LLC", "Ltd",
              "Limited", "Inc", "L.P.", "LP", "Pte", "Company", "Co", "Management",
              "マネジメント", "マネージメント", "キャピタル", "Capital", "Partners",
              "パートナーズ"):
        s = s.replace(w, "")
    return s.lower()


def _aliases(row: dict) -> list[str]:
    extra = row["aliases"].split(";") if row.get("aliases") else []
    return [a for a in ([row["display_name"]] + extra) if a]


def _codes(row: dict) -> list[str]:
    return [c for c in row["edinet_codes"].split(";") if c] if row.get("edinet_codes") else []


def resolve(filer_name: Optional[str] = None,
            edinet_code: Optional[str] = None) -> Optional[dict]:
    """保有者を名簿に名寄せ。edinetCode 優先（複数コード対応）、無ければ別名で名称マッチ。

    名称マッチは first-pass：正規化別名が 4 文字以上なら部分一致、短ければ完全一致のみ
    （誤マッチ回避）。複数一致は最長別名を採用。名簿外は None（＝grouping 対象外）。
    """
    df = registry()
    recs = df.to_dict("records")
    if edinet_code:
        for r in recs:
            if edinet_code in _codes(r):
                return r
    if filer_name:
        fn = _norm(filer_name)
        best: Optional[tuple[dict, int]] = None
        for r in recs:
            for alias in _aliases(r):
                na = _norm(alias)
                if not na:
                    continue
                hit = (na in fn) if len(na) >= 4 else (na == fn)
                if hit and (best is None or len(na) > best[1]):
                    best = (r, len(na))
        if best:
            return best[0]
    return None


def style_of(filer_name: Optional[str] = None,
             edinet_code: Optional[str] = None) -> Optional[str]:
    """スタイルタグ（engagement_hard/soft）。名簿外は None。"""
    m = resolve(filer_name, edinet_code)
    return m["style"] if m else None


def canonical_group_of(filer_name: Optional[str] = None,
                       edinet_code: Optional[str] = None) -> Optional[str]:
    """dedup 後のグループ名（村上系→murakami, ダルトン系→dalton 等）。名簿外は None。"""
    m = resolve(filer_name, edinet_code)
    return m["canonical_group"] if m else None


def is_important_proposal(purpose_text: Optional[str]) -> bool:
    """C2 ユニバース母集合の判定：保有目的に『重要提案行為等』を含むか（docs/10 §7）。

    registry 掲載の有無に依らず、保有目的テキスト（EDINET 大量保有 CSV の保有目的欄）で
    機械判定する＝手編集名簿の主観を排除。registry は判定後のグループ化・タグ付け用。
    """
    return bool(purpose_text) and IMPORTANT_PROPOSAL in str(purpose_text)
