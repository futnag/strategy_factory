"""有価証券報告書の叙述テキスト抽出と前年比変化シグナル（Lazy Prices 型）。

候補①テキストα Stage 1（docs/45 事前登録）。EDINET type=5（XBRL→CSV）から MD&A 系の
叙述セクションを抽出し、同一銘柄の**前年比コサイン類似度**を作る。類似度が低い＝記述を
大きく書き換えた＝Lazy Prices の負シグナル（記述変化銘柄は将来アンダーパフォーム）。

PIT：`submitDateTime` をアンカーに、両方とも提出済みの有報のみ比較（先読みなし）。テキスト
処理は**文字 n-gram**（日本語トークナイザ不要・sklearn のみ・対象2文書のみで完全PIT＝コーパス
IDFのリーク無し）。純関数（`yoy_text_similarity`/`extract_narrative`）はオフラインでテスト可。
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.sources import edinet as ed

# MD&A・前向き・経営判断セクション（定型の財務注記＝金融商品/退職給付/セグメント等は除外＝soft info に絞る）
MD_KEYWORDS = ("業績等の概要", "経営成績等の状況", "経営者による", "財政状態、経営成績",
               "事業等のリスク", "対処すべき課題", "経営方針", "経営環境")
# Stage1.5（脱飽和）：年次変動が大きく経営判断が出る節のみ（ガバナンス定型・大株主等を排除）
NARROW_KEYWORDS = ("事業等のリスク", "対処すべき課題")
ANNUAL_DOC_TYPES = ("120", "130")          # 有報・訂正有報


def extract_narrative(zip_path, keywords=MD_KEYWORDS) -> str:
    """type=5 ZIP → MD&A 系テキストブロックを連結した叙述本文（失敗時は空文字）。"""
    try:
        facts = ed.read_xbrl_csv(zip_path)
    except Exception:                       # noqa: BLE001（壊れZIP・欠損は空扱い）
        return ""
    if facts.empty or "item_name" not in facts.columns:
        return ""
    name = facts["item_name"].astype("string").fillna("")
    val = facts["value"].astype("string").fillna("")
    is_text = (facts["value_num"].isna().to_numpy()
               & (val.str.len().fillna(0) > 80).to_numpy())
    pat = "|".join(re.escape(k) for k in keywords)
    hit = name.str.contains(pat, regex=True, na=False).to_numpy()
    return "\n".join(val[is_text & hit].tolist())


def annual_report_index(base=None, doc_types=ANNUAL_DOC_TYPES) -> pd.DataFrame:
    """EDINET list（by-date）から有報の PIT 索引 [Code, submitDateTime, docID]。

    secCode(5桁)＝J-Quants Code。submitDateTime が PIT アンカー。docID で本体 ZIP を引く。
    """
    root = Path(base) if base is not None else ed._CACHE
    cols = ["docID", "secCode", "docTypeCode", "submitDateTime"]
    frames = []
    d = root / "list"
    if d.exists():
        for p in sorted(d.glob("*.parquet")):
            df = pd.read_parquet(p)
            if "_empty" in df.columns or "docTypeCode" not in df.columns:
                continue
            sub = df[df["docTypeCode"].isin(list(doc_types)) & df["secCode"].notna()]
            if len(sub):
                frames.append(sub[cols])
    if not frames:
        return pd.DataFrame(columns=["Code", "submitDateTime", "docID"])
    out = pd.concat(frames, ignore_index=True)
    out["Code"] = out["secCode"].astype(str)
    out["submitDateTime"] = pd.to_datetime(out["submitDateTime"], errors="coerce")
    out = out.dropna(subset=["submitDateTime"]).drop_duplicates("docID")
    return out.sort_values("submitDateTime")[["Code", "submitDateTime", "docID"]] \
        .reset_index(drop=True)


def _pair_cosine(a: str, b: str) -> float:
    """2 文書の文字 n-gram(2,3) コサイン類似度（対象2文書のみで fit＝完全PIT）。"""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    if not a or not b:
        return float("nan")
    try:
        X = CountVectorizer(analyzer="char", ngram_range=(2, 3),
                            min_df=1).fit_transform([a, b])
    except ValueError:                      # 語彙が空（記号のみ等）
        return float("nan")
    if X.shape[0] < 2:
        return float("nan")
    return float(cosine_similarity(X[0], X[1])[0, 0])


def yoy_text_similarity(texts: pd.DataFrame) -> pd.DataFrame:
    """[Code, submitDateTime, text] → [Code, submitDateTime, text_sim]（前年比コサイン）。

    各銘柄で提出日昇順に並べ、連続する有報ペアの類似度を後者の submitDateTime に紐付ける
    （両方とも提出済み＝先読みなし）。初回提出は前年が無いので NaN（自然に除外）。
    """
    cols = ["Code", "submitDateTime", "text_sim"]
    if texts.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for code, g in texts.sort_values("submitDateTime").groupby("Code"):
        g = g[g["text"].astype(str).str.len() > 0]
        prev = None
        for _, r in g.iterrows():
            cur = str(r["text"])
            if prev is not None:
                rows.append({"Code": str(code), "submitDateTime": r["submitDateTime"],
                             "text_sim": _pair_cosine(prev, cur)})
            prev = cur
    return pd.DataFrame(rows, columns=cols)


def _consecutive_pairs(texts: pd.DataFrame):
    """[Code, submitDateTime, text] → [(Code, later_dt, later_text, earlier_text, later_year)]。"""
    out = []
    t = texts.sort_values("submitDateTime")
    for code, g in t.groupby("Code"):
        g = g[g["text"].astype(str).str.len() > 0]
        prev = None
        for _, r in g.iterrows():
            cur = str(r["text"])
            if prev is not None:
                out.append((str(code), r["submitDateTime"], cur, prev,
                            int(pd.Timestamp(r["submitDateTime"]).year)))
            prev = cur
    return out


def yoy_text_similarity_idf(texts: pd.DataFrame, min_corpus: int = 200,
                            max_features: int = 30000, min_df: int = 5) -> pd.DataFrame:
    """trailing-IDF 重み付き前年比コサイン（Stage1.5 脱飽和）。

    各「後者提出年 Y」について **過去（提出年 < Y）の有報コーパスに TF-IDF を fit**（PIT・先読みなし）し、
    そのIDF空間で両年の文書をベクトル化→コサイン。定型語（コーパス頻出＝低IDF）が減衰し、記述変化が立つ。
    コーパスが min_corpus 未満の年（初期 warmup）は NaN。文字 n-gram(2,3)・日本語トークナイザ不要。
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    cols = ["Code", "submitDateTime", "text_sim"]
    if texts.empty:
        return pd.DataFrame(columns=cols)
    t = texts.copy()
    t["_year"] = pd.to_datetime(t["submitDateTime"]).dt.year
    pairs = _consecutive_pairs(texts)
    rows = []
    for Y in sorted({p[4] for p in pairs}):
        corpus = t.loc[t["_year"] < Y, "text"].astype(str).tolist()
        if len(corpus) < min_corpus:
            continue
        vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 3),
                              min_df=min_df, max_features=max_features)
        try:
            vec.fit(corpus)
        except ValueError:
            continue
        for code, dt, later, earlier, _ in (p for p in pairs if p[4] == Y):
            M = vec.transform([later, earlier])
            rows.append({"Code": code, "submitDateTime": dt,
                         "text_sim": float(cosine_similarity(M[0], M[1])[0, 0])})
    return pd.DataFrame(rows, columns=cols)
