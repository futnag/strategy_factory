"""TDnet（適時開示）日次一覧スクレイパ — 前向き蓄積用。

公式 API は無い。公開閲覧は直近約1ヶ月のみ（バックフィル不可＝前向き蓄積）。
ページは静的 HTML（Playwright 不要）。日次一覧 URL::

    https://www.release.tdnet.info/inbs/I_list_{page:03d}_{YYYYMMDD}.html

1ページ最大100件。PDF/XBRL リンクは相対パス（``1401....pdf``）。

用途（docs/47）:
- C3: 自社株買い「発表」イベント（``buyback_announce`` タグ）
- C1 補助: TOB 関連適時開示（``tob_related`` タグ）— 一次は EDINET
- 旗艦防御: 業績修正・大量保有関連の早期検知

設計は ``jpx_tostnet.py`` に倣い、パースを純関数化してオフライン検証可能にする。
取得データは ``data/tdnet/``（gitignore）にのみ保持。
"""
from __future__ import annotations

import re
import time
import urllib.request
from pathlib import Path

import pandas as pd

_HOST = "https://www.release.tdnet.info/inbs/"
_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept-Language": "ja,en;q=0.8",
}

CANON_COLUMNS = [
    "disclosure_date", "disclosure_time", "code", "company_name", "title",
    "pdf_url", "xbrl_url", "exchange", "doc_id", "event_tags", "source",
]

# 表題キーワード → イベントタグ（複数ヒット可・事前固定）
_TAG_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("buyback_announce", (
        "自己株式取得に係る事項の決定",
        "自己株式の取得枠設定",
        "自己投資口取得に係る事項の決定",
    )),
    ("buyback_result", (
        "自己株式立会外買付取引",
        "ToSTNeT-3",
        "ToSTNeT-３",
        "自己株式の取得状況",
        "自己株式の取得結果",
        "自己株式の消却",
    )),
    ("tob_related", (
        "公開買付",
        "TOB",
        "買集め行為",
        "意見表明",
        "対抗措置",
        "MBO",
    )),
    ("guidance_revision", (
        "業績予想の修正",
        "通期業績予想",
        "業績予想の上方修正",
        "業績予想の下方修正",
    )),
    ("earnings_release", (
        "決算短信",
        "四半期決算",
    )),
    ("large_holder", (
        "主要株主の異動",
        "大量保有",
        "支配株主等に関する事項",
    )),
    ("dividend", (
        "剰余金の配当",
        "配当予想の修正",
        "配当方針",
    )),
]

_ROW_RE = re.compile(
    r"<tr>\s*"
    r'<td[^>]*class="[^"]*kjTime[^"]*"[^>]*>([^<]*)</td>\s*'
    r'<td[^>]*class="[^"]*kjCode[^"]*"[^>]*>([^<]*)</td>\s*'
    r'<td[^>]*class="[^"]*kjName[^"]*"[^>]*>([^<]*)</td>\s*'
    r'<td[^>]*class="[^"]*kjTitle[^"]*"[^>]*>.*?href="([^"]+\.pdf)"[^>]*>([^<]*)</a>.*?</td>\s*'
    r'<td[^>]*class="[^"]*kjXbrl[^"]*"[^>]*>(.*?)</td>\s*'
    r'<td[^>]*class="[^"]*kjPlace[^"]*"[^>]*>([^<]*)</td>',
    re.IGNORECASE | re.DOTALL,
)
_PAGE_RE = re.compile(r"I_list_(\d{3})_(\d{8})\.html", re.IGNORECASE)
_XBRL_RE = re.compile(r'href="([^"]+\.zip)"', re.IGNORECASE)
_DOC_ID_RE = re.compile(r"([^/]+\.pdf)$", re.IGNORECASE)


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=CANON_COLUMNS)


def classify_title(title: str) -> str:
    """表題からイベントタグをカンマ区切りで返す（空なら空文字列）。"""
    if not title:
        return ""
    hits = [tag for tag, kws in _TAG_RULES if any(k in title for k in kws)]
    return ",".join(hits)


def normalize_code(raw: str) -> str:
    """TDnet コード → J-Quants 風5桁文字列（英字混じりはそのまま大文字化）。"""
    s = str(raw).strip().upper()
    if not s:
        return s
    if len(s) == 4 and s.isdigit():
        return s + "0"
    return s


def parse_list_page(html: str, disclosure_date: str) -> pd.DataFrame:
    """日次一覧 HTML 1ページ → 開示行 DataFrame（純関数）。"""
    rows = []
    for m in _ROW_RE.finditer(html):
        tm, code, name, pdf, title, xbrl_cell, exchange = m.groups()
        xbrl_m = _XBRL_RE.search(xbrl_cell or "")
        pdf_name = pdf.strip()
        doc_m = _DOC_ID_RE.search(pdf_name)
        rows.append({
            "disclosure_date": pd.to_datetime(disclosure_date, format="%Y%m%d"),
            "disclosure_time": str(tm).strip(),
            "code": normalize_code(code),
            "company_name": str(name).strip(),
            "title": str(title).strip(),
            "pdf_url": _HOST + pdf_name,
            "xbrl_url": (_HOST + xbrl_m.group(1)) if xbrl_m else "",
            "exchange": str(exchange).strip(),
            "doc_id": doc_m.group(1).replace(".pdf", "").replace(".PDF", "") if doc_m else "",
            "event_tags": classify_title(str(title).strip()),
            "source": "tdnet_scrape",
        })
    if not rows:
        return _empty_frame()
    return pd.DataFrame(rows)[CANON_COLUMNS]


def parse_page_links(html: str) -> list[str]:
    """HTML 内の同一日付のページファイル名一覧（001,002,...）。"""
    pages = sorted({m.group(0) for m in _PAGE_RE.finditer(html)},
                   key=lambda s: int(_PAGE_RE.search(s).group(1)))  # type: ignore[union-attr]
    return pages


def fetch_page(path_or_url: str, *, host: str = _HOST, timeout: int = 30,
               headers: dict | None = None) -> str:
    """一覧ページ HTML を取得。"""
    url = path_or_url if path_or_url.startswith("http") else host + path_or_url
    req = urllib.request.Request(url, headers=headers or _HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch_day(disclosure_date: str, *, fetch_page_fn=fetch_page,
              pause: float = 0.5) -> pd.DataFrame:
    """指定日の全ページを取得して1本の DataFrame に（ネットワークあり）。"""
    first = f"I_list_001_{disclosure_date}.html"
    html = fetch_page_fn(first)
    pages = parse_page_links(html) or [first]
    frames = []
    for i, p in enumerate(pages):
        page_html = html if p == first else fetch_page_fn(p)
        part = parse_list_page(page_html, disclosure_date)
        if not part.empty:
            frames.append(part)
        if pause and i < len(pages) - 1:
            time.sleep(pause)
    if not frames:
        return _empty_frame()
    return pd.concat(frames, ignore_index=True)


def update_tdnet(cache_dir: str = "data/tdnet", dates: list[str] | None = None,
                 *, fetch_day_fn=fetch_day, pause: float = 0.5,
                 verbose: bool = True) -> dict:
    """未取得の開示日だけ ``{cache_dir}/{YYYYMMDD}.parquet`` へ保存（冪等）。"""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    if dates is None:
        dates = [pd.Timestamp.today().strftime("%Y%m%d")]
    rep = {"requested": len(dates), "fetched": 0, "skipped": 0,
           "empty": 0, "rows": 0, "errors": 0}
    for d in dates:
        fp = cache / f"{d}.parquet"
        if fp.exists():
            rep["skipped"] += 1
            continue
        try:
            df = fetch_day_fn(d)
        except Exception as e:  # noqa: BLE001
            rep["errors"] += 1
            if verbose:
                print(f"  [warn] {d}: {str(e)[:80]}")
            continue
        if df.empty:
            pd.DataFrame({"_empty": [True],
                          "disclosure_date": [pd.to_datetime(d, format="%Y%m%d")]}
                         ).to_parquet(fp)
            rep["empty"] += 1
        else:
            df.to_parquet(fp)
            rep["rows"] += len(df)
        rep["fetched"] += 1
        if pause:
            time.sleep(pause)
    if verbose:
        print(f"  tdnet: 要求{rep['requested']} / 取得{rep['fetched']}"
              f"（空{rep['empty']}・行{rep['rows']}）/ スキップ{rep['skipped']}"
              f" / エラー{rep['errors']}")
    return rep


def load_tdnet(cache_dir: str = "data/tdnet",
               start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """蓄積済み開示一覧を1本に（``_empty`` マーカーは除外）。"""
    cache = Path(cache_dir)
    if not cache.exists():
        return _empty_frame()
    frames = []
    for p in sorted(cache.glob("*.parquet")):
        if start and p.stem < start:
            continue
        if end and p.stem > end:
            continue
        df = pd.read_parquet(p)
        if df.empty or "_empty" in df.columns:
            continue
        frames.append(df)
    if not frames:
        return _empty_frame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["disclosure_date", "disclosure_time", "code"]
                           ).reset_index(drop=True)


def filter_tagged(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    """``event_tags`` に指定タグを含む行だけ返す。"""
    if df.empty or "event_tags" not in df.columns:
        return df.iloc[0:0]
    mask = df["event_tags"].astype(str).str.contains(tag, regex=False, na=False)
    return df[mask].copy()