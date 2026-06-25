"""JPX「ToSTNeT取引 超大口約定情報」スクレイパ（無料・監視用）。

方針書 = リポジトリ root の ``tostnet_monitoring_plan.md``。検証済みの実態：

- 対象ページ https://www.jpx.co.jp/markets/equities/tostnet/index.html は**静的HTML**
  （Playwright 不要）。表の「取引内容」セルは**日次 Excel(.xlsx) へのリンク**で、ファイル名は
  ``YYYYMMDD_ToSTNeT_Trading_Information.xlsx``（YYYYMMDD = 取引日）。
- Excel は非連番の CMS フォルダ ID 配下＝**過去URL推測不可＝バックフィル不可**。よって本モジュールは
  「現在ページに載っている直近~2週間分のうち、未取得の取引日だけを取得して前向きに蓄積」する。
- Excel の9列は J-Quants Pro ``/prices/tostnet_super_large_lot`` と**同一構造**。ローカルを Pro 準拠の
  正準列名で保存しておけば、将来 Pro 契約時にスクレイプ分と API 分を同一スキーマで合流できる。

設計は他ソース（bitbank.py 等）に倣い、生データ→正準スキーマ変換を純関数（``parse_index_links`` /
``parse_trading_excel``）として分離し、ネットワーク無しでテスト可能にする。Excel 解析に ``openpyxl`` が必要。
"""
from __future__ import annotations

import io
import re
import time
import urllib.request
from pathlib import Path

import pandas as pd

_JPX_HOST = "https://www.jpx.co.jp"
_INDEX_URL = _JPX_HOST + "/markets/equities/tostnet/index.html"
# 公開ページへ礼儀正しくアクセスするための既定 UA（一般的なブラウザ相当）。
_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept-Language": "ja,en;q=0.8",
}

# 正準スキーマ（J-Quants Pro /prices/tostnet_super_large_lot に一致）
CANON_COLUMNS = [
    "PublicationDate", "Date", "TradeTime", "Code", "CompanyName",
    "CompanyNameEnglish", "Price", "Volume", "TurnoverValue", "source",
]

# Excel ヘッダ（"公表日/Publication_Date" のように日英スラッシュ結合）→ 正準列名。
# 英語トークン優先で部分一致（"trading_date" と "trading_value/volume" を取り違えない粒度）。
_ALIASES = [
    ("PublicationDate", ("publication_date", "公表日")),
    ("Date", ("trading_date", "取引日")),
    ("TradeTime", ("trade_time", "約定時刻")),
    ("Code", ("code", "銘柄コード")),
    ("CompanyName", ("issue_name_japanese", "銘柄名_日本語", "銘柄名_日本")),
    ("CompanyNameEnglish", ("issue_name_english", "銘柄名_英語", "銘柄名_英")),
    ("Price", ("price", "価格")),
    ("Volume", ("trading_volume", "売買高")),
    ("TurnoverValue", ("trading_value", "売買代金")),
]

_XLSX_RE = re.compile(
    r'href="([^"]*?(\d{8})_ToSTNeT_Trading_Information\.xlsx)"', re.IGNORECASE)


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=CANON_COLUMNS)


# --- パース（純関数・ネット不要） ----------------------------------------
def parse_index_links(html: str) -> list[tuple[str, str]]:
    """index ページHTMLから日次Excelリンクを抽出。

    返り値：``[(trade_date 'YYYYMMDD', href_path), ...]`` を取引日の降順で。重複日は最初の
    リンクを採用。``href_path`` は相対（'/markets/...'）または絶対のいずれもあり得る。
    """
    seen: dict[str, str] = {}
    for m in _XLSX_RE.finditer(html):
        path, date = m.group(1), m.group(2)
        seen.setdefault(date, path)
    return [(d, seen[d]) for d in sorted(seen, reverse=True)]


def _map_columns(cols) -> dict:
    """実ヘッダ列 → 正準列名の対応（英/日トークンの部分一致、二重割当を防止）。"""
    pool = list(cols)
    low = {c: str(c).lower() for c in cols}
    mapping: dict = {}
    for canon, aliases in _ALIASES:
        found = None
        for c in pool:
            if any(a.lower() in low[c] for a in aliases):
                found = c
                break
        mapping[canon] = found
        if found is not None:
            pool.remove(found)
    return mapping


def _to_float(s: pd.Series) -> pd.Series:
    """カンマ区切り文字列（'5,970,051,048'）→ float。空/欠損は NaN。"""
    cleaned = (s.astype(str).str.replace(",", "", regex=False).str.strip()
               .replace({"": None, "nan": None, "None": None}))
    # 日ごとの int64/float64 揺れを避けスキーマを安定させる（Pro API も float）
    return pd.to_numeric(cleaned, errors="coerce").astype("float64")


def _to_date(s: pd.Series) -> pd.Series:
    """'YYYYMMDD' / 'YYYY/MM/DD' / Excel日付 のいずれも datetime64 に。"""
    txt = (s.astype(str).str.strip()
           .replace({"": None, "nan": None, "None": None, "NaT": None}))
    out = pd.Series(pd.NaT, index=txt.index, dtype="datetime64[ns]")
    is8 = (txt.str.fullmatch(r"\d{8}") == True)  # noqa: E712（NA→False に潰す）
    if is8.any():
        out[is8] = pd.to_datetime(txt[is8], format="%Y%m%d", errors="coerce")
    rest = (~is8) & txt.notna()
    if rest.any():
        out[rest] = pd.to_datetime(txt[rest], errors="coerce")
    return out


def parse_trading_excel(content: bytes) -> pd.DataFrame:
    """日次 ToSTNeT Excel（bytes）→ 正準スキーマの取引明細 DataFrame（純関数）。

    ヘッダ行を自動検出（先頭にタイトル行があっても可）。数値列はカンマ除去、Code は文字列
    （'285A' 等の英数字コードがあるため）。該当取引が無い日（ヘッダのみ）は空フレームを返す。
    """
    try:
        raw = pd.read_excel(io.BytesIO(content), header=None, dtype=str,
                            engine="openpyxl")
    except Exception:
        return _empty_frame()
    if raw.empty:
        return _empty_frame()
    # ヘッダ行を探す（'公表日'/'Publication_Date' を含む最初の行）
    hdr_idx = None
    for i in range(min(len(raw), 10)):
        joined = " ".join(str(x) for x in raw.iloc[i].tolist()).lower()
        if "publication_date" in joined or "公表日" in joined:
            hdr_idx = i
            break
    if hdr_idx is None:
        return _empty_frame()
    header = [str(x) for x in raw.iloc[hdr_idx].tolist()]
    body = raw.iloc[hdr_idx + 1:].copy()
    body.columns = header
    body = body.dropna(how="all")
    if body.empty:
        return _empty_frame()

    colmap = _map_columns(header)
    out = pd.DataFrame(index=body.index)
    for canon in CANON_COLUMNS:
        if canon == "source":
            continue
        src = colmap.get(canon)
        out[canon] = body[src] if src is not None else pd.NA

    for c in ("Price", "Volume", "TurnoverValue"):
        out[c] = _to_float(out[c])
    out["PublicationDate"] = _to_date(out["PublicationDate"])
    out["Date"] = _to_date(out["Date"])
    out["TradeTime"] = out["TradeTime"].astype(str).str.strip()
    code = (out["Code"].astype(str).str.strip()
            .str.replace(r"\.0$", "", regex=True))  # 数値化された '6861.0' を補正
    out["Code"] = code
    for c in ("CompanyName", "CompanyNameEnglish"):
        out[c] = out[c].astype(str).str.strip()
    out["source"] = "jpx_scrape"

    # Code が無い行（脚注・空行）を除外
    valid = out["Code"].notna() & (out["Code"].str.len() > 0) \
        & ~out["Code"].str.lower().isin(["nan", "none", "<na>"])
    out = out[valid]
    if out.empty:
        return _empty_frame()
    return out[CANON_COLUMNS].reset_index(drop=True)


# --- ネットワーク（薄いラッパ・DI可） -------------------------------------
def fetch_index(url: str = _INDEX_URL, *, timeout: int = 30,
                headers: dict | None = None) -> str:
    """index ページHTMLを取得（公開ページ・静的HTML）。"""
    req = urllib.request.Request(url, headers=headers or _HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch_excel(path_or_url: str, *, host: str = _JPX_HOST, timeout: int = 30,
                headers: dict | None = None) -> bytes:
    """日次 Excel を取得（相対パスは host で絶対化）。"""
    url = path_or_url if path_or_url.startswith("http") else host + path_or_url
    req = urllib.request.Request(url, headers=headers or _HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


# --- 前向き蓄積（冪等・再開可能） ----------------------------------------
def update_tostnet(cache_dir: str = "data/jpx_tostnet", *, index_url: str = _INDEX_URL,
                   fetch_index=fetch_index, fetch_excel=fetch_excel,
                   pause: float = 1.0, verbose: bool = True) -> dict:
    """現在ページの未取得取引日だけ取得し ``{cache_dir}/{YYYYMMDD}.parquet`` へ保存。

    既存ファイルはスキップ（=冪等・再開可能）。該当取引が無い日は ``_empty`` マーカーを書いて
    再取得を防ぐ。``fetch_index``/``fetch_excel`` は DI 可能（テスト/差し替え用）。
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    html = fetch_index(index_url)
    links = parse_index_links(html)
    rep = {"available": len(links), "fetched": 0, "skipped": 0,
           "empty": 0, "rows": 0, "errors": 0}
    for date, path in links:
        fp = cache / f"{date}.parquet"
        if fp.exists():
            rep["skipped"] += 1
            continue
        try:
            df = parse_trading_excel(fetch_excel(path))
        except Exception as e:  # noqa: BLE001
            rep["errors"] += 1
            if verbose:
                print(f"  [warn] {date}: {str(e)[:70]}")
            continue
        if df.empty:
            pd.DataFrame({"_empty": [True],
                          "Date": [pd.to_datetime(date, format="%Y%m%d")]}).to_parquet(fp)
            rep["empty"] += 1
        else:
            df.to_parquet(fp)
            rep["rows"] += len(df)
        rep["fetched"] += 1
        if pause:
            time.sleep(pause)
    if verbose:
        print(f"  tostnet: 利用可能{rep['available']} / 取得{rep['fetched']}"
              f"（空{rep['empty']}・行{rep['rows']}）/ スキップ{rep['skipped']}"
              f" / エラー{rep['errors']}")
    return rep


def load_tostnet(cache_dir: str = "data/jpx_tostnet") -> pd.DataFrame:
    """蓄積済みの取引明細を1つの DataFrame に（空マーカーはスキップ）。"""
    cache = Path(cache_dir)
    if not cache.exists():
        return _empty_frame()
    frames = []
    for p in sorted(cache.glob("*.parquet")):
        df = pd.read_parquet(p)
        if df.empty or "_empty" in df.columns:
            continue
        frames.append(df)
    if not frames:
        return _empty_frame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["Date", "TradeTime", "Code"]).reset_index(drop=True)


def daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    """日次の監視サマリ（取引件数・銘柄数・合計代金[億円]・最大銘柄）。"""
    if df.empty:
        return pd.DataFrame(
            columns=["trade_count", "unique_stocks", "total_value_oku",
                     "top_code", "top_value_oku"]).rename_axis("Date")
    rows = []
    for date, sub in df.groupby(df["Date"].dt.normalize()):
        top = sub.loc[sub["TurnoverValue"].idxmax()]
        rows.append({
            "Date": date,
            "trade_count": int(len(sub)),
            "unique_stocks": int(sub["Code"].nunique()),
            "total_value_oku": round(float(sub["TurnoverValue"].sum()) / 1e8, 2),
            "top_code": top["Code"],
            "top_value_oku": round(float(top["TurnoverValue"]) / 1e8, 2),
        })
    return pd.DataFrame(rows).set_index("Date").sort_index()


# --- セクター付与・集計（既存 S33 マスタを再利用） ------------------------
_UNMAPPED_SECTOR = "その他/ETF等"


def sector_map(listed: pd.DataFrame | None = None) -> pd.Series:
    """4桁 Code → S33 業種名（J-Quants マスタ）。

    J-Quants の Code は5桁（'68610'）、ToSTNeT は4桁（'6861'）なので**先頭4桁**で索引して照合する。
    ``listed`` を渡せば DI 可能（未指定なら ``jq.fetch_listed_info()` のキャッシュを読む）。
    """
    if listed is None:
        from invest_system.data.sources import jquants as jq  # 遅延 import
        listed = jq.fetch_listed_info()
    if listed is None or listed.empty or "S33Nm" not in listed.columns:
        return pd.Series(dtype=object)
    key = listed["Code"].astype(str).str[:4]
    out = (listed.assign(_k=key).dropna(subset=["S33Nm"])
           .drop_duplicates("_k", keep="first").set_index("_k")["S33Nm"])
    return out


def add_sector(df: pd.DataFrame, smap: pd.Series | None = None) -> pd.DataFrame:
    """取引明細に ``sector``（S33業種名）列を付与。未収載（ETF等）は 'その他/ETF等'。"""
    out = df.copy()
    if out.empty:
        out["sector"] = pd.Series(dtype=object)
        return out
    if smap is None:
        smap = sector_map()
    key = out["Code"].astype(str).str[:4]
    out["sector"] = key.map(smap).fillna(_UNMAPPED_SECTOR)
    return out


def daily_sector_summary(df: pd.DataFrame, smap: pd.Series | None = None) -> pd.DataFrame:
    """日次×業種の集計（合計代金[億円]・取引件数・銘柄数）を long で返す。"""
    cols = ["Date", "sector", "total_value_oku", "trade_count", "stock_count"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    d = add_sector(df, smap)
    g = d.groupby([d["Date"].dt.normalize().rename("Date"), "sector"])
    out = g.agg(total_value_yen=("TurnoverValue", "sum"),
                trade_count=("Code", "size"),
                stock_count=("Code", "nunique")).reset_index()
    out["total_value_oku"] = (out.pop("total_value_yen") / 1e8).round(2)
    return out[cols].sort_values(["Date", "total_value_oku"],
                                 ascending=[True, False]).reset_index(drop=True)
