"""EDINET（金融庁 開示書類）データ取得：v2 API ＋ ローカル Parquet/バイナリキャッシュ。

EDINET API v2 は 2 本構成（docs/08 §1）:
  書類一覧  GET /documents.json?date=YYYY-MM-DD&type={1|2}&Subscription-Key=KEY
            type=1: メタデータのみ / type=2: 提出書類一覧＋メタデータ。
            レスポンスは {"metadata": {...}, "results": [{...}, ...]}（UTF-8 JSON・日本時間）。
  書類取得  GET /documents/{docID}?type={1..5}&Subscription-Key=KEY
            type 1=本文ZIP(XBRL含) / 2=PDF / 3=代替・添付ZIP / 4=英文ZIP / 5=CSV(ZIP)。
            成功/失敗は HTTP ステータスでなく **Content-Type で判定**（application/json
            =エラー。エラーでも HTTP 200 のことがある）。

認証は .env / 環境変数 EDINET_API_KEY（クエリ Subscription-Key で送る＝J-Quants と違い
ヘッダーではない）。キーはコード・git・ログに書かない。データは data/edinet/（gitignore
済）にキャッシュ：一覧は by-date Parquet（list/{YYYYMMDD}.parquet）、本体は
docs/{docID}_{type}.{zip|pdf}。

PIT / ミラー規律（docs/08 §3,§4）:
- submitDateTime が全候補（C1/C2/C4）の PIT アンカー。
- **追記専用スナップショット**：取下げ・職員修正・閲覧期間満了で過去レコードが API 上
  null 化されるため、初回観測時の一覧を Parquet で保全する（refresh=False が既定＝既存
  キャッシュを上書きしない）。日次更新で公式から消える前の状態を固定できるのはローカル
  ミラーだけ。
- 縦覧期間：大量保有・公開買付は法定縦覧のみ（~5 年）。古い書類は legalStatus=0 で一覧
  レコードが docID 以外 null 化＝バックテスト深度はミラー開始日に律速（有報は ~10 年）。

V2 一覧 results[] のフィールド（docs/08 §2）:
  docID 書類管理番号 / submitDateTime 提出日時(PIT) / secCode 証券コード(5桁,
  J-Quants Code と直結) / edinetCode・filerName 提出者 / ordinanceCode 府令 /
  formCode 様式 / docTypeCode 書類種別 / issuerEdinetCode 大量保有の発行会社(=対象) /
  subjectEdinetCode 公開買付の対象会社 / parentDocID 親書類(案件チェーン) /
  withdrawalStatus 取下 / legalStatus 縦覧区分 / xbrlFlag・csvFlag・pdfFlag 形式有無。
"""
from __future__ import annotations

import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd

from ...config import get_env

_BASE = "https://api.edinet-fsa.go.jp/api/v2"
_CACHE = Path("data/edinet")

# レート制限：数値仕様は非公開（429 のみ定義）。礼儀的スロットル＋指数バックオフで自衛。
# 一覧は 1 日 1 リクエストで軽い。重いのは書類本体（C4 の有報バックフィル）のみ。
# 環境変数で調整可: EDINET_MIN_INTERVAL（秒, 既定 1.0）, EDINET_MAX_RETRIES。
_MIN_INTERVAL = float(get_env("EDINET_MIN_INTERVAL", "1.0") or "1.0")
_MAX_RETRIES = int(get_env("EDINET_MAX_RETRIES", "5") or "5")
_RETRY_CODES = {429, 500, 502, 503, 504}
_last_call = [0.0]

# --- フィルタ定数（docs/08 §2.1・仕様書本文で確定） -------------------------
# 府令コード（ordinanceCode）
ORD_TOB_THIRD_PARTY = "040"   # 発行者以外の者による公開買付け（第三者TOB）= C1 対象
ORD_TOB_SELF = "050"          # 発行者による公開買付け（自社株TOB）= C1 から除外
ORD_LARGE_HOLDING = "060"     # 株券等の大量保有の状況の開示 = C2 対象

# 書類種別コード（docTypeCode）
DOC_TOB_NOTICE = "240"        # 公開買付届出書（C1 イベント起点：価格・期間・下限/上限）
DOC_TOB_NOTICE_AMEND = "250"  # 訂正公開買付届出書（バンプ＝価格変更・期間延長の検出）
DOC_TOB_WITHDRAW = "260"      # 公開買付撤回届出書（不成立系の出口）
DOC_TOB_REPORT = "270"        # 公開買付報告書（成立/応募数の一次ソース＝出口）
DOC_TOB_OPINION = "290"       # 意見表明報告書（賛同/反対＝友好的/敵対的の特徴量）
DOC_TOB_OPINION_AMEND = "300"  # 訂正意見表明報告書
DOC_TOB_QA = "310"            # 対質問回答報告書
DOC_LARGE_HOLDING = "350"     # 大量保有報告書（C2 イベント起点）
DOC_LARGE_HOLDING_AMEND = "360"  # 訂正大量保有報告書
DOC_ANNUAL = "120"            # 有価証券報告書（C4 の BS 明細ソース）
DOC_ANNUAL_AMEND = "130"      # 訂正有価証券報告書
DOC_QUARTERLY = "140"         # 四半期報告書
DOC_SEMIANNUAL = "160"        # 半期報告書
# 三表明細（ファンダ特徴量）ソース。既定は有報＋訂正有報。
ANNUAL_DOC_TYPES = (DOC_ANNUAL, DOC_ANNUAL_AMEND)

# C1 に必要な書類セット（docs/08 §8.4）。出口(270/260)を欠くと損益が確定できない。
TOB_DOC_TYPES = (DOC_TOB_NOTICE, DOC_TOB_NOTICE_AMEND, DOC_TOB_WITHDRAW,
                 DOC_TOB_REPORT, DOC_TOB_OPINION, DOC_TOB_OPINION_AMEND, DOC_TOB_QA)

# results[] のスキーマ（by-date Parquet を日跨ぎで揃えるため固定）。コード・フラグは
# 先頭ゼロ・区分値を保つため文字列のまま（数値化しない＝"040" を 40 にしない）。
_INT_COLS = ["seqNumber"]
_STR_COLS = [
    "docID", "edinetCode", "secCode", "JCN", "filerName", "fundCode",
    "ordinanceCode", "formCode", "docTypeCode", "docDescription",
    "issuerEdinetCode", "subjectEdinetCode", "subsidiaryEdinetCode",
    "currentReportReason", "parentDocID",
    "withdrawalStatus", "docInfoEditStatus", "disclosureStatus",
    "xbrlFlag", "pdfFlag", "attachDocFlag", "englishDocFlag", "csvFlag",
    "legalStatus",
]
_DT_COLS = ["submitDateTime", "opeDateTime"]            # 提出日時（PIT アンカー）
_DATE_COLS = ["periodStart", "periodEnd"]               # 対象期間（日付のみ）
_ALL_COLS = _INT_COLS + _STR_COLS + _DT_COLS + _DATE_COLS


def _throttle() -> None:
    """直前の呼び出しから _MIN_INTERVAL 秒空ける（礼儀的レート遵守）。"""
    dt = time.monotonic() - _last_call[0]
    if dt < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - dt)
    _last_call[0] = time.monotonic()


def _api_key(api_key: Optional[str] = None) -> str:
    key = api_key or get_env("EDINET_API_KEY")
    if not key:
        raise RuntimeError(
            "EDINET APIキーがありません。.env に EDINET_API_KEY を設定してください"
            "（クエリ Subscription-Key で送る方式）。")
    return key


def _ymd(x) -> str:
    """'2026-06-01' / '20260601' / Timestamp -> '20260601'。"""
    return str(x).replace("-", "").replace("/", "")[:8]


def _iso(x) -> str:
    """'20260601' / '2026-06-01' -> '2026-06-01'（EDINET API の date 形式）。"""
    d = _ymd(x)
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}"


def _request_json(url: str, timeout: int = 60) -> dict:
    """GET＋JSON（一覧 API）。429/5xx は指数バックオフで自動リトライ（ペーシング付き）。

    HTTP 200 でも metadata.status がエラーを示すことがあるため両方を確認する。
    """
    last_err = None
    for attempt in range(_MAX_RETRIES + 1):
        _throttle()
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                data = json.load(resp)
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in _RETRY_CODES and attempt < _MAX_RETRIES:
                time.sleep(_backoff(e, attempt))
                continue
            detail = e.read().decode("utf-8", "ignore")[:300]
            raise RuntimeError(f"EDINET API HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            last_err = e
            if attempt < _MAX_RETRIES:
                time.sleep(min(30.0, 3.0 * 2 ** attempt))
                continue
            raise RuntimeError(f"EDINET API network error: {e}") from e
        meta = data.get("metadata") or {}
        status = str(meta.get("status") or "200")
        if status not in ("200", "0"):
            if status == "429" and attempt < _MAX_RETRIES:
                time.sleep(min(120.0, 30.0 * 2 ** attempt))
                continue
            raise RuntimeError(f"EDINET API status {status}: {meta.get('message')}")
        return data
    raise RuntimeError(f"EDINET API failed after retries: {last_err}")


def _backoff(e: urllib.error.HTTPError, attempt: int) -> float:
    """429/5xx の待機秒。Retry-After 優先、無ければ 30 秒〜の指数（上限 120 秒）。"""
    ra = e.headers.get("Retry-After") if e.headers else None
    if ra and str(ra).strip().isdigit():
        return float(ra)
    return min(120.0, 30.0 * 2 ** attempt)


def _request_binary(url: str, timeout: int = 120) -> tuple[bytes, str]:
    """書類本体 (ZIP/PDF) を取得。返り値 (bytes, ext)。エラーは Content-Type で検出。"""
    last_err = None
    for attempt in range(_MAX_RETRIES + 1):
        _throttle()
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                ctype = (resp.headers.get("Content-Type") or "").lower()
                body = resp.read()
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in _RETRY_CODES and attempt < _MAX_RETRIES:
                time.sleep(_backoff(e, attempt))
                continue
            raise RuntimeError(f"EDINET doc HTTP {e.code}") from e
        except urllib.error.URLError as e:
            last_err = e
            if attempt < _MAX_RETRIES:
                time.sleep(min(30.0, 3.0 * 2 ** attempt))
                continue
            raise RuntimeError(f"EDINET doc network error: {e}") from e
        if "application/json" in ctype:          # エラー応答（本体が JSON）
            try:
                err = json.loads(body.decode("utf-8", "ignore"))
                msg = (err.get("metadata") or {}).get("message") or err
            except Exception:                    # noqa: BLE001
                msg = body[:200]
            raise RuntimeError(f"EDINET doc error: {msg}")
        return body, ("pdf" if "pdf" in ctype else "zip")
    raise RuntimeError(f"EDINET doc failed after retries: {last_err}")


# --- パース（純関数・ネットワーク不要） -----------------------------------
def parse_documents(results: list) -> pd.DataFrame:
    """documents.json の results[] → DataFrame（スキーマを固定して日跨ぎ突合を頑健化）。

    コード・フラグ（ordinanceCode/docTypeCode/secCode/各Flag 等）は先頭ゼロ・区分値を
    保つため文字列のまま。submitDateTime/opeDateTime は datetime 化（PIT アンカー）。
    未知の追加フィールドは末尾に温存する。
    """
    df = pd.DataFrame(results)
    for c in _ALL_COLS:                          # 欠ける列も NA で用意（schema 安定）
        if c not in df.columns:
            df[c] = pd.NA
    for c in _STR_COLS:
        df[c] = df[c].astype("string")
    for c in _DT_COLS:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in _DATE_COLS:
        df[c] = pd.to_datetime(df[c], errors="coerce", format="mixed")
    df["seqNumber"] = pd.to_numeric(df["seqNumber"], errors="coerce").astype("Int64")
    extra = [c for c in df.columns if c not in _ALL_COLS]
    return df[_ALL_COLS + extra]


# --- type=5（XBRL→CSV）パース（純関数・ネットワーク不要） -----------------
# type=5 CSV は UTF-16・タブ区切り・固定 9 列（仕様書 / 実データで確認）:
#   要素ID / 項目名 / コンテキストID / 相対年度 / 連結・個別 / 期間・時点 / ユニットID / 単位 / 値
_XBRL_CSV_COLS = ["element_id", "item_name", "context_id", "rel_year",
                  "consolidated", "period_type", "unit_id", "unit", "value"]
# 当期・連結のコンテキスト（メンバー無し）。親会社単体は _NonConsolidatedMember、
# セグメント・内訳は各 *Member が付くため、素の CurrentYear* に限定して本体値を採る。
CURRENT_CONSOLIDATED_CTX = ("CurrentYearDuration", "CurrentYearInstant")


def read_xbrl_csv(path) -> pd.DataFrame:
    """type=5（XBRL→CSV）の ZIP もしくは CSV を財務ファクトの長形式 DataFrame に読む。

    ZIP には本報告（jpcrp*.csv）と監査報告（jpaud*.csv）が同梱される。本報告のみ採用
    （jpaud を除く最大 CSV）。value は文字列のまま（"-"・テキストブロック混在）保持し、
    数値版を value_num に、名前空間 prefix（jppfs_cor/jpigp_cor/jpcrp_cor 等）を
    namespace 列に分離する。
    """
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist()
                     if n.lower().endswith(".csv") and "jpaud" not in n.lower()]
            if not names:
                raise RuntimeError(f"type=5 ZIP に本報告CSVが無い: {path.name}")
            main = max(names, key=lambda n: z.getinfo(n).file_size)
            raw = z.read(main)
    else:
        raw = path.read_bytes()
    df = pd.read_csv(io.BytesIO(raw), sep="\t", encoding="utf-16", dtype=str)
    if df.shape[1] >= 9:                         # 末尾に予備列が付く版に備え先頭 9 列
        df = df.iloc[:, :9]
        df.columns = _XBRL_CSV_COLS
    df["namespace"] = df["element_id"].str.split(":").str[0]
    df["value_num"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def consolidated_current(facts: pd.DataFrame) -> pd.DataFrame:
    """当期・連結のコンテキストのみに絞る（CurrentYearDuration/Instant、メンバー無し）。"""
    return facts[facts["context_id"].isin(CURRENT_CONSOLIDATED_CTX)]


# --- 取得（キャッシュ付き） -----------------------------------------------
def _save_list(df: pd.DataFrame, cache: Path) -> None:
    """空（書類なしの日）もマーカー保存し by-date ミラーを冪等化（再取得しない）。"""
    cache.parent.mkdir(parents=True, exist_ok=True)
    (df if not df.empty
     else pd.DataFrame({"_empty": pd.Series([], dtype="bool")})).to_parquet(cache)


def _read_list_cache(cache: Path) -> pd.DataFrame:
    df = pd.read_parquet(cache)
    if "_empty" in df.columns:                   # 空マーカー → 全列付きの空 df を返す
        return parse_documents([])
    return df


def fetch_documents_list(date, doc_type: int = 2, api_key: Optional[str] = None,
                         refresh: bool = False) -> pd.DataFrame:
    """指定日（YYYY-MM-DD or YYYYMMDD）の提出書類一覧。by-date で Parquet キャッシュ。

    refresh=False（既定）は既存キャッシュをそのまま返す＝初回観測スナップショットを
    保全（取下げ・職員修正・縦覧満了で API が過去レコードを書き換える前の状態を固定）。
    """
    ymd = _ymd(date)
    cache = _CACHE / "list" / f"{ymd}.parquet"
    if cache.exists() and not refresh:
        return _read_list_cache(cache)
    q = urllib.parse.urlencode({"date": _iso(date), "type": doc_type,
                                "Subscription-Key": _api_key(api_key)})
    data = _request_json(f"{_BASE}/documents.json?{q}")
    df = parse_documents(data.get("results") or [])
    _save_list(df, cache)
    return df


def fetch_document(doc_id: str, doc_type: int = 5, api_key: Optional[str] = None,
                   refresh: bool = False) -> Path:
    """書類本体を取得しローカル保存、保存パスを返す（成功/失敗は Content-Type 判定）。

    doc_type: 1=本文ZIP(XBRL) / 2=PDF / 3=添付ZIP / 4=英文ZIP / 5=CSV(ZIP)。
    既定 5（XBRL→CSV 変換済み＝タクソノミ処理なしで項目抽出可。csvFlag=1 の書類のみ）。
    """
    base = _CACHE / "docs"
    for ext in ("zip", "pdf"):                   # 既存（拡張子不明）を再利用
        p = base / f"{doc_id}_{doc_type}.{ext}"
        if p.exists() and not refresh:
            return p
    q = urllib.parse.urlencode({"type": doc_type,
                                "Subscription-Key": _api_key(api_key)})
    body, ext = _request_binary(f"{_BASE}/documents/{doc_id}?{q}")
    base.mkdir(parents=True, exist_ok=True)
    out = base / f"{doc_id}_{doc_type}.{ext}"
    out.write_bytes(body)
    return out


# --- フィルタ（C1/C2 の書類抽出。純関数） ---------------------------------
def tob_documents(df: pd.DataFrame, include_self: bool = False,
                  doc_types: tuple = TOB_DOC_TYPES) -> pd.DataFrame:
    """公開買付関連書類（C1）。既定は第三者TOB（府令 040）のみ＝自社株TOB（050）除外。"""
    if df.empty:
        return df
    ords = [ORD_TOB_THIRD_PARTY] + ([ORD_TOB_SELF] if include_self else [])
    m = df["ordinanceCode"].isin(ords)
    if doc_types:
        m = m & df["docTypeCode"].isin(list(doc_types))
    return df[m]


def large_holding_documents(df: pd.DataFrame) -> pd.DataFrame:
    """大量保有関連書類（C2）。府令 060 で引く（350/360 ＋ 変更報告書を取りこぼさない）。

    docs/08 §2.1：変更報告書の docTypeCode は別紙1 で要確認だが、府令 060 で一覧を引けば
    様式に依らず取りこぼしが無い。
    """
    if df.empty:
        return df
    return df[df["ordinanceCode"] == ORD_LARGE_HOLDING]


def annual_report_documents(df: pd.DataFrame,
                            doc_types: tuple = ANNUAL_DOC_TYPES) -> pd.DataFrame:
    """有価証券報告書（三表明細＝ファンダ特徴量ソース）。既定は 120/130（有報・訂正有報）。

    by-date 一覧から docTypeCode で抽出。secCode を持つ提出のみ（ファンド等を除外）。
    """
    if df.empty:
        return df
    m = df["docTypeCode"].isin(list(doc_types)) & df["secCode"].notna()
    return df[m]
