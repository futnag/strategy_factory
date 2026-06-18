"""補助ソース edinetdb.jp REST クライアント（GKX Phase 2・#5）。

公式 EDINET（一次ソース）の **独立クロスチェック**・**公式 10 年窓から落ちた古年（FY2012-2015）の
穴埋め** に使う補助。**バルクには使わない**（Free=100req/日）。チャット側 MCP コネクタには依存せず
REST を直接叩く（handoff §1-B）。

- ベース `https://edinetdb.jp/v1`、認証ヘッダ `X-API-Key`（`.env` の `EDINET_DB_API_KEY`）。
- **100req/日スロットル**：日次カウンタを `data/edinet/edinetdb/quota.json` に永続化し、上限
  （既定 95＝余裕）に達したら QuotaExceeded を送出（翌日に回す）。キャッシュ読みは消費しない。
- `/companies/{edinet_code}/financials` は **FY2012〜最新**の時系列（accounting_standard・revenue・
  operating_income・net_income・total_assets・net_assets・cf_*・shares_issued・split_adjustment_factor 等）。
  公式の ~10 年窓（2016+）より古い 2012-2015 を持つ＝古年穴埋めの正味の価値。
- 加工済み比率（roe_official 等）は規律により使わない＝生ライン項目のみ採る。
- セグメント専用エンドポイントは Free REST に無い（保留）。

注意：実コード（{code}）は **EDINET コード（E始まり）**。secCode(5桁) からの変換は EDINET 一覧
ミラーの edinetCode↔secCode で行う（edinetdb の quota を消費しない）。
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from ...config import get_env
from . import edinet as ed

_BASE = "https://edinetdb.jp/v1"
_CACHE = ed._CACHE / "edinetdb"
_QUOTA_PATH = _CACHE / "quota.json"
_MIN_INTERVAL = float(get_env("EDINET_DB_MIN_INTERVAL", "0.5") or "0.5")
_DAILY_CAP = int(get_env("EDINET_DB_DAILY_CAP", "95") or "95")   # 100/日に余裕
_last_call = [0.0]

# edinetdb フィールド → 正準フィールド（生ライン項目のみ。比率は取り込まない）。
EDINETDB_TO_CANONICAL = {
    "revenue": "net_sales", "operating_income": "operating_income",
    "ordinary_income": "ordinary_income", "profit_before_tax": "pretax_income",
    "net_income": "profit", "total_assets": "total_assets",
    "net_assets": "net_assets", "shareholders_equity": "equity",
    "cf_operating": "cfo", "cf_investing": "cfi", "cf_financing": "cff",
    "cash": "cash", "shares_issued": "shares_outstanding",
}
# 公式パースと突合できる、定義が一致する正準フィールド（equity/net_assets は定義差で除外）。
CROSSCHECK_FIELDS = ["net_sales", "operating_income", "ordinary_income",
                     "pretax_income", "profit", "total_assets",
                     "cfo", "cfi", "cff", "cash"]


class QuotaExceeded(RuntimeError):
    """その日の edinetdb 取得上限（既定 95/日）に達した。翌日に回す。"""


class DailyQuota:
    """edinetdb の 1 日あたり取得回数を永続カウント（再実行・別プロセスでも継続）。"""

    def __init__(self, path: Path = _QUOTA_PATH, cap: int = _DAILY_CAP):
        self.path, self.cap = Path(path), cap
        self.day, self.count = date.today().isoformat(), 0
        if self.path.exists():
            try:
                d = json.loads(self.path.read_text(encoding="utf-8"))
                if d.get("day") == self.day:
                    self.count = int(d.get("count", 0))
            except Exception:  # noqa: BLE001
                pass

    def remaining(self) -> int:
        if self.day != date.today().isoformat():     # 日付跨ぎでリセット
            self.day, self.count = date.today().isoformat(), 0
        return max(0, self.cap - self.count)

    def consume(self) -> None:
        if self.remaining() <= 0:
            raise QuotaExceeded(f"edinetdb 日次上限 {self.cap} に到達（{self.day}）。翌日に回してください。")
        self.count += 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"day": self.day, "count": self.count}),
                             encoding="utf-8")


_quota = DailyQuota()


def _api_key(api_key: Optional[str] = None) -> str:
    key = api_key or get_env("EDINET_DB_API_KEY")
    if not key:
        raise RuntimeError("EDINET_DB_API_KEY が未設定（.env）。edinetdb 補助は使えません。")
    return key


def _throttle() -> None:
    dt = time.monotonic() - _last_call[0]
    if dt < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - dt)
    _last_call[0] = time.monotonic()


def _request(path: str, api_key: Optional[str] = None, count: bool = True) -> dict:
    """GET（X-API-Key）。count=True は日次クォータを 1 消費（上限で QuotaExceeded）。"""
    if count:
        _quota.consume()                              # 送信前に確保（429 連発を避ける）
    _throttle()
    req = urllib.request.Request(_BASE + path)
    req.add_header("X-API-Key", _api_key(api_key))
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "ignore")[:200]
        raise RuntimeError(f"edinetdb HTTP {e.code}: {detail}") from e


# --- パース（純関数・ネットワーク不要） -----------------------------------
def _norm_basis(s) -> Optional[str]:
    if not s:
        return None
    u = str(s).upper().replace(" ", "").replace("-", "")
    if "IFRS" in u:
        return "IFRS"
    if "US" in u:
        return "USGAAP"
    if "JP" in u or "JGAAP" in u or "JAPAN" in u:
        return "JGAAP"
    return u


def parse_financials(payload: dict, edinet_code: Optional[str] = None) -> pd.DataFrame:
    """/financials の JSON → 正準フィールドの長形式（行=会計年度）。

    列：edinet_code, fiscal_year, basis, ＋ EDINETDB_TO_CANONICAL の正準名。生額のみ。
    """
    rows = (payload or {}).get("data") or []
    if not rows:
        return pd.DataFrame(columns=["edinet_code", "fiscal_year", "basis"])
    df = pd.DataFrame(rows)
    out = pd.DataFrame(index=df.index)                 # 行数を先に確定（scalar 代入の展開のため）
    out["fiscal_year"] = pd.to_numeric(df.get("fiscal_year"), errors="coerce").astype("Int64")
    out["basis"] = df.get("accounting_standard").map(_norm_basis) if "accounting_standard" in df else None
    for src, dst in EDINETDB_TO_CANONICAL.items():
        out[dst] = pd.to_numeric(df.get(src), errors="coerce") if src in df.columns else pd.NA
    if "split_adjustment_factor" in df.columns:        # 分割調整係数（純株式発行の補正に有用）
        out["split_adjustment_factor"] = pd.to_numeric(df["split_adjustment_factor"], errors="coerce")
    out["edinet_code"] = edinet_code                   # 行確定後に代入（全行へ展開）
    return out.sort_values("fiscal_year").reset_index(drop=True)


def reconcile(ours: pd.DataFrame, edb: pd.DataFrame,
              fields=CROSSCHECK_FIELDS, tol: float = 0.01) -> pd.DataFrame:
    """公式パース long（1 銘柄）と edinetdb long を会計年度で突合し相対差を返す（純関数）。

    ours: period_end を持つ公式 long（1 Code）。edb: fiscal_year を持つ edinetdb long。
    返り値：(fiscal_year, field, official, edinetdb, rel_diff, flag[OK/WARN/MISS])。
    """
    o = ours.copy()
    o["fiscal_year"] = pd.to_datetime(o["period_end"]).dt.year
    rows = []
    e_by = edb.set_index("fiscal_year")
    for _, orow in o.iterrows():
        fy = orow["fiscal_year"]
        if fy not in e_by.index:
            continue
        erow = e_by.loc[fy]
        for f in fields:
            ov, ev = orow.get(f), erow.get(f)
            ov = None if ov is None or pd.isna(ov) else float(ov)
            ev = None if ev is None or pd.isna(ev) else float(ev)
            if ev is None:
                continue
            if ov is None:
                rows.append((fy, f, ov, ev, None, "MISS"))
                continue
            rel = abs(ov - ev) / max(abs(ev), 1.0)
            rows.append((fy, f, ov, ev, rel, "OK" if rel <= tol else "WARN"))
    return pd.DataFrame(rows, columns=["fiscal_year", "field", "official",
                                       "edinetdb", "rel_diff", "flag"])


# --- 取得（キャッシュ＋クォータ） -----------------------------------------
def fetch_financials(edinet_code: str, refresh: bool = False,
                     api_key: Optional[str] = None) -> pd.DataFrame:
    """edinetdb の財務時系列（正準 long）。Parquet キャッシュ。キャッシュ命中は quota を消費しない。"""
    cache = _CACHE / "financials" / f"{edinet_code}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    payload = _request(f"/companies/{urllib.parse.quote(edinet_code)}/financials", api_key)
    df = parse_financials(payload, edinet_code)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def fetch_company(edinet_code: str, refresh: bool = False,
                  api_key: Optional[str] = None) -> dict:
    """会社詳細（sec_code・is_delisted・data_years 等）。JSON キャッシュ。"""
    cache = _CACHE / "companies" / f"{edinet_code}.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text(encoding="utf-8"))
    payload = _request(f"/companies/{urllib.parse.quote(edinet_code)}", api_key)
    data = (payload or {}).get("data") or {}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def seccode_to_edinet(list_dir: Optional[Path] = None) -> dict[str, str]:
    """EDINET 一覧ミラーの edinetCode↔secCode から secCode(5桁)→edinetCode の対応表（quota 不要）。"""
    list_dir = Path(list_dir) if list_dir else ed._CACHE / "list"
    out: dict[str, str] = {}
    for p in sorted(list_dir.glob("*.parquet")):
        df = pd.read_parquet(p)
        if "secCode" not in df.columns or "edinetCode" not in df.columns:
            continue
        sub = df.dropna(subset=["secCode", "edinetCode"])
        for sc, ec in zip(sub["secCode"].astype(str), sub["edinetCode"].astype(str)):
            out.setdefault(sc, ec)
    return out


def old_year_backfill(edinet_code: str, max_year: int = 2015,
                      api_key: Optional[str] = None) -> pd.DataFrame:
    """公式 ~10 年窓から落ちた古年（既定 FY≤2015）の正準 long を edinetdb から取得。"""
    fin = fetch_financials(edinet_code, api_key=api_key)
    if fin.empty:
        return fin
    return fin[fin["fiscal_year"] <= max_year].reset_index(drop=True)


def remaining_quota() -> int:
    """本日の残り取得回数（クォータ管理用）。"""
    return _quota.remaining()
