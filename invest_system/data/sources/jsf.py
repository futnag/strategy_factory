"""日本証券金融（JSF）貸借取引データ — 借株コスト（逆日歩）と需給の前向き蓄積。

**用途**: ショートの執行コストを実値化する（`engine` の `short_borrow_bps` 既定は ~115bps の
保守仮定にすぎない）。逆日歩（品貸料）= 制度信用の貸株が逼迫した銘柄に課される 円/株/日 の追加コスト
＝(a) 全ショート戦略（value L/S 等）の**実コスト化**、(b) **踏み上げ/クラウディング信号**そのもの。
これは docs/03 の「独立α探索」ではなく**プラットフォームの借株現実化**＝既存スリーブの執行精度向上。

**重要（限界）**: ストップ高リバーサル（docs/48）は**取引コストで死ぬ（借株は二次的）**ため、本データでも
判定は変わらない（docs/49 §0）。価値は value L/S 等の借株現実化と踏み上げ回避。

**出所・ToS**: JSF（https://www.jsf.co.jp/）が貸借取引残高・逆日歩・規制措置を**日次公表**（公開ファイル）。
公式 API なし。リポジトリ規律に従い**手動DLのみ・自動スクレイピングはしない**（M&A Online と同じ。
ToSTNeT/TDnet は JPX/TDnet の静的公開ページを薄く取得しているが、JSF はサイト構造/ToS を確認するまで
ネット取得関数を同梱しない＝**手動DLした生ファイルを `ingest_dir` で parquet 化**する設計）。
代替の公式ソース＝**J-Quants Premium の貸借**（契約時。スキーマを CANON に合わせれば合流可能）。

**PIT規律**: 取引日 T の逆日歩は T+1 に確定・公表。バックテストでは公表日アンカー（≤t-lag）で参照する
（本モジュールは `Date`=基準日で保存し、ラグ適用は呼び出し側の責務＝§ `borrow_cost_bps_panel` 注記）。

設計は `tdnet.py` / `jpx_tostnet.py` に倣い、生データ→正準スキーマ変換を純関数（`parse_jsf_csv`）に分離して
オフライン検証可能にする。取得データは `data/jsf/`（gitignore）にのみ保持。
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

# 正準スキーマ（J-Quants Premium 貸借と将来合流できる粒度）
CANON_COLUMNS = [
    "Date", "Code", "loan_balance", "lending_balance", "net_balance",
    "premium_rate", "regulation", "source",
]

# 実ヘッダ（日/英トークン）→ 正準列名。部分一致・二重割当防止（jpx_tostnet と同方式）。
_ALIASES: list[tuple[str, tuple[str, ...]]] = [
    ("Date", ("基準日", "申込日", "日付", "date")),
    ("Code", ("銘柄コード", "コード", "code")),
    ("loan_balance", ("融資残高", "融資残", "loan_balance", "loan")),
    ("lending_balance", ("貸株残高", "貸株残", "lending_balance", "stock_loan")),
    ("net_balance", ("差引残", "差引", "貸借差引", "net")),
    ("premium_rate", ("逆日歩", "品貸料率", "品貸料", "premium", "backwardation")),
    ("regulation", ("規制措置", "規制", "注意喚起", "貸借申込", "regulation", "status")),
]


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=CANON_COLUMNS)


def normalize_code(raw) -> str:
    """JSF コード → J-Quants 風5桁文字列（4桁数字は末尾 0 補完・英字混じりは大文字化）。"""
    s = str(raw).strip().upper().replace(".0", "")
    if not s:
        return s
    if len(s) == 4 and s.isdigit():
        return s + "0"
    return s


def _to_float(s: pd.Series) -> pd.Series:
    """カンマ区切り文字列（'1,234,500'）→ float。空/欠損/'-' は NaN。"""
    cleaned = (s.astype(str).str.replace(",", "", regex=False).str.strip()
               .replace({"": None, "nan": None, "None": None, "-": None, "－": None}))
    return pd.to_numeric(cleaned, errors="coerce").astype("float64")


def _to_date(s: pd.Series) -> pd.Series:
    """'YYYYMMDD' / 'YYYY/MM/DD' / 'YYYY-MM-DD' を datetime64 に。"""
    txt = (s.astype(str).str.strip()
           .replace({"": None, "nan": None, "None": None, "NaT": None}))
    out = pd.Series(pd.NaT, index=txt.index, dtype="datetime64[ns]")
    is8 = (txt.str.fullmatch(r"\d{8}") == True)  # noqa: E712
    if is8.any():
        out[is8] = pd.to_datetime(txt[is8], format="%Y%m%d", errors="coerce")
    rest = (~is8) & txt.notna()
    if rest.any():
        out[rest] = pd.to_datetime(txt[rest], errors="coerce")
    return out


def _map_columns(cols) -> dict:
    """実ヘッダ → 正準列名の対応（部分一致・二重割当を防止）。"""
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


def parse_jsf_csv(content, *, default_date: str | None = None) -> pd.DataFrame:
    """JSF 貸借取引の生 CSV/TSV（手動DL）→ 正準スキーマ DataFrame（純関数・ネット不要）。

    タイトル行が先頭にあってもヘッダ行（'銘柄コード'/'コード'/'code' を含む最初の行）を自動検出。
    数値はカンマ除去、Code は文字列（'285A0' 等の英数字対応）。`Date` 列が無いファイルは
    `default_date`（'YYYYMMDD'）で補完できる。該当行なしは空フレーム。
    """
    text = (content.decode("utf-8", errors="replace")
            if isinstance(content, (bytes, bytearray)) else str(content))
    lines = text.splitlines()
    hdr_i = None
    for i, ln in enumerate(lines[:15]):
        if "銘柄コード" in ln or "コード" in ln or "code" in ln.lower():
            hdr_i = i
            break
    if hdr_i is None:
        return _empty_frame()
    sep = "\t" if "\t" in lines[hdr_i] else ","
    try:
        raw = pd.read_csv(io.StringIO("\n".join(lines[hdr_i:])), sep=sep, dtype=str)
    except Exception:  # noqa: BLE001
        return _empty_frame()
    if raw.empty:
        return _empty_frame()
    colmap = _map_columns(raw.columns)
    out = pd.DataFrame(index=raw.index)
    for canon in CANON_COLUMNS:
        if canon == "source":
            continue
        src = colmap.get(canon)
        out[canon] = raw[src] if src is not None else pd.NA

    if out["Date"].isna().all() and default_date:
        out["Date"] = default_date
    out["Date"] = _to_date(out["Date"])
    out["Code"] = out["Code"].map(normalize_code)
    for c in ("loan_balance", "lending_balance", "net_balance", "premium_rate"):
        out[c] = _to_float(out[c])
    # 差引が無ければ 融資−貸株 で補完
    miss_net = out["net_balance"].isna()
    out.loc[miss_net, "net_balance"] = (
        out.loc[miss_net, "loan_balance"] - out.loc[miss_net, "lending_balance"])
    out["regulation"] = (out["regulation"].astype(str).str.strip()
                         .replace({"nan": "", "None": "", "<NA>": ""}))
    out["source"] = "jsf_manual"

    valid = out["Code"].notna() & (out["Code"].astype(str).str.len() > 0) \
        & ~out["Code"].astype(str).str.lower().isin(["nan", "none", "<na>"])
    out = out[valid]
    if out.empty:
        return _empty_frame()
    return out[CANON_COLUMNS].reset_index(drop=True)


# --- 前向き蓄積（手動DL生ファイル → parquet・冪等） ----------------------
def ingest_dir(raw_dir: str, cache_dir: str = "data/jsf", *,
               glob: str = "*.csv", overwrite: bool = False,
               verbose: bool = True) -> dict:
    """`raw_dir` の手動DL生ファイルを parse し `{cache_dir}/{YYYYMMDD}.parquet` へ（冪等）。

    ファイル名先頭の 8 桁（YYYYMMDD）を既定日付に使う（`Date` 列があればそちら優先）。
    既存日付はスキップ（overwrite=True で上書き）。**ネットワーク不要**＝手動DL運用専用。
    """
    raw = Path(raw_dir)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    rep = {"files": 0, "written": 0, "skipped": 0, "rows": 0, "errors": 0}
    for fp in sorted(raw.glob(glob)):
        rep["files"] += 1
        stem8 = "".join(ch for ch in fp.stem if ch.isdigit())[:8] or None
        try:
            df = parse_jsf_csv(fp.read_bytes(), default_date=stem8)
        except Exception as e:  # noqa: BLE001
            rep["errors"] += 1
            if verbose:
                print(f"  [warn] {fp.name}: {str(e)[:80]}")
            continue
        if df.empty:
            continue
        for d, sub in df.groupby(df["Date"].dt.strftime("%Y%m%d")):
            out_fp = cache / f"{d}.parquet"
            if out_fp.exists() and not overwrite:
                rep["skipped"] += 1
                continue
            sub.reset_index(drop=True).to_parquet(out_fp)
            rep["written"] += 1
            rep["rows"] += len(sub)
    if verbose:
        print(f"  jsf: ファイル{rep['files']} / 書込{rep['written']}"
              f"（行{rep['rows']}）/ スキップ{rep['skipped']} / エラー{rep['errors']}")
    return rep


def load_jsf(cache_dir: str = "data/jsf",
             start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """蓄積済み JSF 貸借データを1本に（`_empty` マーカーは除外）。"""
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
    return (pd.concat(frames, ignore_index=True)
            .sort_values(["Date", "Code"]).reset_index(drop=True))


# --- 借株コスト・需給シグナル（wide 変換） --------------------------------
def premium_panel(jsf_long: pd.DataFrame) -> pd.DataFrame:
    """逆日歩（円/株/日）の wide パネル（index=Date, col=Code）。無ければ 0。"""
    if jsf_long.empty:
        return pd.DataFrame()
    return jsf_long.pivot_table(index="Date", columns="Code",
                                values="premium_rate", aggfunc="last")


def borrow_cost_bps_panel(jsf_long: pd.DataFrame, close: pd.DataFrame, *,
                          base_bps: float = 115.0, ann_days: int = 245) -> pd.DataFrame:
    """逆日歩 → **年率 bps** の借株コスト wide パネル（engine のショート控除に接続）。

    逆日歩は 円/株/日。日次コスト率 = 逆日歩 / 株価、年率 bps = (率)×ann_days×1e4 ＋ base_bps
    （制度貸株料の床 ~115bps）。`close`（生株価 wide・円/株）の (Date×Code) で整列し、逆日歩が無い
    セルは床 `base_bps` のみ。**PIT**: 逆日歩は T+1 確定のため、呼び出し側で公表ラグ（shift）を
    掛けてから使うこと（先読み防止）。engine 統合は docs/49 §3 を参照（per-name 借株は
    `costs_bps` パネル併用 or `short_borrow_bps` のパネル化拡張）。
    """
    if close is None or close.empty:
        return pd.DataFrame()
    pr = premium_panel(jsf_long)
    contrib = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    if not pr.empty:
        daily = (pr / close.reindex(index=pr.index, columns=pr.columns))
        contrib = (daily * ann_days * 1e4).reindex(
            index=close.index, columns=close.columns).fillna(0.0)
    return contrib + float(base_bps)


def loan_lending_ratio(jsf_long: pd.DataFrame) -> pd.DataFrame:
    """貸借倍率 = 融資残/貸株残 の wide（< 1 ＝ 貸株超過＝クラウディング/踏み上げ余地）。"""
    if jsf_long.empty:
        return pd.DataFrame()
    loan = jsf_long.pivot_table(index="Date", columns="Code",
                                values="loan_balance", aggfunc="last")
    lend = jsf_long.pivot_table(index="Date", columns="Code",
                                values="lending_balance", aggfunc="last")
    return loan / lend.where(lend > 0)


def squeeze_flags(jsf_long: pd.DataFrame) -> pd.DataFrame:
    """逆日歩発生（>0）＝借株逼迫＝踏み上げリスクの bool wide。"""
    pr = premium_panel(jsf_long)
    if pr.empty:
        return pd.DataFrame()
    return pr.fillna(0.0) > 0.0
