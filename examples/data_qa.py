"""データ品質監査（/data-qa の決定論的コア）— flag, don't clean.

検査クラス（一次資料: QuantStart/Databento/Backtrex チェックリスト・J-Quants 仕様・
Norgate 上場廃止規約。research_loop/surveys/2026-07-04 参照）:
  [nightly 既定]
  1. wide 概観（store.health_check 再利用: 被覆・NaN率・close 整合）
  2. バー不変条件: High≥max(O,C), Low≤min(O,C), High≥Low, 出来高>0 で価格>0
  3. カレンダー/ミラー整合: wide index の重複/単調性・by-date ファイルとの相互包含
  4. 調整整合: adj_close ≈ close×Π(adj_factor[s>t]) の再計算 diff
  5. 外れ値スクリーン: |日次リターン|>25% を adj_factor・UL/LL と突合（正当事由なしのみ WARN）
  6. リステートメント検知: ファイル SHA-256 台帳と直近窓の再ハッシュ diff（bitemporal ログ）
  7. スキーマ契約: 主要ディレクトリの列集合/型 vs research_ops/data_contract.json
  [--deep 追加]
  8. 全期間の調整整合・全ファイルハッシュ・上場廃止監査（消滅→復活の穴・終端の切れ方）

出力: コンソール要約＋research_ops/data_qa_log.jsonl へ findings を append（自動修復はしない）。
状態: data/qa/file_hashes.parquet（gitignore 圏・ハッシュ台帳）。
実行: $env:PYTHONUTF8="1"; .venv\\Scripts\\python.exe examples\\data_qa.py [--deep|--init-contract]
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from invest_system.data.store import health_check, load_wide  # noqa: E402

DATA = ROOT / "data"
OPS = ROOT / "research_ops"
LOG = OPS / "data_qa_log.jsonl"
CONTRACT = OPS / "data_contract.json"
HASHES = DATA / "qa" / "file_hashes.parquet"
CONTRACT_DIRS = ["jquants/daily", "jquants/fins_summary", "jquants/margin_weekly",
                 "jquants/options_225", "jquants/margin_alert", "edinet/list", "tdnet"]
RET_OUTLIER = 0.25
NIGHTLY_DAYS = 45

FINDINGS: list[dict] = []


def finding(check: str, severity: str, count: int, detail: str) -> None:
    FINDINGS.append({"check": check, "severity": severity, "count": int(count),
                     "detail": detail[:400]})
    mark = {"INFO": "·", "WARN": "⚠", "FAIL": "✗"}[severity]
    print(f"  {mark} [{severity}] {check}: {detail}" + (f"（{count}件）" if count else ""))


def _latest_file(d: Path) -> Path | None:
    """最新の**実データ**ファイル（_empty マーカー・0行ファイルはスキップ）。"""
    import pyarrow.parquet as pq
    for p in sorted((p for p in d.glob("*.parquet") if p.is_file()), reverse=True):
        try:
            if ("_empty" not in pq.read_schema(p).names
                    and pq.read_metadata(p).num_rows > 0):
                return p
        except Exception:  # noqa: BLE001
            continue
    return None


# --- 1. wide 概観 -----------------------------------------------------------
def check_wide_overview() -> None:
    print("\n== 1. wide 概観（health_check）==")
    hc = health_check(str(DATA))
    if hc.empty:
        finding("wide_overview", "FAIL", 0, "wide ストアが空")
        return
    print(hc.to_string(index=False))
    bad = hc[~hc["aligned_to_close"]]
    if len(bad):
        finding("wide_overview", "WARN", len(bad),
                f"close と日付不整合のフィールド: {bad['field'].tolist()}")
    else:
        finding("wide_overview", "INFO", 0, f"全{len(hc)}フィールド close 整合")


# --- 2. バー不変条件 --------------------------------------------------------
def check_bar_invariants(deep: bool) -> None:
    print("\n== 2. バー不変条件 ==")
    o, h, l_, c = (load_wide(f) for f in ("open", "high", "low", "close"))
    v = load_wide("volume")
    if not deep:
        o, h, l_, c, v = (d.tail(NIGHTLY_DAYS) for d in (o, h, l_, c, v))
    traded = v > 0
    viol_h = ((h < o) | (h < c)) & traded & h.notna()
    viol_l = ((l_ > o) | (l_ > c)) & traded & l_.notna()
    viol_hl = (h < l_) & traded
    nonpos = ((c <= 0) | (o <= 0)) & traded
    for name, m in [("high<max(O,C)", viol_h), ("low>min(O,C)", viol_l),
                    ("high<low", viol_hl), ("価格<=0", nonpos)]:
        n = int(m.to_numpy().sum())
        if n:
            idx = m.any(axis=1)
            finding("bar_invariants", "FAIL", n,
                    f"{name} 例: {m.loc[idx].stack()[lambda s: s].index[:3].tolist()}")
        else:
            finding("bar_invariants", "INFO", 0, f"{name} 違反なし")


# --- 3. カレンダー / ミラー整合 ---------------------------------------------
def check_calendar(deep: bool) -> None:
    print("\n== 3. カレンダー/ミラー整合 ==")
    c = load_wide("close")
    idx = pd.to_datetime(c.index)
    if idx.duplicated().any():
        finding("calendar", "FAIL", int(idx.duplicated().sum()), "wide index に重複日付")
    if not idx.is_monotonic_increasing:
        finding("calendar", "FAIL", 0, "wide index が非単調")
    wk = idx[idx.weekday >= 5]
    if len(wk):
        finding("calendar", "WARN", len(wk), f"週末日付が混入: {[str(d.date()) for d in wk[:3]]}")
    # 祝日等の空マーカーファイル（_empty 列のみ）はミラー側集合から除外（誤検知防止）
    files = set()
    for f in (DATA / "jquants" / "daily").glob("*.parquet"):
        try:
            import pyarrow.parquet as pq
            if ("_empty" in pq.read_schema(f).names
                    or pq.read_metadata(f).num_rows == 0):
                continue
        except Exception:  # noqa: BLE001
            pass
        files.add(f.stem)
    widedays = {d.strftime("%Y%m%d") for d in idx}
    scope = widedays if deep else {d.strftime("%Y%m%d") for d in idx[-NIGHTLY_DAYS:]}
    miss_in_files = sorted(scope - files)
    miss_in_wide = sorted({f for f in files if f >= min(scope, default="99999999")} - widedays)
    if miss_in_files:
        finding("calendar", "FAIL", len(miss_in_files),
                f"wide にあるが by-date ミラーに無い日: {miss_in_files[:5]}")
    if miss_in_wide:
        finding("calendar", "WARN", len(miss_in_wide),
                f"ミラーにあるが wide 未反映の日: {miss_in_wide[:5]}（materialize 遅延?）")
    if not miss_in_files and not miss_in_wide:
        finding("calendar", "INFO", 0, f"wide⇔ミラー整合（{len(scope)}日分）")


# --- 4. 調整整合 -------------------------------------------------------------
def check_adjustment(deep: bool) -> None:
    print("\n== 4. 調整整合（adj_close 再計算）==")
    close = load_wide("close")
    adj = load_wide("adj_close")
    fac = load_wide("adj_factor")
    if not deep:
        # 直近1年＋直近に adj_factor≠1 があった銘柄を重点
        close, adj, fac = (d.tail(260) for d in (close, adj, fac))
    f = fac.fillna(1.0)
    cum = f.iloc[::-1].cumprod().iloc[::-1].shift(-1, fill_value=1.0)
    recomputed = close * cum
    denom = adj.abs().where(adj.abs() > 1e-9)
    rel = ((recomputed - adj).abs() / denom)
    bad = (rel > 1e-4) & adj.notna() & close.notna()
    n = int(bad.to_numpy().sum())
    if n:
        cells = bad.stack()[lambda s: s].index[:5].tolist()
        finding("adjustment", "FAIL", n, f"再計算と保存値の乖離>1e-4 例: {cells}"
                                         "（rebuild_adjusted の実行漏れ or 係数欠損疑い）")
    else:
        finding("adjustment", "INFO", 0, "adj_close 再計算一致")
    # 生値ジャンプ×係数=1（未調整コーポレートアクション疑い）
    r_raw = close.pct_change(fill_method=None)
    susp = (r_raw.abs() > 0.4) & (f == 1.0)
    m = int(susp.to_numpy().sum())
    if m:
        finding("adjustment", "WARN", m,
                f"|生リターン|>40% かつ adj_factor=1 の日（要目視）例: "
                f"{susp.stack()[lambda s: s].index[:5].tolist()}")


# --- 5. 外れ値スクリーン ------------------------------------------------------
def check_outliers(deep: bool) -> None:
    print("\n== 5. 外れ値/ステイル・スクリーン ==")
    adj = load_wide("adj_close")
    ul = load_wide("upper_limit")
    ll = load_wide("lower_limit")
    v = load_wide("volume")
    if not deep:
        adj, ul, ll, v = (d.tail(60) for d in (adj, ul, ll, v))
    r = adj.pct_change(fill_method=None)
    big = r.abs() > RET_OUTLIER
    lim = (ul == 1) | (ll == 1)
    unexplained = big & ~lim & ~lim.shift(1, fill_value=False)
    n = int(unexplained.to_numpy().sum())
    tot = int(big.to_numpy().sum())
    if n:
        finding("outliers", "WARN", n,
                f"|リターン|>{RET_OUTLIER:.0%} かつ値幅制限で説明されない: "
                f"{unexplained.stack()[lambda s: s].index[:5].tolist()}（全外れ値{tot}件中）")
    else:
        finding("outliers", "INFO", tot, "値幅制限外の異常リターンなし")
    stale = ((v == 0) & (adj.diff() == 0.0)).rolling(5).sum() >= 5
    ns = int(stale.any(axis=0).sum())
    finding("outliers", "INFO", ns, f"5日以上のゼロ出来高・同値継続あり銘柄（停止/パディング相当）")


# --- 6. リステートメント検知 --------------------------------------------------
def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check_restatement(deep: bool) -> None:
    print("\n== 6. リステートメント検知（ハッシュ台帳）==")
    HASHES.parent.mkdir(parents=True, exist_ok=True)
    ledger = (pd.read_parquet(HASHES) if HASHES.exists()
              else pd.DataFrame(columns=["path", "sha256", "first_seen", "last_seen"]))
    known = dict(zip(ledger["path"], ledger["sha256"]))
    targets: list[Path] = []
    for d in CONTRACT_DIRS:
        base = DATA / d
        if not base.exists():
            continue
        fs = sorted(base.glob("*.parquet"))
        targets += fs if deep else fs[-NIGHTLY_DAYS:]
    now = datetime.now().isoformat(timespec="seconds")
    changed, new = [], 0
    rows = []
    for p in targets:
        rel = str(p.relative_to(DATA))
        h = _sha(p)
        if rel in known and known[rel] != h:
            changed.append(rel)
        elif rel not in known:
            new += 1
        rows.append({"path": rel, "sha256": h, "first_seen": known.get(rel) and
                     ledger.loc[ledger["path"] == rel, "first_seen"].iloc[0] or now,
                     "last_seen": now})
    upd = pd.DataFrame(rows)
    ledger = pd.concat([ledger[~ledger["path"].isin(upd["path"])], upd],
                       ignore_index=True)
    ledger.to_parquet(HASHES)
    if changed:
        finding("restatement", "WARN", len(changed),
                f"既知ファイルの内容変更を検知（bitemporal 記録済み）: {changed[:5]}"
                "（事前登録済み結果の再現性に影響し得る＝要確認）")
    else:
        finding("restatement", "INFO", new,
                f"変更なし（新規登録 {new}件・台帳 {len(ledger):,}件・対象 {len(targets):,}件）")


# --- 7. スキーマ契約 ----------------------------------------------------------
def check_schema(init: bool) -> None:
    print("\n== 7. スキーマ契約 ==")
    current: dict[str, dict] = {}
    for d in CONTRACT_DIRS:
        f = _latest_file(DATA / d)
        if f is None:
            continue
        df = pd.read_parquet(f)
        current[d] = {"columns": sorted(map(str, df.columns)),
                      "dtypes": {str(c): str(t) for c, t in df.dtypes.items()}}
    if init or not CONTRACT.exists():
        CONTRACT.parent.mkdir(parents=True, exist_ok=True)
        CONTRACT.write_text(json.dumps(current, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        finding("schema", "INFO", len(current), f"契約を初期化/更新: {CONTRACT.name}")
        return
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    drift = 0
    for d, spec in contract.items():
        if d not in current:
            finding("schema", "WARN", 0, f"{d}: データ不在")
            continue
        miss = set(spec["columns"]) - set(current[d]["columns"])
        added = set(current[d]["columns"]) - set(spec["columns"])
        if miss or added:
            drift += 1
            finding("schema", "FAIL" if miss else "WARN", len(miss) + len(added),
                    f"{d}: 欠落{sorted(miss)[:5]} 追加{sorted(added)[:5]}")
    if not drift:
        finding("schema", "INFO", len(contract), "全ディレクトリ契約一致")


# --- 8. 上場廃止監査（deep）---------------------------------------------------
def check_delisting() -> None:
    print("\n== 8. 上場廃止監査（deep）==")
    c = load_wide("close")
    idx = c.index
    last_valid = c.apply(lambda s: s.last_valid_index())
    ended = last_valid[last_valid < idx[-30]]
    finding("delisting", "INFO", len(ended),
            f"直近30営業日より前に終了した銘柄（廃止/変更相当・パネル脱落方式）")
    # 消滅→復活（>5日の穴）
    holes = 0
    sample = []
    for code in c.columns[:: max(1, len(c.columns) // 800)]:  # サンプリング監査
        s = c[code].dropna()
        if len(s) < 10:
            continue
        gaps = pd.Series(s.index).diff().dt.days
        if (gaps > 10).any():
            holes += 1
            if len(sample) < 5:
                sample.append(code)
    if holes:
        finding("delisting", "WARN", holes,
                f"時系列に>10暦日の穴のある銘柄（サンプル監査）: {sample}（再上場/データ穴の別を要確認）")


def main(argv: list[str]) -> int:
    deep = "--deep" in argv
    init = "--init-contract" in argv
    print(f"データ品質監査  {datetime.now():%Y-%m-%d %H:%M}  mode={'deep' if deep else 'nightly'}")
    check_wide_overview()
    check_bar_invariants(deep)
    check_calendar(deep)
    check_adjustment(deep)
    check_outliers(deep)
    check_restatement(deep)
    check_schema(init)
    if deep:
        check_delisting()

    n_fail = sum(1 for f in FINDINGS if f["severity"] == "FAIL")
    n_warn = sum(1 for f in FINDINGS if f["severity"] == "WARN")
    print(f"\n== 要約: FAIL={n_fail}  WARN={n_warn}  "
          f"INFO={len(FINDINGS) - n_fail - n_warn} ==")
    OPS.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"run": datetime.now().isoformat(timespec='seconds'),
                             "mode": "deep" if deep else "nightly",
                             "fail": n_fail, "warn": n_warn,
                             "findings": [f for f in FINDINGS
                                          if f["severity"] != "INFO"]},
                            ensure_ascii=False) + "\n")
    print(f"（findings を {LOG.relative_to(ROOT)} に追記。自動修復はしない＝flag, don't clean）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
