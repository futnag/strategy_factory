"""J-Quants V2 ライブ・スモークテスト（要 .env の J_QUANTS_API_KEY）。

APIキーは表示しない。取得データの形状・列名・キャッシュ生成のみを確認する。
無料プランは約12週の配信遅延があるため、日次株価は十分過去の日付を使い、
祝日・休場で空ならば数日遡って再試行する。

実行例（Windows）:  .venv\\Scripts\\python.exe examples\\jquants_smoke.py
実行例（Linux）  :  .venv/bin/python examples/jquants_smoke.py
"""
from __future__ import annotations

import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.config import get_env  # noqa: E402
from invest_system.data.sources import jquants as jq  # noqa: E402


def mask_check() -> bool:
    """キーの有無と長さのみ表示（値は出さない）。"""
    key = get_env("J_QUANTS_API_KEY")
    print(f"[key] present={bool(key)} length={len(key) if key else 0}")
    return bool(key)


def probe_raw(path: str, params: dict) -> None:
    """生レスポンスの最上位キー構造を確認（初回接続デバッグ用）。"""
    try:
        key = jq._api_key()
    except Exception as e:  # noqa: BLE001
        print(f"[probe] no key: {e}")
        return
    import json as _json

    url = f"{jq._BASE}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"x-api-key": key})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = _json.load(resp)
        print(f"[probe] {path} top-level keys: {list(data.keys())}")
        for k, v in data.items():
            if isinstance(v, list):
                sample = list(v[0].keys()) if v else "[]"
                print(f"        '{k}': list(len={len(v)}) sample_fields={sample}")
            else:
                print(f"        '{k}': {type(v).__name__}={v!r}")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")[:300]
        print(f"[probe] {path} HTTP {e.code}: {body}")
    except Exception as e:  # noqa: BLE001
        print(f"[probe] {path} ERROR: {e}")


def check_listed_info() -> None:
    print("\n=== 1) 上場銘柄マスタ /equities/master ===")
    try:
        info = jq.fetch_listed_info()
    except Exception as e:  # noqa: BLE001
        print(f"[listed_info] ERROR: {e}")
        probe_raw("/equities/master", {})
        return
    print(f"[listed_info] shape={info.shape}")
    print(f"[listed_info] columns={list(info.columns)}")
    if info.empty:
        print("[listed_info] EMPTY -> probing raw response")
        probe_raw("/equities/master", {})
        return
    cols = [c for c in ("Code", "CoNameEn", "S33", "S33Nm", "Mkt", "MktNm",
                        "ScaleCat")
            if c in info.columns]
    print(info[cols].head(5).to_string(index=False))


def check_daily_quotes() -> None:
    print("\n=== 2) 日次株価 /equities/bars/daily（無料枠 ~12週遅延） ===")
    base = pd.Timestamp("2026-02-25")  # 余裕をもった過去日（祝日なら遡る）
    got = None
    for i in range(8):
        d = (base - pd.Timedelta(days=i)).strftime("%Y%m%d")
        try:
            q = jq.fetch_daily_quotes(d)
        except Exception as e:  # noqa: BLE001
            print(f"[daily {d}] ERROR: {e}")
            if i == 0:
                probe_raw("/equities/bars/daily", {"date": d})
            return
        if not q.empty:
            got = (d, q)
            break
        print(f"[daily {d}] empty（休場?）-> 前日へ")
    if not got:
        print("[daily] 8日遡っても空。無料枠の配信範囲外の可能性。日付を見直す。")
        return
    d, q = got
    print(f"[daily {d}] shape={q.shape}")
    print(f"[daily {d}] columns={list(q.columns)}")
    try:
        ac = jq.adjusted_close_col(q)
        print(f"[daily {d}] adjusted_close_col -> '{ac}'")
    except KeyError as e:
        print(f"[daily {d}] adjusted_close_col 解決失敗: {e}")
    show = [c for c in ("Date", "Code", "O", "H", "L", "C", "AdjC", "Vo",
                        "Va", "AdjVo")
            if c in q.columns][:7]
    print(q[show].head(5).to_string(index=False))
    # dtype 確認（Vo/Va/AdjVo が数値であること = パーサ修正の検証）
    dtype_cols = [c for c in ("C", "AdjC", "Vo", "Va", "AdjVo", "AdjFactor")
                  if c in q.columns]
    print("[daily dtypes] " + ", ".join(f"{c}={q[c].dtype}" for c in dtype_cols))


def check_statements() -> None:
    print("\n=== 4) 財務 /fins/summary（ファンダ項目の確認・1銘柄プローブ） ===")
    code = "72030"  # トヨタ自動車（V2 は5桁コード: 7203 -> 72030）
    try:
        st = jq.fetch_statements(code=code)
    except Exception as e:  # noqa: BLE001
        print(f"[statements] ERROR: {e}")
        probe_raw("/fins/details", {"code": code})
        return
    print(f"[statements {code}] shape={st.shape}")
    if st.empty:
        print("[statements] EMPTY -> probing raw response")
        probe_raw("/fins/details", {"code": code})
        return
    print(f"[statements {code}] columns({len(st.columns)})={list(st.columns)}")
    # 最新開示の非NULL項目を一覧（利用可能なファンダ項目の把握）
    if "DisclosedDate" in st.columns:
        st = st.sort_values("DisclosedDate")
    row = st.iloc[-1]
    nonnull = [(c, row[c]) for c in st.columns
               if pd.notna(row[c]) and str(row[c]) != ""]
    print(f"[statements {code}] 最新開示の非NULL項目数={len(nonnull)}（先頭60件）:")
    for c, v in nonnull[:60]:
        sv = str(v)
        if len(sv) > 40:
            sv = sv[:40] + "…"
        print(f"    {c} = {sv}")


def check_cache() -> None:
    print("\n=== 3) Parquet キャッシュ ===")
    cache = Path("data/jquants")
    if not cache.exists():
        print("[cache] data/jquants は未生成")
        return
    files = sorted(p.relative_to(cache).as_posix() for p in cache.rglob("*.parquet"))
    print(f"[cache] {len(files)} files: {files[:12]}")


def main() -> int:
    print("J-Quants V2 ライブ・スモークテスト")
    if not mask_check():
        print("ERROR: .env に J_QUANTS_API_KEY を設定してください。")
        return 1
    check_listed_info()
    check_daily_quotes()
    check_statements()
    check_cache()
    print("\n完了。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
