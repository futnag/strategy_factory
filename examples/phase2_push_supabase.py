"""Phase 2 結果の Supabase 同期（無人運用の配管・D5）。

データフロー（ステートレス設計はそのまま）：
  pull : phase2_files テーブル → data/phase2/（fresh runner の状態復元）
  push : data/phase2/ → テーブル群＋phase2_files（ダッシュボードと永続正本を更新）

テーブル（書き込みは phase2_writer ロールのみ・RLS で REST anon は遮断済み）：
  phase2_status        … status.json を jsonb で 1 行
  phase2_months        … months.csv（全置換）
  phase2_equity_daily  … equity_daily.csv（全置換＝連鎖再計算で履歴も変わるため）
  phase2_orders        … orders_eq/ts_*.csv（全置換）
  phase2_runs          … 実行ログ（追記）
  phase2_files         … data/phase2/ の全ファイル（テキストは utf-8・parquet は base64）

接続は .env / 環境変数（PHASE2_DB_HOST / PHASE2_DB_PORT / PHASE2_DB_REF /
PHASE2_DB_PASSWORD_WRITER）。requires: psycopg[binary]（ops 専用依存）。

使い方:
  python examples/phase2_push_supabase.py --push [--kind nightly]
  python examples/phase2_push_supabase.py --pull
  python examples/phase2_push_supabase.py --log-failure "message" [--kind nightly]
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from invest_system.config import get_env  # noqa: E402

DIR = Path("data/phase2")
_TEXT_SUFFIX = {".json", ".csv", ".md"}


def encode_file(fp: Path) -> tuple[str, str]:
    """ファイル → (content, encoding)。テキストは utf-8、その他は base64。"""
    if fp.suffix in _TEXT_SUFFIX:
        return fp.read_text(encoding="utf-8"), "utf-8"
    return base64.b64encode(fp.read_bytes()).decode("ascii"), "base64"


def decode_file(content: str, encoding: str) -> bytes:
    """(content, encoding) → バイト列（pull の書き戻し用）。"""
    if encoding == "utf-8":
        return content.encode("utf-8")
    return base64.b64decode(content)


def order_rows(tag: str, sleeve: str, df: pd.DataFrame) -> list[tuple]:
    """orders CSV → phase2_orders 行（eq/ts の列差を吸収する純関数）。"""
    rows = []
    for _, r in df.iterrows():
        if sleeve == "switch":
            rows.append((tag, sleeve, str(r["code"]), float(r["weight"]),
                         float(r["price"]), float(r["shares"]), float(r["yen"]),
                         str(r.get("instruction", ""))))
        else:
            lots = r.get("lots")
            rows.append((tag, sleeve, str(r["asset"]), float(r["weight"]),
                         None, (float(lots) if pd.notna(lots) else None),
                         float(r["target_yen"]), str(r.get("instruction", ""))))
    return rows


def _num(v):
    """CSV 欠損（NaN）→ None（Postgres NULL）。"""
    return None if pd.isna(v) else float(v)


def connect():
    import psycopg

    host = get_env("PHASE2_DB_HOST")
    ref = get_env("PHASE2_DB_REF")
    pw = get_env("PHASE2_DB_PASSWORD_WRITER")
    if not (host and ref and pw):
        raise SystemExit("ERROR: PHASE2_DB_HOST / PHASE2_DB_REF / "
                         "PHASE2_DB_PASSWORD_WRITER が必要です（.env / 環境変数）。")
    return psycopg.connect(host=host, port=int(get_env("PHASE2_DB_PORT", "5432")),
                           dbname="postgres", user=f"phase2_writer.{ref}",
                           password=pw, sslmode="require", connect_timeout=20)


def push(kind: str) -> int:
    status_fp = DIR / "status.json"
    if not status_fp.exists():
        print("ERROR: data/phase2/status.json がありません。先に "
              "phase2_reconcile.py を実行してください。")
        return 1
    status = json.loads(status_fp.read_text(encoding="utf-8"))
    months = pd.read_csv(DIR / "months.csv")
    daily = pd.read_csv(DIR / "equity_daily.csv")
    orders: list[tuple] = []
    for fp in sorted(DIR.glob("orders_eq_*.csv")):
        tag = fp.stem.replace("orders_eq_", "")
        orders += order_rows(tag, "switch", pd.read_csv(fp, dtype={"code": str}))
    for fp in sorted(DIR.glob("orders_ts_*.csv")):
        tag = fp.stem.replace("orders_ts_", "")
        orders += order_rows(tag, "tsmom", pd.read_csv(fp))
    files = sorted(p for p in DIR.iterdir() if p.is_file())

    with connect() as conn, conn.cursor() as cur:
        cur.execute("insert into phase2_status (id, payload) values (1, %s) "
                    "on conflict (id) do update set payload = excluded.payload, "
                    "updated_at = now()", (json.dumps(status, ensure_ascii=False),))
        cur.execute("delete from phase2_months")
        for _, r in months.iterrows():
            cur.execute(
                "insert into phase2_months (month, status, val_date, ret_eq, ret_ts,"
                " combo_gross, combo_net, long_fill_yen, unfilled_names, hedge_yen,"
                " ts_live_gap_yen) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (str(r["month"]), str(r["status"]), str(r["val_date"]),
                 _num(r["ret_eq"]), _num(r["ret_ts"]), _num(r["combo_gross"]),
                 _num(r["combo_net"]), _num(r["long_fill_yen"]),
                 int(r["unfilled_names"]), _num(r["hedge_yen"]),
                 _num(r["ts_live_gap_yen"])))
        cur.execute("delete from phase2_equity_daily")
        for _, r in daily.iterrows():
            cur.execute("insert into phase2_equity_daily (date, eq, ts, combo_net) "
                        "values (%s,%s,%s,%s)",
                        (str(r["date"]), _num(r["eq"]), _num(r["ts"]),
                         _num(r["combo_net"])))
        cur.execute("delete from phase2_orders")
        for row in orders:
            cur.execute("insert into phase2_orders (month, sleeve, key, weight, "
                        "price, qty, yen, instruction) "
                        "values (%s,%s,%s,%s,%s,%s,%s,%s)", row)
        for fp in files:
            content, enc = encode_file(fp)
            cur.execute("insert into phase2_files (path, content, encoding) "
                        "values (%s,%s,%s) on conflict (path) do update set "
                        "content = excluded.content, encoding = excluded.encoding, "
                        "updated_at = now()", (fp.name, content, enc))
        msg = (f"push: {len(months)}ヶ月 / 日次{len(daily)}行 / 注文{len(orders)}行 / "
               f"ファイル{len(files)}件 / kill={status.get('kill')}")
        cur.execute("insert into phase2_runs (kind, ok, message, meta) "
                    "values (%s, true, %s, %s)",
                    (kind, msg, json.dumps({"cum_net": status.get("cum_net"),
                                            "cur_dd": status.get("cur_dd"),
                                            "freshness": status.get("freshness")},
                                           ensure_ascii=False)))
        conn.commit()
    print(msg)
    return 0


def pull() -> int:
    DIR.mkdir(parents=True, exist_ok=True)
    with connect() as conn, conn.cursor() as cur:
        cur.execute("select path, content, encoding from phase2_files")
        rows = cur.fetchall()
    for path, content, enc in rows:
        (DIR / path).write_bytes(decode_file(content, enc))
    print(f"pull: {len(rows)} ファイルを {DIR} へ復元")
    return 0


def log_run(kind: str, message: str, ok: bool) -> int:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("insert into phase2_runs (kind, ok, message) "
                    "values (%s, %s, %s)", (kind, ok, message[:500]))
        conn.commit()
    print(f"logged ({'ok' if ok else 'failure'}): {message[:80]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 2 Supabase 同期")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--push", action="store_true")
    g.add_argument("--pull", action="store_true")
    g.add_argument("--log-failure", metavar="MSG")
    g.add_argument("--log-note", metavar="MSG")
    ap.add_argument("--kind", default="local")
    args = ap.parse_args()
    if args.pull:
        return pull()
    if args.log_failure:
        return log_run(args.kind, args.log_failure, ok=False)
    if args.log_note:
        return log_run(args.kind, args.log_note, ok=True)
    return push(args.kind)


if __name__ == "__main__":
    raise SystemExit(main())
