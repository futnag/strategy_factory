"""ダッシュボード用 data.json の生成（Phase 2・無人運用の配管）。

data/phase2/（status.json / months.csv / equity_daily.csv / orders_*.csv）を
1つの JSON に束ねる。GitHub Actions が毎晩これを ops リポジトリへコミットし、
Vercel の Git 連携が自動デプロイする（ダッシュボードはこの JSON を同梱で読む
＝DB接続・環境変数・トークン不要の静的データ方式）。

使い方:
  python examples/phase2_dashboard_data.py --out ../strategy-factory-ops/data.json \
      [--run-note "nightly ok"] [--run-kind nightly]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

DIR = Path("data/phase2")


def _rows(df: pd.DataFrame) -> list[dict]:
    """NaN→None で JSON 安全な行リストへ。"""
    return json.loads(df.to_json(orient="records", force_ascii=False))


def build() -> dict:
    status = json.loads((DIR / "status.json").read_text(encoding="utf-8"))
    months = _rows(pd.read_csv(DIR / "months.csv"))
    daily = _rows(pd.read_csv(DIR / "equity_daily.csv"))
    orders: list[dict] = []
    for fp in sorted(DIR.glob("orders_eq_*.csv")):
        tag = fp.stem.replace("orders_eq_", "")
        for _, r in pd.read_csv(fp, dtype={"code": str}).iterrows():
            orders.append({"month": tag, "sleeve": "switch", "key": str(r["code"]),
                           "weight": float(r["weight"]), "price": float(r["price"]),
                           "qty": float(r["shares"]), "yen": float(r["yen"]),
                           "instruction": str(r.get("instruction", ""))})
    for fp in sorted(DIR.glob("orders_ts_*.csv")):
        tag = fp.stem.replace("orders_ts_", "")
        for _, r in pd.read_csv(fp).iterrows():
            lots = r.get("lots")
            orders.append({"month": tag, "sleeve": "tsmom", "key": str(r["asset"]),
                           "weight": float(r["weight"]), "price": None,
                           "qty": (float(lots) if pd.notna(lots) else None),
                           "yen": float(r["target_yen"]),
                           "instruction": str(r.get("instruction", ""))})
    latest = months[-1]["month"] if months else None
    return {"built_at": f"{pd.Timestamp.now():%Y-%m-%d %H:%M}",
            "status": status, "months": months, "daily": daily,
            "orders": [o for o in orders if o["month"] == latest],
            "latest": latest}


def main() -> int:
    ap = argparse.ArgumentParser(description="dashboard data.json 生成")
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-note", default="")
    ap.add_argument("--run-kind", default="local")
    args = ap.parse_args()
    data = build()
    data["run"] = {"kind": args.run_kind, "message": args.run_note}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"data.json: {out}（months={len(data['months'])} daily={len(data['daily'])} "
          f"orders={len(data['orders'])}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
