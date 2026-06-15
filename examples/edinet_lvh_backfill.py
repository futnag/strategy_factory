"""大量保有（府令060・一般新規 formCode 010000）の本体をバックフィルし C2 ユニバースを作る。

各報告本体（type=5 CSV）から保有目的・保有割合・提出者を抽出（large_holdings.enrich_holding）、
保有目的に「重要提案行為等」を含む報告を C2 母集合として標識する（docs/10 §7）。提出者は
activist registry で名寄せ（canonical_group / hard・soft タグ）。

スキャン対象＝formCode 010000（一般・新規候補）。新規/変更は **本体タイトル**で判定し
（formCode は不可・docs 訂正）、report_class=='new' かつ is_important_proposal が C2 の
新規参入エントリー母集団。特例報告(02/03)は定義上アクティビスト無し＝スキャンしない。

冪等・再開可：既に本体取得済み(body_ok)の doc_id はスキップ。途中保存（既定 100 件毎）。

usage:
  python examples/edinet_lvh_backfill.py
  python examples/edinet_lvh_backfill.py --form-codes 010000,010002   # 変更報告も含める
  python examples/edinet_lvh_backfill.py --limit 50                    # 動作確認
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.data.sources import edinet as ed          # noqa: E402
from invest_system.equities import large_holdings as lh       # noqa: E402

OUT = ed._CACHE / "large_holdings.parquet"

COLS = ["doc_id", "submit_dt", "issuer_edinet", "sec_code", "form_code",
        "special_report", "report_class", "doc_title", "change_reason",
        "purpose", "is_important_proposal", "holding_ratio", "holding_ratio_prev",
        "n_joint_holders", "shares_held", "shares_outstanding",
        "filer_edinet", "filer_name_jp", "filer_name_en", "oblig_date",
        "canonical_group", "style", "in_registry", "body_ok"]


def load_listing() -> pd.DataFrame:
    frames = []
    for p in sorted(glob.glob(str(ed._CACHE / "list" / "*.parquet"))):
        df = pd.read_parquet(p)
        if "_empty" not in df.columns:
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--form-codes", default="010000",
                    help="スキャンする formCode（カンマ区切り）。既定=一般新規のみ")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=100)
    args = ap.parse_args()
    form_codes = [c.strip() for c in args.form_codes.split(",") if c.strip()]

    listing = load_listing()
    d060 = listing[(listing["ordinanceCode"] == "060")
                   & (listing["formCode"].isin(form_codes))].copy()
    d060["submitDateTime"] = pd.to_datetime(d060["submitDateTime"], errors="coerce")
    d060 = d060.sort_values("submitDateTime")
    if args.limit:
        d060 = d060.head(args.limit)
    print(f"大量保有（formCode {form_codes}）{len(d060)} 件")

    rows: dict[str, dict] = {}
    done: set[str] = set()
    if OUT.exists():
        prev = pd.read_parquet(OUT)
        for _, r in prev.iterrows():
            rows[r["doc_id"]] = {c: r[c] for c in COLS if c in prev.columns}
            if r.get("body_ok"):
                done.add(r["doc_id"])
        print(f"既存 {len(rows)} 件（本体取得済み {len(done)}）→ 残りを取得")

    todo = [d for d in d060["docID"] if d not in done]
    print(f"取得対象 {len(todo)} 件")

    def flush():
        df = pd.DataFrame(list(rows.values()))
        for c in COLS:
            if c not in df.columns:
                df[c] = pd.NA
        df = df[COLS]
        OUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT)

    by_id = {r["docID"]: r for _, r in d060.iterrows()}
    for i, did in enumerate(todo, 1):
        rows[did] = lh.enrich_holding(did, by_id[did].to_dict())
        if i % args.save_every == 0:
            flush()
            print(f"  {i}/{len(todo)} … 途中保存")
    flush()

    final = pd.read_parquet(OUT)
    ok = int(final["body_ok"].fillna(False).sum())
    new_act = final[(final["report_class"] == "new")
                    & (final["is_important_proposal"] == True)]      # noqa: E712
    print(f"\n完了：{len(final)} 件 / 本体OK {ok}")
    print(f"新規×重要提案（C2 エントリー母集団）：{len(new_act)} 件 / "
          f"名簿一致 {int(new_act['in_registry'].fillna(False).sum())}")
    print(f"ユニーク提出者(filer_edinet)：{new_act['filer_edinet'].nunique()}")
    print(f"保存先: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
