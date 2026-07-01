"""C1（TOB リスクアーブ）前向き監視 — 凍結仕様で新規案件をスクリーニング。

正式 DSR 裁定（docs/09 §6）後の Phase 2 小サイズ前向き運用向け。**パラメータ探索はしない**
（PBO 多重性を増やさない）。凍結仕様は docs/47 §1.2。

凍結仕様（2026-06 確定）:
- サブ期間: 2020-2026 のみ（現行レジーム）
- 競合案件: 除外
- スプレッド下限: >=3%
- 現金 TOB・上場対象・約定可能（T+1 ストップ高ロック除外）
- 出口: 成立=買付価格テンダー / 不成立=ギャップ

usage:
  python examples/c1_forward_monitor.py
  python examples/c1_forward_monitor.py --since 2026-01-01
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import research_c1_tob_arb as base  # noqa: E402
from invest_system.equities.frictions import limit_lock_flags  # noqa: E402
from invest_system.equities import tob_arb as ta  # noqa: E402

# --- 凍結仕様（docs/47 §1.2・再裁定まで変更しない） -------------------------
FROZEN = {
    "subperiod": "2020-2026",
    "competing": False,
    "spread_min": 0.03,
    "scope": "tob_arb_forward",
    "strategy_id": "tob_arb(excl,spread>=3%)",
}
JQ_DAILY = "data/jquants/daily"


def _panels(codes: set, start: str) -> dict:
    fields = {"C": "close", "O": "open", "H": "high", "L": "low",
              "UL": "ul", "LL": "ll", "Vo": "vol", "Va": "val"}
    rows = []
    for p in sorted(glob.glob(f"{JQ_DAILY}/*.parquet")):
        if Path(p).stem < start:
            continue
        df = pd.read_parquet(p)
        if "Code" not in df.columns or df.empty:
            continue
        rows.append(df[df["Code"].isin(codes)][["Date", "Code"]
                                               + [c for c in fields if c in df.columns]])
    long = pd.concat(rows, ignore_index=True)
    long["Date"] = pd.to_datetime(long["Date"])
    return {name: long.pivot_table(index="Date", columns="Code", values=src,
                                   aggfunc="last").sort_index()
            for src, name in fields.items() if src in long.columns}


def screen(d: pd.DataFrame, panels: dict) -> pd.DataFrame:
    """凍結仕様で案件をフィルタし、約定可能性・スプレッドを付与。"""
    close, opn = panels["close"], panels["open"]
    no_buy, _ = limit_lock_flags(close, panels["high"], panels["low"],
                                 panels["ul"], panels["ll"], panels.get("vol"))
    rows = []
    for _, r in d.iterrows():
        if r["subperiod"] != FROZEN["subperiod"]:
            continue
        if bool(r["competing"]) != FROZEN["competing"]:
            continue
        sec = r["sec"]
        if sec is None or sec not in close.columns:
            continue
        cdates = close[sec].dropna().index
        after = cdates[cdates > r["announce_date"].normalize()]
        if len(after) == 0:
            continue
        entry = after[0]
        prev = cdates[cdates < entry]
        prev_close = float(close[sec].loc[prev[-1]]) if len(prev) else None
        eo = opn[sec].get(entry) if sec in opn.columns else None
        if eo is None or pd.isna(eo) or float(eo) <= 0:
            continue
        if entry in no_buy.index and sec in no_buy.columns and bool(no_buy.at[entry, sec]):
            continue
        m = ta.deal_metrics(r["tob_result"], float(r["initial_price"]),
                            float(r["final_price"]), prev_close,
                            float(eo), None)
        spread = m.get("arb_spread")
        if spread is None or spread < FROZEN["spread_min"]:
            continue
        rows.append({
            "announce_date": r["announce_date"].date(),
            "entry_date": entry.date(),
            "sec": sec,
            "target_name": r.get("target_name", ""),
            "initial_price": float(r["initial_price"]),
            "final_price": float(r["final_price"]),
            "entry_open": float(eo),
            "arb_spread_pct": round(float(spread) * 100, 2),
            "premium_pct": round(float(m["premium"]) * 100, 2) if m["premium"] else None,
            "result": r["tob_result"],
            "price_source": r.get("price_source", ""),
            "competing": bool(r["competing"]),
        })
    return pd.DataFrame(rows)


def main() -> int:
    since = "2016-01-01"
    if "--since" in sys.argv:
        i = sys.argv.index("--since")
        since = sys.argv[i + 1]

    d = base.join_edinet(base.load_spine())
    d = d[d["announce_date"] >= pd.Timestamp(since)]
    codes = set(d["sec"].dropna())
    start = (d["announce_date"].min() - pd.Timedelta(days=10)).strftime("%Y%m%d")
    panels = _panels(codes, start)
    hit = screen(d, panels)

    print("=== C1 前向き監視（凍結仕様）===")
    print(f"scope: {FROZEN['scope']} / strategy: {FROZEN['strategy_id']}")
    print(f"条件: {FROZEN['subperiod']}・競合{'含む' if FROZEN['competing'] else '除外'}"
          f"・spread>={int(FROZEN['spread_min']*100)}%・ロック除外")
    print(f"母集団: {since} 以降 {len(d)} 件 → 適合 {len(hit)} 件\n")

    if hit.empty:
        print("適合案件なし")
        return 0

    show = hit.sort_values("announce_date", ascending=False)
    cols = ["announce_date", "entry_date", "sec", "target_name",
            "arb_spread_pct", "result", "initial_price", "entry_open"]
    print(show[cols].head(20).to_string(index=False))
    print(f"\n直近12ヶ月: {len(hit[hit['announce_date'] >= (pd.Timestamp.today() - pd.DateOffset(months=12)).date()])} 件")
    print("\n運用チェック（docs/47 §1.4）:")
    print("  [ ] 公開買付代理人の口座を保有しているか")
    print("  [ ] 応募締切までに応募手続きが完了するか")
    print("  [ ] 案件あたり拘束資本が容量上限（p25 ¥2-3M）以内か")
    print("  [ ] テンダー成立時の応募・決済をペーパーで記録したか")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())