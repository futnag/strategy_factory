"""検証ファクトリ・デモ：ギャップ戦略を“パラメータ格子”で判定器にかける。

ユーザー例（前日終値比でのギャップ後に翌寄りで建玉）を、PIT安全な日次バックテストと
判定器で裁く。1つのアイデアでも threshold×hold×side で複数試行に化けることを示し、
**格子全体でデフレートすると“生の最良”が消える**様子（＝判定器自体のp-hack不能）を
実演する。

実行: $env:J_QUANTS_MIN_INTERVAL="0.7"; .venv\\Scripts\\python.exe examples\\research_gap_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.config import get_env  # noqa: E402
from invest_system.data.sources import jquants as jq  # noqa: E402
from invest_system.equities.universe import filter_common_stocks  # noqa: E402
from invest_system.research import AsOfView, GapReversal, judge_grid  # noqa: E402
from invest_system.validation.registry import default_registry  # noqa: E402

START = get_env("J_EQ_START_D", "2020-01-01") or "2020-01-01"
END = get_env("J_EQ_END_D", "2026-05-31") or "2026-05-31"
TOP_N = int(get_env("J_GAP_TOPN", "50") or "50")
LIQ_DATE = "20260225"   # 流動性ランキング用のキャッシュ済みスナップショット


def build_panel(frames: dict, field: str) -> pd.DataFrame:
    cols = {}
    for c, df in frames.items():
        if field in df.columns and "Date" in df.columns:
            s = df.dropna(subset=["Date"]).set_index("Date")[field]
            cols[c] = s[~s.index.duplicated(keep="last")]
    return pd.DataFrame(cols).sort_index()


def main() -> int:
    if not get_env("J_QUANTS_API_KEY"):
        print("ERROR: .env に J_QUANTS_API_KEY が必要です。")
        return 1
    listed = jq.fetch_listed_info()
    common = set(filter_common_stocks(listed)["Code"].astype(str))
    snap = jq.fetch_daily_quotes(LIQ_DATE)        # キャッシュ済み
    snap = snap[snap["Code"].astype(str).isin(common)]
    codes = (snap.sort_values("Va", ascending=False)["Code"].astype(str)
             .head(TOP_N).tolist())
    print(f"ユニバース: 流動性上位{len(codes)}銘柄  期間 {START}〜{END}")

    print("日次履歴を取得中（銘柄別・初回のみ実取得）…")
    frames = {}
    for i, c in enumerate(codes, 1):
        try:
            d = jq.fetch_daily_history(c, frm=START, to=END)
            if not d.empty:
                frames[c] = d
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] {c}: {str(e)[:60]}")
        if i % 25 == 0:
            print(f"  {i}/{len(codes)}")
    adjc = build_panel(frames, "AdjC")
    adjo = build_panel(frames, "AdjO").reindex(adjc.index)
    advp = build_panel(frames, "Va").reindex(adjc.index)   # 売買代金=容量用ADV
    print(f"パネル: {adjc.shape[0]} 営業日 × {adjc.shape[1]} 銘柄")

    view = AsOfView({"open": adjo, "close": adjc})
    grid = [GapReversal(threshold=th, hold=h, side=s)
            for th in (0.05, 0.08, 0.10) for h in (1, 5) for s in (1, -1)]
    print(f"戦略格子: {len(grid)} 通り（threshold×hold×side）\n")

    with default_registry() as reg:
        verdict = judge_grid(
            grid, view, scope="gap_reversal_demo",
            hypothesis="大幅な寄りギャップダウン後、過剰反応の修正で短期リバーサルが起きる",
            economic_rationale="流動性ショックや強制売りによる一時的ミスプライスが数日で解消される行動仮説",
            registry=reg, costs_bps=15.0, execution_lag=1, adv=advp, participation=0.1)
    print(verdict.report_md)
    print(f"\n（参考）格子{len(grid)}通り＝独立試行{verdict.k}としてデフレート。"
          "1つでも閾値0.95を超えれば PASS、超えねば『このアイデアにエッジ無し』。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
