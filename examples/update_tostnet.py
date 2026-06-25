r"""JPX ToSTNeT 超大口約定情報を前向きに蓄積（無料スクレイプ・監視用）。

毎日1回実行する想定。現在ページに載っている未取得の取引日だけを ``data/jpx_tostnet/`` に
追記する（冪等・再開可能）。**過去2週間より前は取得不可**なので、運用が2週間以上止まると
その間は恒久的に欠測になる点に注意（前向き蓄積のみ）。詳細は ``tostnet_monitoring_plan.md``。

    # Windows (PowerShell)
    $env:PYTHONUTF8="1"; .\.venv\Scripts\python.exe examples\update_tostnet.py
    # Linux / macOS
    .venv/bin/python examples/update_tostnet.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from invest_system.data.sources.jpx_tostnet import (  # noqa: E402
    daily_summary, load_tostnet, update_tostnet)

CACHE = "data/jpx_tostnet"


if __name__ == "__main__":
    print("=== JPX ToSTNeT 超大口 前向き蓄積 ===")
    rep = update_tostnet(CACHE)
    print("report:", rep)

    df = load_tostnet(CACHE)
    n_days = df["Date"].nunique() if len(df) else 0
    print(f"\n蓄積累計: {len(df)} 件 / {n_days} 取引日")
    summary = daily_summary(df)
    if len(summary):
        print("\n日次サマリ（直近14営業日）:")
        print(summary.tail(14).to_string())
