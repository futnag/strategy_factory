"""ブラックリスト生成（探索期間のみ・先読み回避）。

2016-2022 のトレードから「n>=5 & 累積損失 & target率0%」銘柄を抽出。
MA上昇フィルターは OFF（探索期間のサンプル数確保）。本番は MA上昇 ON のまま適用。
OOS 検証（**2023以降**＝構築期間 2016-2022 と非重複）は kaiten_filtered_compare.py で実施。

実行: python examples/kaiten_build_blacklist.py
"""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from invest_system.research.kaiten import (  # noqa: E402
    CONFIG,
    build_blacklist_from_trades,
    load_symbols,
    run_backtest,
)

_TRAIN_START = "2016-01-01"
_TRAIN_END = "2022-12-31"
_MIN_TRADES = 5
_MAX_TARGET_RATE = 0.0


def main() -> None:
    cfg = deepcopy(CONFIG)
    cfg.update({
        "entry_method": "bb",
        "bb_sigma": 2.5,
        "trend_ma_period": 60,
        # ブラックリスト構築のみ MA 上昇 OFF（トレード数確保）
        "use_ma_rise_filter": False,
        "ma_rise_periods": [5, 20, 60],
        "ma_rise_days": 7,
        "use_ma_order_filter": False,
        "start_date": _TRAIN_START,
        "end_date": _TRAIN_END,
        # ブラックリスト構築時はフィルター OFF
        "use_regime_filter": False,
        "use_symbol_blacklist": False,
        "exclude_sectors": [],
    })

    print(f"[INFO] 探索期間 {_TRAIN_START} 〜 {_TRAIN_END} でバックテスト...", flush=True)
    data, mask = load_symbols(cfg)
    _, tr = run_backtest(data, cfg, universe_mask=mask)

    bl = build_blacklist_from_trades(
        tr,
        min_trades=_MIN_TRADES,
        max_target_rate=_MAX_TARGET_RATE,
        max_total_pnl=0.0,
    )
    out = Path(CONFIG["output_dir"]) / "blacklist.csv"
    bl.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[INFO] ブラックリスト: {len(bl)} 銘柄 → {out}")
    if not bl.empty:
        print(bl.head(20).to_string(index=False))


if __name__ == "__main__":
    main()