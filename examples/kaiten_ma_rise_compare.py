"""MA 連続上昇フィルター ON/OFF 比較（bb2.5_ma60・全期間）。"""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.research.kaiten import CONFIG
from invest_system.research.kaiten.data import load_symbols
from invest_system.research.kaiten.sweep import run_single


def main() -> None:
    base = deepcopy(CONFIG)
    base.update({"entry_method": "bb", "bb_sigma": 2.5, "trend_ma_period": 60})
    data, mask = load_symbols(base)

    for label, rise in [("MA rise OFF", False), ("MA rise ON", True)]:
        cfg = deepcopy(base)
        cfg["use_ma_rise_filter"] = rise
        _, _, sm = run_single(cfg, data, mask)
        print(f"\n=== {label} ===")
        print(f"  ma_rise_periods={cfg['ma_rise_periods']}  days={cfg['ma_rise_days']}")
        print(f"  trades={sm['n_trades']}  ret={sm['total_ret']:+.2%}  PF={sm['profit_factor']:.3f}")
        print(f"  maxDD={sm['max_dd']:+.2%}  sharpe={sm['sharpe']:.3f}")
        print(f"  timeout={sm['pct_timeout']:.1%}  target={sm['pct_target']:.1%}  stop={sm['pct_stop']:.1%}")


if __name__ == "__main__":
    main()