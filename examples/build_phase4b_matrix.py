"""Phase 4b 設計行列のフル材化（小型寄りユニバース・mcap≥¥30B）。

Phase 4（¥10B 床）とは別キャッシュ data/phase4b/ に保存。判定は run_phase4b_judgment.py が
一度だけ読み直す。K 不変（材化のみ）。

実行: .venv\\Scripts\\python.exe examples\\build_phase4b_matrix.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001  # pragma: no cover
    pass

from invest_system.equities import design_matrix as dm  # noqa: E402

CACHE = "data/phase4b"
PRESET = "smallcap_30b"


def main() -> int:
    t0 = time.time()
    print(f"=== Phase 4b 設計行列フル材化（preset={PRESET}・mcap≥¥30B・K不変）===", flush=True)
    uni = dm.production_universe(preset=PRESET)
    print(f"universe: {uni.shape}, avg {int(uni.sum(axis=1).mean())} 銘柄/月 "
          f"（{uni.index.min():%Y-%m}〜{uni.index.max():%Y-%m}）", flush=True)
    D = dm.build_design_matrix(uni)
    print(f"design matrix built in {time.time()-t0:.0f}s | X {D.X.shape} | "
          f"chars {len(D.char_cols)} | macro {D.macro_cols}", flush=True)
    print(f"  rows/month median {int(D.X.groupby(level=0).size().median())} | "
          f"feat NaN {float(D.X.isna().mean().mean()):.4f} | "
          f"y std {float(D.y.std()):.4f} | months {len(D.months)}", flush=True)
    dm.save_design_matrix(D, CACHE)
    print(f"→ saved {CACHE}/ in {time.time()-t0:.0f}s total", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())