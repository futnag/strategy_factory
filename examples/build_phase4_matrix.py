"""Phase 4 設計行列のフル材化＋キャッシュ（一度だけ・重い＝判定/評価で再利用）。

本番ユニバース（流動・PIT）×全凍結特徴量×マクロ×ラベルを 1 つの設計行列に組み、
data/phase4/ に保存する。判定・評価はこのキャッシュを読み直す（再材化しない）。
これはデータ材化であって判定ではない（K 不変）。

実行: .venv\\Scripts\\python.exe examples\\build_phase4_matrix.py
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


def main() -> int:
    t0 = time.time()
    print("=== Phase 4 設計行列フル材化（本番ユニバース・全特徴量・K不変）===", flush=True)
    uni = dm.production_universe()                       # 流動・PIT（¥100/¥10B/ADV¥50M）
    print(f"universe: {uni.shape}, avg {int(uni.sum(axis=1).mean())} 銘柄/月 "
          f"（{uni.index.min():%Y-%m}〜{uni.index.max():%Y-%m}）", flush=True)
    D = dm.build_design_matrix(uni)
    print(f"design matrix built in {time.time()-t0:.0f}s | X {D.X.shape} | "
          f"chars {len(D.char_cols)} | macro {D.macro_cols}", flush=True)
    print(f"  rows/month median {int(D.X.groupby(level=0).size().median())} | "
          f"feat NaN {float(D.X.isna().mean().mean()):.4f} | "
          f"y std {float(D.y.std()):.4f} | months {len(D.months)}", flush=True)
    dm.save_design_matrix(D, "data/phase4")
    print(f"→ saved data/phase4/ (X,y,macro,universe,meta) in {time.time()-t0:.0f}s total",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
