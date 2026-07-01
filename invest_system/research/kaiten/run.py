"""回転手法バックテストの実行オーケストレーション。"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .config import CONFIG
from .data import load_symbols
from .engine import run_backtest
from .metrics import performance_report, plot_equity


def main(cfg: dict | None = None) -> tuple:
    """データ読込 → バックテスト → レポート → 成果物保存。"""
    cfg = deepcopy(CONFIG if cfg is None else cfg)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    data, universe_mask = load_symbols(cfg)
    if not data:
        print("[ERROR] 有効なデータがありません。CONFIG を確認してください。")
        return None, None

    print(f"[INFO] 対象銘柄数: {len(data)}")
    if cfg.get("instrument_type") == "futures":
        print(f"[INFO] 先物: {cfg.get('instrument_label', cfg.get('external_key'))} / "
              f"倍率={cfg.get('contract_multiplier')} / "
              f"証拠金率={cfg.get('margin_rate'):.0%}")
    print(f"[INFO] エントリー方式: {cfg['entry_method']} / "
          f"同日両ヒット: {cfg['same_day_priority']}")
    if universe_mask is not None:
        print(f"[INFO] ユニバース: Va 上位 {cfg['universe_top_n']} (PIT, "
              f"lookback={cfg['universe_lookback']}日)")

    eq, tr = run_backtest(data, cfg, universe_mask=universe_mask)
    performance_report(eq, tr, cfg)

    if not tr.empty:
        tr.to_csv(cfg["trades_csv_path"], index=False, encoding="utf-8-sig")
        print(f"[INFO] トレード履歴を保存: {cfg['trades_csv_path']}")
    plot_equity(eq, cfg)
    return eq, tr