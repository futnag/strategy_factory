"""回転手法バックテスト実行例。

Silver wide（adj_* + turnover）から PIT ユニバース上位300を抽出し、
押し目指値ロング戦略をポートフォリオ共有資金でシミュレートする。

実行（リポジトリ root から）:
  $env:PYTHONUTF8="1"; python examples/kaiten_backtest.py

環境変数（任意）:
  KAITEN_SOURCE     wide | synthetic
  KAITEN_START      開始日 YYYY-MM-DD
  KAITEN_END        終了日
  KAITEN_ENTRY      atr | bb
  KAITEN_ATR_MULT   ATR 指値倍率
  KAITEN_TREND_MA   トレンド MA 期間
  KAITEN_TOP_N      ユニバース上位 N
  KAITEN_MA_RISE    1 で MA 連続上昇フィルター ON（既定 ON）
  KAITEN_MA_RISE_D  連続上昇必要日数（既定 7）
  KAITEN_MA_ORDER   1 でパーフェクトオーダー ON（既定 ON）
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

from invest_system.config import get_env  # noqa: E402
from invest_system.research.kaiten import CONFIG, main  # noqa: E402


def _cfg() -> dict:
    cfg = deepcopy(CONFIG)
    src = get_env("KAITEN_SOURCE", "wide") or "wide"
    cfg["data_source"] = src
    if start := get_env("KAITEN_START"):
        cfg["start_date"] = start
    if end := get_env("KAITEN_END"):
        cfg["end_date"] = end
    if entry := get_env("KAITEN_ENTRY"):
        cfg["entry_method"] = entry
    if mult := get_env("KAITEN_ATR_MULT"):
        cfg["atr_mult"] = float(mult)
    if ma := get_env("KAITEN_TREND_MA"):
        cfg["trend_ma_period"] = int(ma)
    if top := get_env("KAITEN_TOP_N"):
        cfg["universe_top_n"] = int(top)
    rise = get_env("KAITEN_MA_RISE", "1")
    cfg["use_ma_rise_filter"] = rise not in ("0", "false", "False", "off")
    if days := get_env("KAITEN_MA_RISE_D"):
        cfg["ma_rise_days"] = int(days)
    order = get_env("KAITEN_MA_ORDER", "1")
    cfg["use_ma_order_filter"] = order not in ("0", "false", "False", "off")
    return cfg


if __name__ == "__main__":
    main(_cfg())