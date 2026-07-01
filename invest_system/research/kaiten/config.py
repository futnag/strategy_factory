"""回転手法バックテストの設定（CONFIG 辞書）。

生存者バイアス注意: 上場廃止銘柄をデータに含まないと成績が過大評価される。
J-Quants wide は退場銘柄の履歴を保持するが、取得開始前の廃止銘柄は欠落する。

⚠ 選択バイアス注意: 「探索最良」とあるパラメータ・セクター除外・ブラックリストは
in-sample 探索由来（TrialRegistry／DSR デフレートの外）。採用判定には judge 系での
再評価（試行数 K の計上＋真の OOS＝探索期間外）が必須。ブラックリストは 2016-2022 で
構築 → OOS は 2023 以降（examples/kaiten_filtered_compare.py）。
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_OUT = _ROOT / "output" / "kaiten"

CONFIG: dict = {
    # ---------- データ入力 ----------
    "data_source": "wide",              # "wide" | "external" | "synthetic"
    "external_key": "nk225_fut",        # data_source=external 時（investers/）
    "instrument_type": "equity",        # "equity" | "futures"
    "contract_multiplier": 1.0,         # 先物: 円/ポイント（日経マイクロ=10）
    "margin_rate": 0.11,                # 想定証拠金率（先物）
    "commission_per_contract": 0.0,     # 先物: 片道手数料（円/枚）
    "data_base": "data",
    "start_date": None,                 # 例 "2020-01-01"
    "end_date": None,
    "filter_common_stocks": True,       # ETF 等を除外（listed マスタ要）
    "use_synthetic_if_empty": True,

    # ---------- ユニバース（PIT・売買代金 Va） ----------
    "universe_top_n": 300,
    "universe_lookback": 60,            # 営業日
    "universe_min_obs": 30,             # lookback 内の最低観測日数

    # ---------- トレンドフィルター ----------
    "use_trend_filter": True,
    "trend_ma_period": 60,              # bb 探索最良
    # 複数 MA が N 営業日連続で上昇しているときだけエントリー（上昇局面の押し目限定）
    "use_ma_rise_filter": True,
    "ma_rise_periods": [5, 20, 60],    # 探索最良: 短期〜中期 MA の連続上昇
    "ma_rise_days": 7,                  # 各 MA が連続上昇している必要日数
    # パーフェクトオーダー: 短期 MA > 中期 MA > …（上昇トレンドの配列）
    "use_ma_order_filter": True,
    "ma_order_periods": [5, 20, 60],    # ma_rise_periods と揃えるのが一般的

    # ---------- エントリー (押し目買い指値) ----------
    "entry_method": "bb",               # "atr" | "bb"（探索最良: bb σ=2.5）
    "atr_period": 14,
    "atr_mult": 2.5,
    "bb_period": 20,
    "bb_sigma": 2.5,
    "limit_valid_days": 1,

    # ---------- イグジット ----------
    "take_profit_pct": 0.07,
    "stop_loss_pct": 0.05,
    "max_hold_days": 5,
    "same_day_priority": "stop",        # "stop" | "target" | "auto"

    # ---------- 資金管理 ----------
    "initial_capital": 1_000_000,
    "risk_per_trade_pct": 0.01,
    "leverage": 1.0,
    "max_positions": 10,
    "allow_multiple_per_symbol": False,

    # ---------- コスト ----------
    "commission_pct": 0.0005,
    "slippage_pct": 0.001,

    # ---------- 出力 ----------
    "output_dir": str(_OUT),
    "plot_path": str(_OUT / "equity_curve.png"),
    "trades_csv_path": str(_OUT / "trades.csv"),
    "annualization_days": 252,

    # ---------- レジーム・ブラックリスト（株式向け） ----------
    "use_regime_filter": False,
    "regime_vol_max": 1.0,              # vol_regime > 1 をブロック（高ボラ=2）
    "regime_require_trend_up": False,
    "regime_feature_path": None,        # None → data/features/regime.parquet
    "use_symbol_blacklist": False,
    "symbol_blacklist": [],
    "symbol_blacklist_path": None,
    "exclude_sectors": [],              # 例: ["医薬品", "食料品", "情報･通信業"]
    # レジーム適応: none|gate|sizing|exit|entry|sizing_exit|sizing_entry|exit_entry|full
    "regime_adapt_mode": None,
    "regime_vol_profiles": None,        # None → regime_adapt.DEFAULT_VOL_PROFILES
    "regime_trend_down_scale": True,    # trend_up=0 でサイズ半減
}

# 探索に基づくセクター除外（PIT マスタの S33Nm）
_DEFAULT_EXCLUDE_SECTORS = ["医薬品", "食料品", "情報･通信業"]


def filtered_equity_config(*, regime_adapt_mode: str | None = "gate") -> dict:
    """レジーム + ブラックリスト有効の株式 CONFIG。

    regime_adapt_mode:
      "gate" = 高ボラブロック（従来）
      その他 = regime_adapt.REGIME_ADAPT_MODES 参照
    """
    cfg = deepcopy(CONFIG)
    bl_path = _OUT / "blacklist.csv"
    use_gate = regime_adapt_mode in (None, "gate")
    cfg.update({
        "use_regime_filter": use_gate,
        "regime_vol_max": 1.0,
        "use_symbol_blacklist": True,
        "symbol_blacklist_path": str(bl_path) if bl_path.exists() else None,
        "exclude_sectors": list(_DEFAULT_EXCLUDE_SECTORS),
        "regime_adapt_mode": regime_adapt_mode if regime_adapt_mode != "gate" else "gate",
    })
    if not use_gate:
        cfg["use_regime_filter"] = False
    return cfg

# 先物キー別の契約仕様 + 探索最良パラメータ
# （2020-2024 探索＝in-sample。「全期間」評価は探索期間を含むため OOS でない — 参考値）
_FUTURES_SPECS: dict[str, dict] = {
    "nk225_fut": {
        "label": "日経225先物（マイクロ想定）",
        "contract_multiplier": 10.0,
        "margin_rate": 0.11,
        "commission_per_contract": 50.0,
        "default_entry": "bb",  # PF/DD 優先。リターン重視は atr
        "strategies": {
            "bb": {
                "entry_method": "bb",
                "bb_sigma": 3.0,
                "trend_ma_period": 40,
                "stop_loss_pct": 0.05,
                "take_profit_pct": 0.05,
                "max_hold_days": 5,
                "limit_valid_days": 10,
                "use_ma_rise_filter": True,
                "ma_rise_periods": [5, 20, 60],
                "ma_rise_days": 7,
                "use_ma_order_filter": True,
                "ma_order_periods": [5, 20, 60],
            },
            "atr": {
                "entry_method": "atr",
                "atr_mult": 2.0,
                "trend_ma_period": 40,
                "stop_loss_pct": 0.05,
                "take_profit_pct": 0.05,
                "max_hold_days": 3,
                "limit_valid_days": 7,
                "use_ma_rise_filter": True,
                "ma_rise_periods": [5, 20, 60],
                "ma_rise_days": 7,
                "use_ma_order_filter": True,
                "ma_order_periods": [5, 20, 60],
            },
        },
    },
    "topix_fut": {
        "label": "TOPIX先物（ミニ想定）",
        "contract_multiplier": 1000.0,
        "margin_rate": 0.11,
        "commission_per_contract": 50.0,
        "default_entry": "bb",
        "strategies": {
            "bb": {
                "entry_method": "bb",
                "bb_sigma": 2.5,
                "trend_ma_period": 40,
                "stop_loss_pct": 0.05,
                "take_profit_pct": 0.05,
                "max_hold_days": 5,
                "limit_valid_days": 7,
                "use_ma_rise_filter": True,
                "ma_rise_periods": [5, 20, 60],
                "ma_rise_days": 7,
                "use_ma_order_filter": True,
                "ma_order_periods": [5, 20, 60],
            },
        },
    },
}


def futures_config(key: str = "nk225_fut", entry: str | None = None) -> dict:
    """先物シミュレーション用 CONFIG。entry='bb'|'atr' でエントリー方式を切替。"""
    spec = _FUTURES_SPECS.get(key, {})
    strategies = spec.get("strategies", {})
    method = entry or spec.get("default_entry", "bb")
    if method not in strategies:
        raise ValueError(
            f"{key} は entry={method!r} 未対応。"
            f" 利用可: {list(strategies)}"
        )
    cfg = deepcopy(CONFIG)
    out = _OUT / "futures"
    suffix = f"{key}_{method}" if len(strategies) > 1 else key
    cfg.update({
        "data_source": "external",
        "external_key": key,
        "instrument_type": "futures",
        "filter_common_stocks": False,
        "use_synthetic_if_empty": False,
        "max_positions": 1,
        "allow_multiple_per_symbol": False,
        "contract_multiplier": spec.get("contract_multiplier", 10.0),
        "margin_rate": spec.get("margin_rate", 0.11),
        "commission_per_contract": spec.get("commission_per_contract", 50.0),
        "commission_pct": 0.0,
        "output_dir": str(out),
        "plot_path": str(out / f"equity_{suffix}.png"),
        "trades_csv_path": str(out / f"trades_{suffix}.csv"),
    })
    cfg.update(strategies[method])
    cfg.update({
        "use_regime_filter": True,
        "regime_vol_max": 1.0,
    })
    label = spec.get("label", key)
    cfg["instrument_label"] = f"{label} [{method}]"
    return cfg


def futures_entry_methods(key: str) -> list[str]:
    """先物キーで利用可能なエントリー方式一覧。"""
    return list(_FUTURES_SPECS.get(key, {}).get("strategies", {}))


# CONFIG で戦略・執行に影響する操作可能キー（メタデータ・出力パス除く）
TUNABLE_PARAMS: dict[str, str] = {
    # データ・対象
    "external_key": "先物キー（nk225_fut / topix_fut）",
    "start_date": "開始日",
    "end_date": "終了日",
    # トレンドフィルター
    "use_trend_filter": "終値 > 長期MA",
    "trend_ma_period": "トレンドMA期間",
    "use_ma_rise_filter": "MA連続上昇フィルター",
    "ma_rise_periods": "上昇判定MA期間リスト",
    "ma_rise_days": "連続上昇必要日数",
    "use_ma_order_filter": "パーフェクトオーダー",
    "ma_order_periods": "オーダー判定MA期間",
    # エントリー
    "entry_method": "atr | bb",
    "atr_period": "ATR期間",
    "atr_mult": "ATR指値倍率",
    "bb_period": "BB期間",
    "bb_sigma": "BBσ",
    "limit_valid_days": "指値有効日数",
    # イグジット
    "take_profit_pct": "利確%",
    "stop_loss_pct": "損切%",
    "max_hold_days": "最大保有日数",
    "same_day_priority": "同日両ヒット優先",
    # 資金・コスト（先物）
    "initial_capital": "初期資金",
    "risk_per_trade_pct": "1トレードリスク%",
    "leverage": "レバレッジ上限",
    "contract_multiplier": "先物倍率（円/ポイント）",
    "margin_rate": "証拠金率",
    "commission_per_contract": "片道手数料/枚",
    "slippage_pct": "スリッページ%",
}