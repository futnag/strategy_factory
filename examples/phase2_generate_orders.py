"""Phase 2 シグナル生成（バックログ1）：月末に実行し、翌営業日寄付の注文リストを出す。

確定ポートフォリオ（docs/03 §6.16）＝ value↔PEAD switch（株式スリーブ）＋ TSMOM
オーバーレイ。戦略パラメータは §6.9-6.15 の事前登録のまま**凍結**（DP10）。本スクリプトは
ウェイト計算と発注単位への写像のみを行う（invest_system/production/orders.py・D5）。

実装仕様（docs/02 D5）：
- 株式ロング脚 … 単元未満株（かぶミニ等・1株単位）。**寄付取引＝翌営業日始値で約定**
  ＝検証の T+1 始値規約（DP17・§6.13）と一致する。
- 株式ショート脚 … 合計想定元本を日経225マイクロ先物の売りに集約（小資金の執行近似）。
- TSMOM … 日経225マイクロのみ枚数化、他資産はペーパー段階では想定元本のまま保持
  （実弾移行時にブローカーのロット表で較正＝D5チェックリスト）。

出力：
  data/phase2/orders_eq_<月>.csv   … 株式ロング注文（コード/株数/参考価格）＋ヘッジ枚数
  data/phase2/orders_ts_<月>.csv   … TSMOM 注文（資産/ロットまたは想定元本）
  data/phase2/intended_<月>.parquet … 意図ウェイト（照合レポートの基準・バックログ2が使用）

実行（月末の夜または翌朝）: .venv\\Scripts\\python.exe examples\\phase2_generate_orders.py
オプション: --capital-eq 600000 --capital-ts 300000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from invest_system.config import get_env  # noqa: E402
from invest_system.data.external import load_external_prices  # noqa: E402
from invest_system.data.sources import jquants as jq  # noqa: E402
from invest_system.equities import events  # noqa: E402
from invest_system.equities.universe import (  # noqa: E402
    apply_universe_mask, filter_common_stocks, point_in_time_universe,
    universe_members,
)
from invest_system.equities.panel import (  # noqa: E402
    assemble_panel, fetch_month_end_snapshots, load_daily_panel,
)
from invest_system.equities.fundamentals import load_fundamentals, point_in_time  # noqa: E402
from invest_system.equities.factors import (  # noqa: E402
    cross_sectional_zscore, sector_neutralize, value_quality_size_factors,
)
from invest_system.production import equity_orders, hedge_contracts, lot_orders  # noqa: E402
from invest_system.research import (  # noqa: E402
    AsOfView, CrossSectionalStrategy, RegimeSwitch,
)
from invest_system.research.strategies_tsmom import (  # noqa: E402
    annualized_vol, blend_weights, tsmom_weights,
)
from invest_system.timeseries import vol_regime  # noqa: E402

START = "2016-07"
TSMOM_KEYS = ["nk225_fut", "sp500", "nasdaq_comp", "gold", "silver", "platinum",
              "wti", "copper", "usdjpy", "eurjpy", "audjpy"]
OUT_DIR = Path("data/phase2")


def _switch_weights_latest() -> tuple[pd.Timestamp, pd.Series, pd.Series, float]:
    """旗艦 switch の最新月末ウェイトを返す（§6.10 と同一の組立・PIT）。"""
    end = pd.Timestamp.today().strftime("%Y-%m")
    listed = jq.fetch_listed_info()
    snaps = fetch_month_end_snapshots(START, end)
    adj, raw, turn = (assemble_panel(snaps, c) for c in ("AdjC", "C", "Va"))
    common = set(filter_common_stocks(listed)["Code"].astype(str))
    turn_c = turn[[c for c in turn.columns if str(c) in common]]
    umask = point_in_time_universe(turn_c, top_n=300, lookback=12, min_obs=6)
    superset = universe_members(umask)
    adj, raw = adj.reindex(columns=superset), raw.reindex(columns=superset)
    umask = umask.reindex(columns=superset).fillna(False)
    sector = listed.assign(Code=listed["Code"].astype(str)).set_index("Code")["S33"]
    rebal = adj.index
    view = AsOfView({"close": adj})
    fund = load_fundamentals(superset)

    def zN(f):
        return cross_sectional_zscore(sector_neutralize(apply_universe_mask(f, umask),
                                                        sector))
    pit = point_in_time(fund, rebal, ["ShOutFY", "TrShFY", "Eq"], lag_days=1)
    value = zN(value_quality_size_factors(pit, raw, adj)["book_to_market"])
    pead = zN(point_in_time(events.forecast_revision(fund), rebal, ["fcst_revision"],
                            date_col="DiscDate", lag_days=1)["fcst_revision"]
              .reindex(columns=superset))
    daily = load_daily_panel(field="AdjC")
    vol_m = vol_regime(daily).reindex(rebal, method="ffill")
    value_ls = CrossSectionalStrategy(value, 0.2, name="value")
    pead_lt = CrossSectionalStrategy(pead, 0.2, name="pead_longtilt", long_only=True)
    switch = RegimeSwitch(vol_m, {0: value_ls, 1: value_ls, 2: pead_lt},
                          name="switch")
    t = rebal[-1]
    w = switch.target_weights(view.asof(t))
    regime = float(vol_m.get(t, np.nan))
    return t, w, raw.loc[t], regime


def _tsmom_weights_latest() -> tuple[pd.Timestamp, pd.Series, pd.Series]:
    """TSMOM ブレンドの最新**月末**ウェイト（§6.15 と同一構成）。

    月中に実行した場合、進行中の月の最新日は意思決定日ではない（月次の規律＝DP10）。
    「行の翌営業日が翌月に属する」行のみを月末とみなし、その最後を採用する
    （月末当日の夜に実行すればその日、月中なら前月末になる）。
    """
    cl = load_external_prices(TSMOM_KEYS, field="close")
    cl_ff = cl.ffill(limit=7)
    m_close = cl_ff.groupby(cl_ff.index.to_period("M")).tail(1)
    is_eom = [(d + pd.offsets.BDay(1)).month != d.month for d in m_close.index]
    rebal = m_close.index
    vol_m = annualized_vol(cl, window=63, floor=0.05).ffill(limit=7).reindex(rebal)
    sets = [tsmom_weights(m_close, vol_m, lb, vol_target=0.10) for lb in (3, 6, 12)]
    blend = blend_weights(sets)
    t = rebal[np.asarray(is_eom)][-1]
    w = blend.get(t, pd.Series(dtype="float64"))
    return t, w, m_close.loc[t]


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 2 注文生成（ペーパー/実弾共通）")
    ap.add_argument("--capital-eq", type=float, default=600_000)
    ap.add_argument("--capital-ts", type=float, default=300_000)
    args = ap.parse_args()
    if not get_env("J_QUANTS_API_KEY"):
        print("ERROR: .env に J_QUANTS_API_KEY が必要です。")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 株式スリーブ（switch）---
    t_eq, w_eq, px_eq, regime = _switch_weights_latest()
    stale = (pd.Timestamp.today().normalize() - t_eq).days
    print(f"=== Phase 2 注文生成 ===")
    print(f"[株式 switch] 決定日 {t_eq:%Y-%m-%d}（{stale}日前）"
          f" レジーム vol={regime:.0f}（0/1=value, 2=PEAD）銘柄数 {len(w_eq)}")
    if stale > 7:
        print(f"  ⚠ 決定日が {stale} 日前です。examples/update_data.py で"
              "データを最新化してから再実行してください。")
    orders_eq, short_notional = equity_orders(w_eq, px_eq, args.capital_eq)
    fut_px = load_external_prices(["nk225_fut"], field="close")["nk225_fut"] \
        .dropna().iloc[-1]
    n_hedge, hedge_yen = hedge_contracts(short_notional, float(fut_px))
    long_yen = float(orders_eq["yen"].sum())
    print(f"  ロング {len(orders_eq)}銘柄 ¥{long_yen:,.0f} / ショート想定元本 "
          f"¥{short_notional:,.0f} → 225マイクロ売り {n_hedge}枚（¥{hedge_yen:,.0f}・"
          f"先物 {fut_px:,.0f}）")

    # --- TSMOM スリーブ ---
    t_ts, w_ts, px_ts = _tsmom_weights_latest()
    print(f"[TSMOM blend] 決定日 {t_ts:%Y-%m-%d} 建玉 {len(w_ts)}資産 "
          f"グロス {w_ts.abs().sum():.2f}")
    orders_ts = lot_orders(w_ts, px_ts, args.capital_ts,
                           lot_units={"nk225_fut": 10.0},
                           lot_steps={"nk225_fut": 1})

    # --- 出力 ---
    tag = f"{t_eq:%Y-%m}"
    f_eq = OUT_DIR / f"orders_eq_{tag}.csv"
    f_ts = OUT_DIR / f"orders_ts_{tag}.csv"
    f_int = OUT_DIR / f"intended_{tag}.parquet"
    eq_out = orders_eq.copy()
    eq_out["instruction"] = "かぶミニ等で翌営業日寄付・買い"
    hedge_row = pd.DataFrame([{"code": "N225M", "weight": -short_notional /
                               args.capital_eq, "price": float(fut_px),
                               "shares": -n_hedge, "yen": -hedge_yen,
                               "instruction": "日経225マイクロ売建（ショート脚ヘッジ）"}])
    pd.concat([eq_out, hedge_row], ignore_index=True).to_csv(
        f_eq, index=False, encoding="utf-8-sig")
    ts_out = orders_ts.copy()
    ts_out["instruction"] = np.where(ts_out["asset"] == "nk225_fut",
                                     "225マイクロ（枚数）", "ペーパー想定元本で記帳")
    ts_out.to_csv(f_ts, index=False, encoding="utf-8-sig")
    intended = pd.concat([
        pd.DataFrame({"sleeve": "switch", "key": w_eq.index.astype(str),
                      "weight": w_eq.values, "asof": t_eq}),
        pd.DataFrame({"sleeve": "tsmom", "key": w_ts.index.astype(str),
                      "weight": w_ts.values, "asof": t_ts}),
    ], ignore_index=True)
    intended.to_parquet(f_int)
    manifest = {"month": tag, "capital_eq": args.capital_eq,
                "capital_ts": args.capital_ts,
                "decision_eq": f"{t_eq:%Y-%m-%d}", "decision_ts": f"{t_ts:%Y-%m-%d}",
                "regime_vol": regime, "hedge_contracts": n_hedge,
                "hedge_yen": hedge_yen, "short_notional": short_notional,
                "long_yen": long_yen}
    f_man = OUT_DIR / f"manifest_{tag}.json"
    f_man.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                     encoding="utf-8")
    print(f"\n出力: {f_eq}\n      {f_ts}\n      {f_int}\n      {f_man}")
    print("\n※ 執行規約（D5/DP17）：このリストは決定日の翌営業日**寄付**で約定させる"
          "（かぶミニ寄付取引・先物は寄成）。ペーパーでは翌営業日始値を約定価格として"
          "台帳に記録する（バックログ2の照合スクリプトが intended と突き合わせる）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
