"""Phase 2 月次照合（バックログ2）：台帳の再構成と live-vs-backtest レポート。

設計（docs/02 D5）：照合は**ステートレス**＝毎回 data/phase2/ の manifest/orders から
約定をシミュレートして equity curve を再構成する（壊れる台帳状態を持たない）。
- 約定 … 決定日の翌取引日の**始値**（T+1・DP17）。祝日/欠損は5日以内で繰延。
  `fills_actual_<月>.csv`（列: sleeve,key,fill_price[,fill_date]）があれば実約定で
  上書きし、ペーパー約定との差を**実測スリッページ**として報告（Phase 2b 用）。
- 会計 … 円建て×調整後リターン（投下円 × adjO(出口)/adjO(入口) − 1）＝株式分割に頑健。
  株式ショート脚は 225マイクロ売りの円損益。TSMOM はペーパー段階では**想定元本
  （target_yen）で記帳**し、ロット化で建たない玉（shortfall）は別欄で可視化する。
- 判定 … 合成 DD をキルスイッチ水準（警報−8%/デリスク−12%/停止−15%・D5）と照合。
  確定月＝次月リバランスの約定で手仕舞い評価、最新月＝直近終値でのマークトゥマーケット
  （「進行中」フラグ）。

実行（毎月のリバランス約定後・週次/日次監視でも可）:
  .venv\\Scripts\\python.exe examples\\phase2_reconcile.py
出力: コンソール＋ data/phase2/ の
  report_latest.md   … 人間向けレポート（従来どおり）
  status.json        … 機械可読サマリ（DD・キルスイッチ・データ鮮度＝ダッシュボード用）
  months.csv         … 月次テーブル
  equity_daily.csv   … 日次 equity curve（終値マーク補間・確定値は月次会計が正）
"""
from __future__ import annotations

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

from invest_system.data.external import load_external_prices  # noqa: E402
from invest_system.data.external_fetch import PHASE2_KEYS  # noqa: E402
from invest_system.equities.panel import load_daily_panel  # noqa: E402
from invest_system.production import (  # noqa: E402
    ALERT_DD, DERISK_DD, STOP_DD, apply_actual_fills, daily_pnl_curve,
    drawdown_status, next_open_fills, yen_positions_pnl,
)

DIR = Path("data/phase2")
COST_EQ_BPS, COST_TS_BPS = 15.0, 5.0   # モデルコスト（参考列・§6.10/§6.15 と同一）


def _months() -> list[str]:
    return sorted(p.stem.replace("manifest_", "")
                  for p in DIR.glob("manifest_*.json"))


def _load(tag: str):
    man = json.loads((DIR / f"manifest_{tag}.json").read_text(encoding="utf-8"))
    oe = pd.read_csv(DIR / f"orders_eq_{tag}.csv", dtype={"code": str})
    ots = pd.read_csv(DIR / f"orders_ts_{tag}.csv")
    stocks = oe[oe["code"] != "N225M"].copy()
    return man, stocks, ots


def _actual(tag: str, sleeve: str) -> pd.DataFrame | None:
    fp = DIR / f"fills_actual_{tag}.csv"
    if not fp.exists():
        return None
    df = pd.read_csv(fp, dtype={"key": str})
    return df[df["sleeve"] == sleeve] if "sleeve" in df.columns else df


def _adj_at(panel: pd.DataFrame, fills: pd.DataFrame) -> pd.Series:
    """各 (key, fill_date) の調整後価格を引く（未約定は NaN）。"""
    out = {}
    for _, r in fills.iterrows():
        k, d = r["key"], r["fill_date"]
        out[k] = (float(panel.at[d, k])
                  if pd.notna(d) and k in panel.columns and d in panel.index
                  else np.nan)
    return pd.Series(out, dtype="float64")


def main() -> int:
    months = _months()
    if not months:
        print("ERROR: data/phase2 に manifest がありません。先に "
              "phase2_generate_orders.py を実行してください。")
        return 1
    op_raw = load_daily_panel(field="O")
    op_adj = load_daily_panel(field="AdjO")
    cl_adj = load_daily_panel(field="AdjC")
    ext_op = load_external_prices(field="open")
    ext_cl = load_external_prices(field="close")

    rows, slips, pos_months = [], [], []
    prev_eq_yen: pd.Series | None = None
    prev_ts_yen: pd.Series | None = None
    for i, tag in enumerate(months):
        man, stocks, ots = _load(tag)
        cap_eq, cap_ts = float(man["capital_eq"]), float(man["capital_ts"])
        d_eq, d_ts = pd.Timestamp(man["decision_eq"]), pd.Timestamp(man["decision_ts"])

        # --- 入口約定（T+1始値・実約定があれば上書き）---
        f_eq = next_open_fills(list(stocks["code"]), d_eq, op_raw)
        f_eq, sl1 = apply_actual_fills(f_eq, _actual(tag, "switch"))
        f_fut = next_open_fills(["nk225_fut"], d_eq, ext_op)
        f_ts = next_open_fills([a for a in ots["asset"]], d_ts, ext_op)
        f_ts, sl2 = apply_actual_fills(f_ts, _actual(tag, "tsmom"))
        slips += [sl1, sl2]
        f_eq.to_parquet(DIR / f"fills_{tag}.parquet")

        shares = stocks.set_index("code")["shares"].astype(float)
        fill_px = f_eq.set_index("key")["fill_price"].reindex(shares.index)
        invested = (shares * fill_px).dropna()            # 実投下円（ロング）
        unfilled = int(fill_px.isna().sum())
        adj0 = _adj_at(op_adj, f_eq).reindex(invested.index)
        fut0 = float(f_fut["fill_price"].iloc[0])
        n_hedge = int(man["hedge_contracts"])
        # n_hedge=0 のとき fut0 が NaN でも 0×NaN=NaN で月全体を汚染しないよう明示 0
        hedge_notional = (-n_hedge * fut0 * 10.0) if n_hedge else 0.0  # 売り＝負
        ts_notional = ots.set_index("asset")["target_yen"]  # ペーパー＝想定元本
        ts0 = f_ts.set_index("key")["fill_price"].reindex(ts_notional.index)
        live_gap = float(ots["shortfall_yen"].abs().sum())

        # --- 出口評価：次月の約定 or 最新値（進行中）---
        if i + 1 < len(months):
            man2, _, _ = _load(months[i + 1])
            x_eq = next_open_fills(list(invested.index),
                                   pd.Timestamp(man2["decision_eq"]), op_adj)
            adj1 = x_eq.set_index("key")["fill_price"].reindex(invested.index)
            fut1 = float(next_open_fills(["nk225_fut"],
                                         pd.Timestamp(man2["decision_eq"]),
                                         ext_op)["fill_price"].iloc[0])
            ts1 = next_open_fills(list(ts_notional.index),
                                  pd.Timestamp(man2["decision_ts"]),
                                  ext_op).set_index("key")["fill_price"] \
                .reindex(ts_notional.index)
            status, val_date = "確定", pd.Timestamp(man2["decision_eq"])
        else:
            # 進行中：全資産を**同一日**（JP 最新営業日）で as-of 評価する。
            # 外部系列だけ先の日付で評価するとヘッジ・TSMOM に見かけの損益が出る。
            val_date = cl_adj.index[-1]
            adj1 = cl_adj.iloc[-1].reindex(invested.index)
            ext_asof = ext_cl.asof(val_date)
            fut1 = float(ext_asof["nk225_fut"])
            ts1 = ext_asof.reindex(ts_notional.index)
            status = "進行中"

        rel_eq = adj1 / adj0.replace(0, np.nan) - 1.0
        pnl_stocks = yen_positions_pnl(invested, rel_eq)
        # fut0/fut1 欠損（データ鮮度等）の NaN は意図的に伝播させる＝下流の
        # DATA-ERROR ガードで月ごと検出する（黙って 0 にしない）
        pnl_hedge = hedge_notional * (fut1 / fut0 - 1.0) if hedge_notional else 0.0
        rel_ts = ts1 / ts0.replace(0, np.nan) - 1.0
        pnl_ts = yen_positions_pnl(ts_notional, rel_ts)
        ret_eq = (pnl_stocks + pnl_hedge) / cap_eq
        ret_ts = pnl_ts / cap_ts
        combo = (pnl_stocks + pnl_hedge + pnl_ts) / (cap_eq + cap_ts)

        # --- モデルコスト（参考）：前月比の円ベース回転 × 片道bps ---
        cur_eq = pd.concat([invested, pd.Series({"N225M": hedge_notional})])
        turn_eq = (cur_eq.abs().sum() if prev_eq_yen is None else
                   (cur_eq.reindex(cur_eq.index.union(prev_eq_yen.index))
                    .fillna(0.0) - prev_eq_yen.reindex(
                        cur_eq.index.union(prev_eq_yen.index)).fillna(0.0))
                   .abs().sum())
        cur_ts = ts_notional.fillna(0.0)
        turn_ts = (cur_ts.abs().sum() if prev_ts_yen is None else
                   (cur_ts.reindex(cur_ts.index.union(prev_ts_yen.index))
                    .fillna(0.0) - prev_ts_yen.reindex(
                        cur_ts.index.union(prev_ts_yen.index)).fillna(0.0))
                   .abs().sum())
        cost = (turn_eq * COST_EQ_BPS + turn_ts * COST_TS_BPS) / 1e4
        combo_net = combo - cost / (cap_eq + cap_ts)
        prev_eq_yen, prev_ts_yen = cur_eq, cur_ts

        rows.append({
            "month": tag, "status": status, "val_date": f"{val_date:%Y-%m-%d}",
            "ret_eq": ret_eq, "ret_ts": ret_ts, "combo_gross": combo,
            "combo_net": combo_net, "long_fill_yen": float(invested.sum()),
            "unfilled_names": unfilled, "hedge_yen": hedge_notional,
            "ts_live_gap_yen": live_gap,
        })
        pos_months.append({
            "tag": tag, "d_eq": d_eq, "cap_eq": cap_eq, "cap_ts": cap_ts,
            "invested": invested, "adj0": adj0,
            "fill_dates_eq": f_eq.set_index("key")["fill_date"],
            "hedge_notional": hedge_notional, "fut0": fut0,
            "fut_date": f_fut["fill_date"].iloc[0],
            "ts_notional": ts_notional, "ts0": ts0,
            "fill_dates_ts": f_ts.set_index("key")["fill_date"],
            "cost_frac": cost / (cap_eq + cap_ts),
            "ret_eq": ret_eq, "ret_ts": ret_ts, "combo_net": combo_net,
        })

    df = pd.DataFrame(rows).set_index("month")
    # データ障害ガード：月次リターンの NaN（約定/評価価格の欠損）は黙って落とさない。
    # drawdown_status が DATA-ERROR を返し、本スクリプトは exit 2（Actions 失敗→Issue）。
    bad_months = list(df.index[df[["ret_eq", "ret_ts", "combo_net"]]
                               .isna().any(axis=1)])
    net = pd.Series(df["combo_net"].values,
                    index=pd.PeriodIndex(df.index, freq="M").to_timestamp("M"))
    dd, cur_dd, kill = drawdown_status(net)
    cum = float((1.0 + net).prod() - 1.0)
    slip_all = pd.concat([s for s in slips if len(s)]) if any(len(s) for s in slips) \
        else pd.Series(dtype="float64")

    lines = ["# Phase 2 照合レポート",
             f"- 生成: {pd.Timestamp.now():%Y-%m-%d %H:%M} / 対象 {len(df)}ヶ月 / "
             f"累積ネット **{cum:+.2%}** / 現在DD **{cur_dd:+.2%}** → **{kill}**",
             f"- 計画帯（§6.13）: 年率 OOS +0.45〜+0.82（判定には12ヶ月以上の蓄積が必要）",
             f"- 実測スリッページ: " +
             (f"平均 {slip_all.mean() * 1e4:+.0f}bp（n={len(slip_all)}）"
              if len(slip_all) else "なし（ペーパー＝T+1始値どおり）"),
             "",
             "| 月 | 状態 | 評価日 | 株式 | TSMOM | 合成(粗) | 合成(净) | ロング充足 | 未約定 | 実弾ギャップ |",
             "|---|---|---|--:|--:|--:|--:|--:|--:|--:|"]
    for tag, r in df.iterrows():
        lines.append(
            f"| {tag} | {r['status']} | {r['val_date']} | {r['ret_eq']:+.2%} | "
            f"{r['ret_ts']:+.2%} | {r['combo_gross']:+.2%} | {r['combo_net']:+.2%} | "
            f"¥{r['long_fill_yen']:,.0f} | {r['unfilled_names']} | "
            f"¥{r['ts_live_gap_yen']:,.0f} |")
    if bad_months:
        lines.append("")
        lines.append(f"⚠ **DATA-ERROR**: リターン欠損月 = {', '.join(bad_months)}"
                     "（約定/評価価格の欠損。外部価格の鮮度・nk225_fut を確認）")
    report = "\n".join(lines)
    (DIR / "report_latest.md").write_text(report, encoding="utf-8")
    print(report)

    # --- 機械可読出力（ダッシュボード/監視用）---
    # 日次 equity curve：月内は終値マークで補間し、月境界は月次会計値で連鎖する。
    daily = []
    chain = chain_eq = chain_ts = 1.0
    for j, P in enumerate(pos_months):
        end = (pos_months[j + 1]["d_eq"] if j + 1 < len(pos_months)
               else cl_adj.index[-1])
        win = cl_adj.index[(cl_adj.index > P["d_eq"]) & (cl_adj.index <= end)]
        if len(win):
            pnl_eq = daily_pnl_curve(P["invested"], P["adj0"], cl_adj, win,
                                     P["fill_dates_eq"])
            pnl_eq = pnl_eq + daily_pnl_curve(
                pd.Series({"nk225_fut": P["hedge_notional"]}),
                pd.Series({"nk225_fut": P["fut0"]}), ext_cl, win,
                pd.Series({"nk225_fut": P["fut_date"]}))
            pnl_ts = daily_pnl_curve(P["ts_notional"], P["ts0"], ext_cl, win,
                                     P["fill_dates_ts"])
            cap = P["cap_eq"] + P["cap_ts"]
            for d in win:
                daily.append({
                    "date": d,
                    "eq": chain_eq * (1 + pnl_eq[d] / P["cap_eq"]),
                    "ts": chain_ts * (1 + pnl_ts[d] / P["cap_ts"]),
                    "combo_net": chain * (1 + (pnl_eq[d] + pnl_ts[d]) / cap
                                          - P["cost_frac"]),
                })
        chain *= 1 + P["combo_net"]
        chain_eq *= 1 + P["ret_eq"]
        chain_ts *= 1 + P["ret_ts"]
    anchor = pd.DataFrame([{"date": pos_months[0]["d_eq"], "eq": 1.0, "ts": 1.0,
                            "combo_net": 1.0}])
    daily_df = pd.concat([anchor, pd.DataFrame(daily)],
                         ignore_index=True).set_index("date")
    daily_df.to_csv(DIR / "equity_daily.csv")

    df.reset_index().to_csv(DIR / "months.csv", index=False)
    fresh_ext = {k: (f"{ext_cl[k].dropna().index.max():%Y-%m-%d}"
                     if k in ext_cl.columns and ext_cl[k].notna().any() else None)
                 for k in PHASE2_KEYS}
    status = {
        "generated_at": f"{pd.Timestamp.now():%Y-%m-%d %H:%M}",
        "asof": str(df["val_date"].iloc[-1]),
        "latest_month": months[-1], "latest_status": str(df["status"].iloc[-1]),
        "n_months": int(len(df)), "cum_net": cum, "cur_dd": cur_dd, "kill": kill,
        "thresholds": {"alert": ALERT_DD, "derisk": DERISK_DD, "stop": STOP_DD},
        "slip_mean_bp": (float(slip_all.mean() * 1e4) if len(slip_all) else None),
        "slip_n": int(len(slip_all)),
        "plan_band": "年率 OOS +0.45〜+0.82（判定には12ヶ月以上の蓄積が必要）",
        "freshness": {"jq_daily": f"{cl_adj.index[-1]:%Y-%m-%d}", **fresh_ext},
        "capital": {"eq": float(man["capital_eq"]), "ts": float(man["capital_ts"])},
    }
    (DIR / "status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n出力: {DIR / 'report_latest.md'} / status.json / months.csv / "
          f"equity_daily.csv（{len(daily_df)}日分）")
    if bad_months:
        print(f"\nERROR: 月次リターンに欠損（{', '.join(bad_months)}）。"
              "キルスイッチ判定不能＝データを修復してください。")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
