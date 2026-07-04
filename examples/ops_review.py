"""運用レビュー（/ops-review の決定論的コア）— Phase 2 成果物のリコンサイルと執行監査。

一次資料: pysystemtrade production docs（リコンサイル・スリッページ分解）・Carver の
speed limit（年間コスト ≤ 事前コスト SR の 1/3）・年次レビューテンプレート
（research_loop/surveys/2026-07-04 §A 参照・Area 1/2/7）。

検査:
  1. リコンサイル: equity_daily ⇔ status.json ⇔ months.csv の相互整合・
     intended ⊇ orders ⊇ fills のカバレッジ・manifest 資本整合
  2. 執行整合（ペーパー特有）: fills の fill_price ⇔ wide パネルの T+1 始値の一致
     （乖離＝執行規約 or データのバグ。ペーパーはここが「ブローカー・ブレイク」相当）
  3. コスト監査: 想定コストモデルでの月間コスト・回転・Carver speed limit 判定
  4. 月次アトリビューション: スリーブ別リターン・実現ボラ vs 目標・combo 再構成差
出力: コンソール＋research_ops/ops_review_log.jsonl append。修復・発注はしない（報告のみ）。
実行: $env:PYTHONUTF8="1"; .venv\\Scripts\\python.exe examples\\ops_review.py [--month YYYY-MM]
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

P2 = ROOT / "data" / "phase2"
OPS = ROOT / "research_ops"
LOG = OPS / "ops_review_log.jsonl"
COST_BPS_EQ = 15.0          # 想定片道コスト（判定と同一の床）
CLAIMED_PRE_SR = 0.55       # speed limit 用の事前コストSR目安（認定0.45+コスト戻し・保守）
FINDINGS: list[dict] = []


def finding(check: str, severity: str, detail: str) -> None:
    FINDINGS.append({"check": check, "severity": severity, "detail": detail[:400]})
    mark = {"OK": "✓", "INFO": "·", "WARN": "⚠", "FAIL": "✗"}[severity]
    print(f"  {mark} [{severity}] {check}: {detail}")


def latest_month() -> str:
    ms = sorted(p.stem.split("_")[1] for p in P2.glob("manifest_*.json"))
    return ms[-1] if ms else ""


def main(argv: list[str]) -> int:
    month = argv[argv.index("--month") + 1] if "--month" in argv else latest_month()
    print(f"運用レビュー  {datetime.now():%Y-%m-%d %H:%M}  対象月={month}")
    curve = pd.read_csv(P2 / "equity_daily.csv", parse_dates=["date"]).set_index("date")
    status = json.loads((P2 / "status.json").read_text(encoding="utf-8"))
    months = pd.read_csv(P2 / "months.csv")
    manifest = json.loads((P2 / f"manifest_{month}.json").read_text(encoding="utf-8"))
    orders_eq = pd.read_csv(P2 / f"orders_eq_{month}.csv", encoding="utf-8-sig",
                            dtype={"code": str})
    intended = pd.read_parquet(P2 / f"intended_{month}.parquet")
    fills_p = P2 / f"fills_{month}.parquet"
    fills = pd.read_parquet(fills_p) if fills_p.exists() else pd.DataFrame()

    # --- 1. リコンサイル ---------------------------------------------------
    print("\n== 1. リコンサイル ==")
    cum_curve = float(curve["combo_net"].iloc[-1] - 1.0)
    cum_status = float(status.get("cum_net", np.nan))
    if abs(cum_curve - cum_status) < 5e-3:
        finding("reconcile.curve_vs_status", "OK",
                f"combo_net 累計 {cum_curve:+.4f} ≈ status.cum_net {cum_status:+.4f}")
    else:
        finding("reconcile.curve_vs_status", "FAIL",
                f"カーブ {cum_curve:+.4f} vs status {cum_status:+.4f}＝乖離"
                "（サイレント状態ドリフト＝ペーパー版のブローカー・ブレイク）")
    mrow = months[months["month"] == month]
    if len(mrow):
        finding("reconcile.months_row", "OK",
                f"months.csv に {month} 行あり（status={mrow['status'].iloc[0]}・"
                f"unfilled={int(mrow['unfilled_names'].iloc[0])}）")
    else:
        finding("reconcile.months_row", "FAIL", f"months.csv に {month} 行が無い")
    int_eq = set(intended.loc[intended["sleeve"].astype(str).str.contains("eq|value|pead",
                 case=False, regex=True), "key"].astype(str)) or set(
        intended["key"].astype(str))
    # ヘッジ行（指数先物等・数字コードでない）は株式注文の突合から分離
    is_hedge = ~orders_eq["code"].astype(str).str.fullmatch(r"\d{4,5}[A-Z0-9]?")
    hedge_rows = orders_eq[is_hedge]
    if len(hedge_rows):
        finding("reconcile.hedge_lines", "INFO",
                f"ヘッジ行 {hedge_rows['code'].tolist()}（株式突合から分離）")
    orders_eq = orders_eq[~is_hedge]
    ord_keys = set(orders_eq["code"].astype(str))
    if ord_keys - set(intended["key"].astype(str)):
        finding("reconcile.orders_in_intended", "WARN",
                f"orders に intended 外のコード {sorted(ord_keys - set(intended['key'].astype(str)))[:5]}")
    else:
        finding("reconcile.orders_in_intended", "OK",
                f"orders {len(ord_keys)}銘柄は全て intended（{len(int_eq)}）の部分集合")
    if len(fills):
        fill_keys = set(fills["key"].astype(str))
        missing = ord_keys - fill_keys
        finding("reconcile.fills_coverage", "OK" if not missing else "WARN",
                f"fills {len(fill_keys)}/{len(ord_keys)} 銘柄"
                + (f"・未約定 {sorted(missing)[:5]}" if missing else ""))
    cap = manifest.get("capital_eq", 0) + manifest.get("capital_ts", 0)
    cap_s = status.get("capital", {})
    if cap == cap_s.get("eq", 0) + cap_s.get("ts", 0):
        finding("reconcile.capital", "OK", f"資本整合 ¥{cap:,.0f}")
    else:
        finding("reconcile.capital", "FAIL", f"manifest {cap} vs status {cap_s}")

    # --- 2. 執行整合（fill ⇔ T+1 始値）--------------------------------------
    print("\n== 2. 執行整合（ペーパー：fill は翌営業日寄付と一致すべき）==")
    if len(fills):
        try:
            from invest_system.data.store import load_wide
            opn = load_wide("open")
            devs = []
            for _, r in fills.iterrows():
                k, d, fp = str(r["key"]), pd.Timestamp(r["fill_date"]), float(r["fill_price"])
                if k in opn.columns and d in opn.index and pd.notna(opn.at[d, k]):
                    devs.append((k, abs(fp / float(opn.at[d, k]) - 1.0) * 1e4))
            if devs:
                arr = np.array([x for _, x in devs])
                bad = [(k, x) for k, x in devs if x > 10.0]
                if bad:
                    finding("execution.fill_vs_open", "FAIL",
                            f"fill と T+1 始値の乖離>10bp が {len(bad)}件 例: {bad[:3]}")
                else:
                    finding("execution.fill_vs_open", "OK",
                            f"{len(devs)}件全て寄付一致（最大乖離 {arr.max():.1f}bp）")
        except Exception as e:  # noqa: BLE001
            finding("execution.fill_vs_open", "INFO", f"照合スキップ: {e}")
    else:
        finding("execution.fill_vs_open", "INFO", "fills なし（月中）")

    # --- 3. コスト監査（Carver speed limit）---------------------------------
    print("\n== 3. コスト監査 ==")
    notional = float(orders_eq["yen"].abs().sum())   # 売買とも取引額＝絶対値集計
    cap_eq = float(manifest.get("capital_eq", np.nan))
    turn_m = notional / cap_eq if cap_eq else np.nan
    cost_m = turn_m * COST_BPS_EQ / 1e4
    cost_ann = cost_m * 12
    finding("cost.model", "INFO",
            f"約定予定額 ¥{notional:,.0f}（資本比 {turn_m:.1%}/月）→ 想定コスト "
            f"{cost_m * 1e4:.1f}bp/月 ≈ 年率 {cost_ann:.2%}")
    # speed limit: 年間コスト（SR単位）≤ 事前コストSRの 1/3
    vol_ann = float(curve["eq"].pct_change().std() * np.sqrt(252)) if len(curve) > 5 else np.nan
    if np.isfinite(vol_ann) and vol_ann > 0:
        cost_sr = cost_ann / vol_ann
        lim = CLAIMED_PRE_SR / 3
        sev = "OK" if cost_sr <= lim else "WARN"
        finding("cost.speed_limit", sev,
                f"コストのSR単位 {cost_sr:.2f} vs 上限 {lim:.2f}"
                f"（事前コストSR {CLAIMED_PRE_SR} の1/3・実現ボラ{vol_ann:.1%}換算）")
    for mult, label in [(0.5, "0.5×"), (2.0, "2×")]:
        finding("cost.sensitivity", "INFO",
                f"コスト{label}時の年率ドラッグ ≈ {cost_ann * mult:.2%}")

    # --- 4. 月次アトリビューション -------------------------------------------
    print("\n== 4. 月次アトリビューション ==")
    r = curve.pct_change().dropna()
    per = r.index.to_period("M")
    for col, label, vt in [("eq", "旗艦", None), ("ts", "TSMOM", 0.10),
                           ("combo_net", "合成", None)]:
        mret = (1 + r[col]).groupby(per).prod() - 1
        vol = float(r[col].std() * np.sqrt(252))
        vline = f"・実現ボラ {vol:.1%}" + (f"（目標 {vt:.0%}）" if vt else "")
        finding(f"attribution.{col}", "INFO",
                f"{label}: 月次 " + " ".join(f"{p}={x:+.2%}" for p, x in mret.items())
                + vline)
    # combo 再構成（0.5/0.5×資本按分）との差
    w_eq = manifest.get("capital_eq", 0) / max(cap, 1)
    recon = w_eq * r["eq"] + (1 - w_eq) * r["ts"]
    diff = float(((1 + r["combo_net"]).prod() - (1 + recon).prod()))
    finding("attribution.combo_recon", "OK" if abs(diff) < 0.005 else "WARN",
            f"combo_net と資本按分再構成の累計差 {diff:+.4f}（コスト・端数・ヘッジ差）")

    n_fail = sum(1 for f in FINDINGS if f["severity"] == "FAIL")
    n_warn = sum(1 for f in FINDINGS if f["severity"] == "WARN")
    print(f"\n== 要約: FAIL={n_fail}  WARN={n_warn} ==")
    OPS.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"run": datetime.now().isoformat(timespec="seconds"),
                             "month": month, "fail": n_fail, "warn": n_warn,
                             "findings": FINDINGS}, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
