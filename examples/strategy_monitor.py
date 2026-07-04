"""稼働戦略の健全性監視（/strategy-monitor の決定論的コア）。

一次資料: Bailey & López de Prado (J.Risk 2012)＝PSR/MinTRL・Carver の非対称キルルール
（「有意に負」でのみ殺す）・Fan-Jiao-Wang (Biometrika 2025)＝平均分散 e 過程・
WWZ (Mgmt Sci 2026)＝anytime-valid 閾値慣行（research_loop/surveys/2026-07-04 §A T3）。

対象: data/phase2/equity_daily.csv の各系列（eq / ts / combo_net）。
帰無仮説（認定値）: research_ops/monitor_config.json（人間が凍結・事後調整禁止）。

出力（系列ごと）:
  - n・年率SR・PSR(0)・PSR(認定SR)・MinTRL 充足率
  - 判定: insufficient_evidence（MinTRL未達＝原則）/ healthy / amber / review / kill_candidate
    ※ 非対称ルール: エスカレーションは「負の証拠」（e値・CUSUM・DD閾値）でのみ発生。
      「まだ有意に正でない」は insufficient のまま＝Carver の教訓。
  - FJW e 過程（月次・E=((m−tol−X)+)²/σ_max²・λ混合・走行最大値）
  - CUSUM（日次・下方一側）・DD vs キルスイッチ閾値距離
状態: research_ops/monitor_state.json（e過程・CUSUM の走行値＝リセット禁止）・
      research_ops/monitor_log.jsonl（append-only）。
実行: $env:PYTHONUTF8="1"; .venv\\Scripts\\python.exe examples\\strategy_monitor.py
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

from scipy import stats as sps  # noqa: E402

from invest_system.validation.dsr import (  # noqa: E402
    probabilistic_sharpe_ratio, min_track_record_length, sharpe_ratio,
)

OPS = ROOT / "research_ops"
CONFIG = OPS / "monitor_config.json"
STATE = OPS / "monitor_state.json"
LOG = OPS / "monitor_log.jsonl"
CURVE = ROOT / "data" / "phase2" / "equity_daily.csv"
ANN = 252.0


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"series": {}}


def monitor_series(name: str, nav: pd.Series, cfg: dict, ecfg: dict, ccfg: dict,
                   state: dict) -> dict:
    r = nav.pct_change().dropna()
    n = int(len(r))
    out: dict = {"series": name, "n_days": n}
    if n < 5:
        out["verdict"] = "insufficient_evidence"
        out["reason"] = f"n={n} < 5（開始直後）"
        return out

    sr_d = float(sharpe_ratio(r))
    sk = float(sps.skew(r, bias=False)) if n >= 8 else 0.0
    ku = float(sps.kurtosis(r, bias=False, fisher=False)) if n >= 8 else 3.0
    claimed_d = cfg["claimed_sr_ann"] / np.sqrt(ANN)
    tol_d = cfg["tol_sr_ann"] / np.sqrt(ANN)
    out.update({
        "sr_ann": round(sr_d * np.sqrt(ANN), 3),
        "psr_gt0": round(float(probabilistic_sharpe_ratio(sr_d, 0.0, n, sk, ku)), 3),
        "psr_vs_claimed": round(float(probabilistic_sharpe_ratio(
            sr_d, claimed_d - tol_d, n, sk, ku)), 3),
    })
    # MinTRL: 「認定SR−tol を下回っていない」と有意に言うために必要な日数
    try:
        mtrl = float(min_track_record_length(sr_d, claimed_d - tol_d, sk, ku))
    except Exception:  # noqa: BLE001
        mtrl = float("inf")
    out["min_trl_days"] = round(mtrl, 0) if np.isfinite(mtrl) else "∞（現SR<認定−tol のため定義不能）"
    out["track_sufficiency"] = round(n / mtrl, 2) if np.isfinite(mtrl) and mtrl > 0 else 0.0

    # --- FJW e 過程（月次・下方＝decay 検知。リセット禁止＝state に走行値）---
    sigma_d = cfg["sigma_ann_max"] / np.sqrt(ANN)
    m_d = claimed_d * sigma_d * np.sqrt(ANN) / np.sqrt(ANN)  # 日次期待リターン m=SR_d×σ_d
    m_d = claimed_d * sigma_d  # 同値（明示）
    monthly = (1.0 + r).groupby(r.index.to_period("M")).prod() - 1.0
    # 月内日数で月次の m/σ をスケール
    days_in_m = r.groupby(r.index.to_period("M")).size()
    st = state["series"].setdefault(name, {"e_wealth": {str(l): 1.0 for l in ecfg["lambda_grid"]},
                                           "e_done_months": [], "cusum": 0.0,
                                           "e_max": 1.0})
    for lam in ecfg["lambda_grid"]:
        st["e_wealth"].setdefault(str(lam), 1.0)
    for per, x in monthly.items():
        p = str(per)
        if p in st["e_done_months"] or days_in_m[per] < 15:  # 完成月のみ・二重計上禁止
            continue
        d = int(days_in_m[per])
        m_m = (m_d - tol_d * sigma_d) * d
        var_m = (sigma_d ** 2) * d
        e_stat = max(0.0, (m_m - float(x))) ** 2 / var_m  # FJW Thm.1
        for lam in ecfg["lambda_grid"]:
            st["e_wealth"][str(lam)] *= (1.0 - lam + lam * e_stat)
        st["e_done_months"].append(p)
    e_mix = float(np.mean(list(st["e_wealth"].values())))
    st["e_max"] = max(st.get("e_max", 1.0), e_mix)
    out["e_value"] = round(e_mix, 3)
    out["e_running_max"] = round(st["e_max"], 3)

    # --- CUSUM（日次・下方一側・k=0.5σ, h=5σ）---
    k = ccfg["k_sigma"] * sigma_d
    h = ccfg["h_sigma"] * sigma_d
    s = float(st.get("cusum", 0.0))
    done = st.get("cusum_done_n", 0)
    for x in r.iloc[done:]:
        s = max(0.0, s + (m_d - float(x)) - k)
    st["cusum"] = s
    st["cusum_done_n"] = n
    out["cusum_ratio"] = round(s / h, 2) if h > 0 else None

    # --- DD vs キルスイッチ（元本基準）---
    dd_from_par = float(nav.iloc[-1] - 1.0)
    out["ret_from_inception"] = round(dd_from_par, 4)
    return out


def verdict(out: dict, ecfg: dict, ddcfg: dict, dd_combo: float | None) -> tuple[str, str]:
    """非対称ルール: エスカレーションは負の証拠のみ。正の不足は insufficient のまま。"""
    th = ecfg["thresholds"]
    e = out.get("e_running_max", 1.0)
    cus = out.get("cusum_ratio") or 0.0
    if dd_combo is not None and dd_combo <= ddcfg["stop"]:
        return "kill_candidate", f"元本DD {dd_combo:.1%} ≤ stop {ddcfg['stop']:.0%}"
    if e >= th["kill_candidate"]:
        return "kill_candidate", f"e値 {e:.1f} ≥ {th['kill_candidate']}（α=5%相当の減衰証拠）"
    dd_s = f"{dd_combo:.1%}" if dd_combo is not None else "n/a"
    if e >= th["review"] or (dd_combo is not None and dd_combo <= ddcfg["derisk"]):
        return "review", f"e値 {e:.1f} / 元本DD {dd_s}"
    if e >= th["amber"] or cus >= 1.0 or (dd_combo is not None and dd_combo <= ddcfg["alert"]):
        why = ("CUSUM 発火" if cus >= 1.0 else
               f"e値 {e:.1f}" if e >= th["amber"] else "DD alert 閾値")
        return "amber", f"{why}（e={e:.1f}・CUSUM={cus:.2f}・元本DD={dd_s}）"
    if (out.get("track_sufficiency") or 0.0) < 1.0:
        return "insufficient_evidence", (
            f"n={out['n_days']}日 < MinTRL {out.get('min_trl_days')}日"
            f"（充足率 {out.get('track_sufficiency', 0):.0%}）＝判定不能が正直な状態")
    return "healthy", f"PSR(認定−tol)={out['psr_vs_claimed']:.2f}・負の証拠なし"


def main() -> int:
    print(f"稼働戦略 健全性監視  {datetime.now():%Y-%m-%d %H:%M}")
    cfgall = json.loads(CONFIG.read_text(encoding="utf-8"))
    curve = pd.read_csv(CURVE, parse_dates=["date"]).set_index("date")
    print(f"入力: {CURVE.relative_to(ROOT)}  {curve.index.min():%Y-%m-%d}〜"
          f"{curve.index.max():%Y-%m-%d}（{len(curve)}営業日）")
    if any(v.get("status") == "TO_CONFIRM" for v in cfgall["series"].values()):
        print("⚠ monitor_config.json の認定値は seed（TO_CONFIRM）＝人間の凍結確認待ち。")

    state = load_state()
    combo_dd = float(curve["combo_net"].iloc[-1] - 1.0) if "combo_net" in curve else None
    results = []
    for name, cfg in cfgall["series"].items():
        if name not in curve.columns:
            continue
        out = monitor_series(name, curve[name], cfg, cfgall["e_process"],
                             cfgall["cusum"], state)
        v, reason = verdict(out, cfgall["e_process"], cfgall["drawdown_thresholds"],
                            combo_dd if name == "combo_net" else None)
        out["verdict"], out["reason"] = v, reason
        results.append(out)
        print(f"\n== {cfg['label']}（{name}）==")
        for k_ in ("n_days", "sr_ann", "psr_gt0", "psr_vs_claimed", "min_trl_days",
                   "track_sufficiency", "e_value", "e_running_max", "cusum_ratio",
                   "ret_from_inception"):
            if k_ in out and out[k_] is not None:
                print(f"  {k_:<20} {out[k_]}")
        print(f"  → 判定: {v}  （{reason}）")

    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"run": datetime.now().isoformat(timespec="seconds"),
                             "asof": str(curve.index.max().date()),
                             "results": results}, ensure_ascii=False) + "\n")
    print(f"\n（走行状態を {STATE.name}・判定ログを {LOG.name} に記録。"
          "e過程/CUSUM はリセット禁止・認定値の事後調整禁止）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
