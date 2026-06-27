"""Phase 4b ― 小型寄りユニバース（mcap≥¥30B）で「一度きりの DSR 判定」（docs/21 FROZEN）。

Phase 4（¥10B 床・FAIL）とは独立した新仮説。唯一の変更点＝mcap 床。前判定 scope は不変。
scope `phase4b_gkx_judgment` に判定 1 件のみ登録（K+=1）。

実行: .venv\\Scripts\\python.exe examples\\run_phase4b_judgment.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001  # pragma: no cover
    pass

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import skew as _skew, kurtosis as _kurtosis  # noqa: E402

from invest_system.equities import design_matrix as dm  # noqa: E402
from invest_system.research import phase4_estimators as est  # noqa: E402
from invest_system.research import phase4_evaluate as ev  # noqa: E402
from invest_system.validation import dsr as _dsr  # noqa: E402
from invest_system.validation.registry import default_registry  # noqa: E402

# === docs/21 FROZEN（Phase 4 と同一・mcap 床のみ ¥30B）===
SCOPE = "phase4b_gkx_judgment"
CACHE = "data/phase4b"
PRESET = "smallcap_30b"
FAMILIES = list(est.ALL_FAMILIES)
N_SPLITS, N_TEST = 6, 2
GAP_BEFORE, GAP_AFTER = 1, 12
OOS_START = "2018-01-01"
COSTS_BPS, DECILE = 15.0, 0.1
THRESH_DSR = 0.95
N_TRIALS = 6
OUT = Path(__file__).resolve().parent.parent / "docs" / "22-phase4b-results.md"
HYP = ("GKX 型 ML が小型寄りユニバース（mcap≥¥30B）で 1M 先トータル超過リターンを予測し、"
       "手数料後・廃止補完後の L/S デシルが多重検定デフレ後も正の Sharpe を持つ。")
RAT = ("GKX アルファは小型に偏在する仮説。mcap 床を ¥100億→¥30億に下げることで"
       "非線形エッジが顕在化するか検定。負なら『小型寄りでも簡単なエッジ無し』が正当な結論。")


def _path_net_sharpes(paths, y, oos, costs_bps=COSTS_BPS):
    out = []
    for p in paths:
        pe = p.where(oos.reindex(p.index))
        net = ev.ls_portfolio_returns(pe, y, DECILE, costs_bps)["net"].dropna()
        if len(net) >= 2 and net.std(ddof=1) > 0:
            out.append(_dsr.sharpe_ratio(net.to_numpy()))
    return out


def _run_preflight_diag() -> str:
    """throwaway 診断を実行し stdout をキャプチャ。"""
    script = Path(__file__).resolve().parent / "phase4b_preflight_diag.py"
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", cwd=script.parent.parent)
    return r.stdout + (r.stderr if r.returncode else "")


def main() -> int:
    t0 = time.time()
    cache = Path(CACHE)
    if not (cache / "X.parquet").exists():
        print(f"[ABORT] {CACHE}/ が無い。先に build_phase4b_matrix.py を実行。", flush=True)
        return 2

    print(f"=== Phase 4b 一度きりの DSR 判定（preset={PRESET}・CPCV k={N_TEST}・φ=5）===", flush=True)
    reg = default_registry()
    if reg.trial_count(SCOPE) > 0:
        print(f"[ABORT] scope '{SCOPE}' に既に {reg.trial_count(SCOPE)} 試行＝判定は一度きり。",
              flush=True)
        return 2

    diag_text = _run_preflight_diag()
    print(diag_text, flush=True)

    D = dm.load_design_matrix(CACHE)
    oos = pd.Series(D.X.index.get_level_values("date") >= pd.Timestamp(OOS_START), index=D.X.index)
    y_lp = D.y
    y_m100 = dm.build_label(D.months, list(D.universe.columns), D.universe,
                            delist_policy="all_minus100").stack(future_stack=True).dropna()
    y_m100 = y_m100.reindex(D.y.index)
    avg_names = int(D.universe.sum(axis=1).mean())
    print(f"D loaded: X {D.X.shape} | OOS≥{OOS_START}: {int(oos.sum())} rows | "
          f"avg {avg_names} 銘柄/月", flush=True)

    fam_eval, fam_paths, fam_pathsr, fam_mean = {}, {}, {}, {}
    for fam in FAMILIES:
        ts = time.time()
        paths = est.cpcv_paths_predict(D, fam, N_SPLITS, N_TEST, GAP_BEFORE, GAP_AFTER)
        cons = est.consensus(paths)
        cons_e = cons.where(oos.reindex(cons.index))
        fe = ev.evaluate_family(fam, cons_e, y_lp, decile=DECILE, costs_bps=COSTS_BPS)
        psr = _path_net_sharpes(paths, y_lp, oos)
        fam_eval[fam], fam_paths[fam], fam_pathsr[fam] = fe, paths, psr
        fam_mean[fam] = float(np.mean(psr)) if psr else float("nan")
        print(f"  [{fam:11}] OOS R²={fe.oos_r2:+.4f} IC={fe.mean_ic:+.3f} "
              f"meanPathSR(月)={fam_mean[fam]:+.3f} ({len(psr)}パス) ({time.time()-ts:.0f}s)",
              flush=True)

    incr = ev.linear_vs_ml_increment(fam_eval)
    cand = {f: m for f, m in fam_mean.items() if np.isfinite(m)}
    best = max(cand, key=cand.get)
    psr_b = fam_pathsr[best]
    sr = float(np.mean(psr_b))
    sr_var = float(np.var(psr_b, ddof=1)) if len(psr_b) > 1 else 0.0
    cons_b = est.consensus(fam_paths[best]).where(oos.reindex(est.consensus(fam_paths[best]).index))
    rep = ev.ls_portfolio_returns(cons_b, y_lp, DECILE, COSTS_BPS)["net"].dropna()
    r = rep.to_numpy()
    sk, ku = float(_skew(r, bias=False)), float(_kurtosis(r, fisher=False, bias=False))
    dsr_val = _dsr.deflated_sharpe_ratio(sr, sr_var, N_TRIALS, len(r), sk, ku)
    psr_m100 = _path_net_sharpes(fam_paths[best], y_m100, oos)
    sr_m100 = float(np.mean(psr_m100)) if psr_m100 else float("nan")
    # コスト感応度（30bps）
    psr_30 = _path_net_sharpes(fam_paths[best], y_lp, oos, costs_bps=30.0)
    sr_30 = float(np.mean(psr_30)) if psr_30 else float("nan")
    cond = {"dsr": dsr_val >= THRESH_DSR, "net": sr > 0, "envelope": sr_m100 > 0}
    verdict = "PASS" if all(cond.values()) else "FAIL"
    print(f"\n採用族={best} SR(月)={sr:+.3f} DSR={dsr_val:.3f} 30bps={sr_30:+.3f} → {verdict}",
          flush=True)

    uid = reg.preregister(scope=SCOPE, hypothesis=HYP, economic_rationale=RAT,
                          strategy_id=f"phase4b_gkx_{best}",
                          params={"preset": PRESET, "min_mcap": 3e9,
                                  "selected_family": best, "families": FAMILIES,
                                  "cpcv": [N_SPLITS, N_TEST], "n_trials": N_TRIALS,
                                  "gap_before": GAP_BEFORE, "gap_after": GAP_AFTER,
                                  "oos_start": OOS_START, "decile": DECILE,
                                  "costs_bps": COSTS_BPS, "threshold_dsr": THRESH_DSR})
    reg.record_result(uid, sharpe=sr, n_obs=int(len(r)), skew=sk, kurt=ku,
                      extra={"deflated_sharpe": dsr_val, "sr_variance_paths": sr_var,
                             "n_trials": N_TRIALS, "selected_family": best,
                             "path_sharpes": [round(x, 4) for x in psr_b],
                             "envelope_minus100_meanSR": sr_m100,
                             "cost_30bps_meanSR": sr_30,
                             "family_meanPathSR": {f: round(v, 4) for f, v in fam_mean.items()},
                             "verdict": verdict, "conditions": cond})
    print(f"レジストリ：scope '{SCOPE}' K={reg.trial_count(SCOPE)}", flush=True)

    _write_results(OUT, fam_eval, fam_mean, fam_pathsr, incr, best, sr, sr_var, dsr_val,
                   sr_m100, sr_30, cond, verdict, len(r), avg_names, diag_text, time.time() - t0)
    print(f"→ {OUT}  ({time.time()-t0:.0f}s)", flush=True)
    return 0


def _write_results(out, fam_eval, fam_mean, fam_pathsr, incr, best, sr, sr_var, dsr_val,
                   sr_m100, sr_30, cond, verdict, n_obs, avg_names, diag_text, secs):
    a = np.sqrt(12)
    # Phase 4 参照値（docs/20）
    p4 = {"best": "rf", "sr": 0.208, "dsr": 0.905, "verdict": "FAIL", "avg_names": 1880}
    L = ["# 22. Phase 4b 結果 ― 小型寄りユニバース GKX ML・一度きりの DSR 判定", "",
         f"事前登録 docs/21（FROZEN）に従い **一度だけ** 実行（{pd.Timestamp.now():%Y-%m-%d}）。"
         f"**唯一の変更＝mcap 床 ¥100億→¥30億**（preset `smallcap_30b`）。他は Phase 4 と同一。",
         f"CPCV k={N_TEST}・φ=5・N={N_TRIALS} 族デフレート・gap_after={GAP_AFTER}ヶ月。", "",
         "---", "",
         "## 0. Phase 4 との並置", "",
         "| 項目 | Phase 4（¥10B） | Phase 4b（¥30B） |",
         "|---|---|---|",
         f"| 月平均銘柄数 | ~{p4['avg_names']} | ~{avg_names} |",
         f"| 採用族 | {p4['best']} | {best} |",
         f"| 平均パス SR(月) | {p4['sr']:+.3f} | {sr:+.3f} |",
         f"| Deflated SR | {p4['dsr']:.3f} | {dsr_val:.3f} |",
         f"| 判定 | {p4['verdict']} | {verdict} |",
         "",
         "## 1. 族別 OOS（CPCV φ=5・2018-01+・15bps・廃止 last-price）", "",
         "| 族 | OOS R² | meanIC | 平均パスSR(月) | 年率SR | パスSR範囲 | 回転率 |",
         "|---|---|---|---|---|---|---|"]
    for f in est.ALL_FAMILIES:
        e = fam_eval.get(f)
        if e:
            ps = fam_pathsr.get(f, [])
            rng = f"[{min(ps):+.2f},{max(ps):+.2f}]" if ps else "—"
            L.append(f"| {f} | {e.oos_r2:+.4f} | {e.mean_ic:+.3f} | {fam_mean[f]:+.3f} | "
                     f"{fam_mean[f]*a:+.2f} | {rng} | {e.turnover:.2f} |")
    L += ["",
          f"- **線形 vs ML 増分**：線形最良 R²={incr['linear_best_r2']:+.4f}／"
          f"ML 最良 R²={incr['ml_best_r2']:+.4f}（ML 超え：{incr['ml_beats_linear_r2']}）。"
          f"ネット Sharpe 線形最良={incr['linear_best_sharpe']:+.3f}／"
          f"ML 最良={incr['ml_best_sharpe']:+.3f}（ML 超え：{incr['ml_beats_linear_sharpe']}）。",
          "",
          "## 2. 判定（deflated DSR・一度きり）", "",
          f"- **採用族**：`{best}`。",
          f"- 平均パス Sharpe = **{sr:+.3f}/月**（年率 {sr*a:+.2f}・n={n_obs}ヶ月）。",
          f"- **V[SR]（φ=5 パス間分散）= {sr_var:.4f}**。",
          f"- **Deflated Sharpe（N={N_TRIALS}）= {dsr_val:.3f}**。",
          f"- **廃止封筒**：last-price {sr:+.3f} ↔ 全−100% {sr_m100:+.3f}。",
          f"- **コスト感応度（採用族・30bps 片道）**：平均パス SR = **{sr_30:+.3f}/月**。",
          f"- 合否：(i)DSR≥{THRESH_DSR}={cond['dsr']} ∧ (ii)ネット>0={cond['net']} ∧ "
          f"(iii)封筒>0={cond['envelope']} → **{verdict}**。",
          "",
          "## 3. 正直な結論", "",
          (f"**{verdict}。** " + ("3 条件を満たした。" if verdict == "PASS" else
           "小型寄りユニバースでも認定基準を越えない＝**正当な FAIL**。"
           "床以外の変更・再判定はしない（Handoff_phase4b §5）。")),
          "",
          "## 4. 小型固有診断（throwaway・K 不変）", "",
          "```", diag_text.strip(), "```", "",
          "## 5. 規律・再現性", "",
          f"- レジストリ scope `{SCOPE}` に判定 1 件＝**K +1**。前 scope `phase4_gkx_judgment` は不変。",
          f"- seed=20260621・決定的。実行 {secs:.0f}s。"]
    out.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())