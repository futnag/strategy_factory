"""Phase 4 ― 事前登録プロトコルで「一度きりの DSR 判定」を実行（docs/19 FROZEN）。

**この判定は貴重な一回。** docs/19 の設定で 6 モデル族を **Combinatorial Purged CV（k=2・φ=5 パス/族）**で
OOS 評価。手数料後・廃止 last-price 補完の上位/下位デシル L/S の per-period Sharpe を、各族 φ=5 パスで
求める。**平均パス Sharpe 最大の族を採用**し、**N=6（族数）と V[SR]＝採用族のパス間分散**でデフレート（DSR）。
purge/embargo は特徴量後ろ向き窓（gap_after）＋ラベル前向き（gap_before=1）。

scope `phase4_gkx_judgment` に **判定を 1 件だけ**事前登録＋結果記録（一度きり・改竄不能）＝**K は +1**。
多重検定は N=6 で DSR 側が吸収（K+=6 は二重補正のためしない）。二重実行ガードで再実行を拒否。
合否：(i) DSR≥0.95 ∧ (ii) ネット Sharpe>0 ∧ (iii) 廃止封筒の全−100% 端でもネット Sharpe>0。
**越えても越えなくても docs/20 に正直に記録。通すための事後変更は一切しない（handoff §0,§3,§6）。**

実行（GO 後）: .venv\\Scripts\\python.exe examples\\run_phase4_judgment.py
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import skew as _skew, kurtosis as _kurtosis  # noqa: E402

from invest_system.equities import design_matrix as dm  # noqa: E402
from invest_system.research import phase4_estimators as est  # noqa: E402
from invest_system.research import phase4_evaluate as ev  # noqa: E402
from invest_system.validation import dsr as _dsr  # noqa: E402
from invest_system.validation.registry import default_registry  # noqa: E402

# === docs/19 FROZEN 設定 ===
SCOPE = "phase4_gkx_judgment"
FAMILIES = list(est.ALL_FAMILIES)                 # 6 族（N=6）
N_SPLITS, N_TEST = 6, 2                           # Combinatorial CPCV（φ=C(5,1)=5 パス/族）
GAP_BEFORE = 1                                    # ラベル 1M フォワードの前向き重複 purge
GAP_AFTER = 12                                    # 特徴量後ろ向き窓 embargo（docs/19 ★・確定12ヶ月）
OOS_START = "2018-01-01"
COSTS_BPS, DECILE = 15.0, 0.1
THRESH_DSR = 0.95
N_TRIALS = 6                                      # デフレート試行数（モデル族）。φ は算入しない。
OUT = Path(__file__).resolve().parent.parent / "docs" / "20-phase4-results.md"
HYP = ("GKX 型 ML が日本株クロスセクションの 1M 先トータル超過リターンを予測し、手数料後・"
       "廃止補完後の L/S デシルが多重検定デフレ後も正の Sharpe を持つ。")
RAT = ("全凍結特徴量（価格/流動性・EDINET 三表・拡充A/B）に銘柄横断の予測情報があれば、"
       "正則化 ML が線形を超える増分を生む。負なら『簡単なエッジ無し』が正当な結論。")


def _path_net_sharpes(paths, y, oos):
    """各 CPCV パスの 2018+ L/S ネット月次 Sharpe のリスト（oos は (date,Code) の bool Series）。"""
    out = []
    for p in paths:
        pe = p.where(oos.reindex(p.index))
        net = ev.ls_portfolio_returns(pe, y, DECILE, COSTS_BPS)["net"].dropna()
        if len(net) >= 2 and net.std(ddof=1) > 0:
            out.append(_dsr.sharpe_ratio(net.to_numpy()))
    return out


def main() -> int:
    t0 = time.time()
    print(f"=== Phase 4 一度きりの DSR 判定（CPCV k={N_TEST}・φ=5・N={N_TRIALS}・"
          f"gap_after={GAP_AFTER}ヶ月）===", flush=True)
    reg = default_registry()
    if reg.trial_count(SCOPE) > 0:
        print(f"[ABORT] scope '{SCOPE}' に既に {reg.trial_count(SCOPE)} 試行＝判定は一度きり。"
              "再実行で K を水増ししない（事後の再判定は p-hacking）。", flush=True)
        return 2
    D = dm.load_design_matrix("data/phase4")
    oos = pd.Series(D.X.index.get_level_values("date") >= pd.Timestamp(OOS_START),
                    index=D.X.index)               # (date,Code) の bool（順序非依存）
    y_lp = D.y                                     # last-price 補完（キャッシュ済）
    y_m100 = dm.build_label(D.months, list(D.universe.columns), D.universe,
                            delist_policy="all_minus100").stack(future_stack=True).dropna()
    y_m100 = y_m100.reindex(D.y.index)
    print(f"D loaded: X {D.X.shape} | OOS≥{OOS_START}: {int(oos.sum())} rows", flush=True)

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
              f"meanPathSR(月)={fam_mean[fam]:+.3f} (±{np.std(psr) if psr else 0:.3f}, "
              f"{len(psr)}パス) turn={fe.turnover:.2f} ({time.time()-ts:.0f}s)", flush=True)

    incr = ev.linear_vs_ml_increment(fam_eval)
    # 採用＝平均パス Sharpe 最大の族
    cand = {f: m for f, m in fam_mean.items() if np.isfinite(m)}
    best = max(cand, key=cand.get)
    psr_b = fam_pathsr[best]
    sr = float(np.mean(psr_b))                      # 採用族の平均パス Sharpe（点推定）
    sr_var = float(np.var(psr_b, ddof=1)) if len(psr_b) > 1 else 0.0   # V[SR]=パス間分散
    cons_b = est.consensus(fam_paths[best])
    cons_b = cons_b.where(oos.reindex(cons_b.index))
    rep = ev.ls_portfolio_returns(cons_b, y_lp, DECILE, COSTS_BPS)["net"].dropna()
    r = rep.to_numpy()
    sk, ku = float(_skew(r, bias=False)), float(_kurtosis(r, fisher=False, bias=False))
    dsr_val = _dsr.deflated_sharpe_ratio(sr, sr_var, N_TRIALS, len(r), sk, ku)
    # 廃止封筒：採用族の同一予測を全−100% ラベルで再評価
    psr_m100 = _path_net_sharpes(fam_paths[best], y_m100, oos)
    sr_m100 = float(np.mean(psr_m100)) if psr_m100 else float("nan")
    cond = {"dsr": dsr_val >= THRESH_DSR, "net": sr > 0, "envelope": sr_m100 > 0}
    verdict = "PASS" if all(cond.values()) else "FAIL"
    print(f"\n採用族={best} 平均パスSR(月)={sr:+.3f} V[SR]={sr_var:.4f} DSR={dsr_val:.3f} "
          f"封筒[last_price {sr:+.3f} / -100% {sr_m100:+.3f}] → {verdict} {cond}", flush=True)

    # レジストリ：判定を 1 件だけ登録（K+=1）。DSR は外部算出（N=6・V[パス]）して extra に保存。
    uid = reg.preregister(scope=SCOPE, hypothesis=HYP, economic_rationale=RAT,
                          strategy_id=f"phase4_gkx_{best}",
                          params={"selected_family": best, "families": FAMILIES,
                                  "cpcv": [N_SPLITS, N_TEST], "n_trials": N_TRIALS,
                                  "gap_before": GAP_BEFORE, "gap_after": GAP_AFTER,
                                  "oos_start": OOS_START, "decile": DECILE,
                                  "costs_bps": COSTS_BPS, "threshold_dsr": THRESH_DSR})
    reg.record_result(uid, sharpe=sr, n_obs=int(len(r)), skew=sk, kurt=ku,
                      extra={"deflated_sharpe": dsr_val, "sr_variance_paths": sr_var,
                             "n_trials": N_TRIALS, "selected_family": best,
                             "path_sharpes": [round(x, 4) for x in psr_b],
                             "envelope_minus100_meanSR": sr_m100,
                             "family_meanPathSR": {f: round(v, 4) for f, v in fam_mean.items()},
                             "verdict": verdict, "conditions": cond})
    print(f"レジストリ登録：scope '{SCOPE}' K={reg.trial_count(SCOPE)}（判定1件）", flush=True)

    _write_results(OUT, fam_eval, fam_mean, fam_pathsr, incr, best, sr, sr_var, dsr_val,
                   sr_m100, cond, verdict, len(r), time.time() - t0)
    print(f"→ {OUT}  (total {time.time()-t0:.0f}s)", flush=True)
    return 0


def _write_results(out, fam_eval, fam_mean, fam_pathsr, incr, best, sr, sr_var, dsr_val,
                   sr_m100, cond, verdict, n_obs, secs):
    a = np.sqrt(12)
    L = ["# 20. Phase 4 結果 ― GKX 型 ML・一度きりの DSR 判定", "",
         f"事前登録 docs/19（FROZEN）に従い **一度だけ** 実行（{pd.Timestamp.now():%Y-%m-%d}）。"
         f"CPCV k={N_TEST}・φ=5 パス/族・N={N_TRIALS} 族デフレート・gap_after={GAP_AFTER}ヶ月。"
         "**合否いずれも正直に記録**し、通すための事後変更はしない（handoff §6）。", "",
         "---", "", "## 1. 族別 OOS（CPCV φ=5・2018-01+・手数料後・廃止 last-price）", "",
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
          "## 2. 判定（事前登録の単一指標・deflated DSR・一度きり）", "",
          f"- **採用族（平均パス Sharpe 最大）**：`{best}`。",
          f"- 平均パス Sharpe = **{sr:+.3f}/月**（年率 {sr*a:+.2f}・n={n_obs}ヶ月）。",
          f"- **V[SR]（φ=5 パス間分散）= {sr_var:.4f}**。",
          f"- **Deflated Sharpe（N={N_TRIALS} 族・E[max SR] 補正）= {dsr_val:.3f}**。",
          f"- **廃止封筒**：last-price {sr:+.3f} ↔ 全−100% {sr_m100:+.3f}（平均パス Sharpe/月）。",
          f"- 合否条件：(i)DSR≥{THRESH_DSR}={cond['dsr']} ∧ (ii)ネット>0={cond['net']} ∧ "
          f"(iii)封筒−100%>0={cond['envelope']} → **判定：{verdict}**。",
          "",
          "## 3. 正直な結論", "",
          (f"**{verdict}。** " + ("3 条件すべてを満たした＝多重検定（N=6）・手数料・廃止封筒を"
           "踏まえても正の経済価値の証拠。" if verdict == "PASS" else
           "条件のいずれかを満たさない＝**この凍結特徴量セット・この期間では、手数料・多重検定・"
           "廃止リスクを越えるエッジは認められない**。これは**正当な結論**であり、通すための特徴量"
           "追加・ユニバース変更・再チューニング・再判定はしない（handoff §0）。")),
          "", "## 4. 規律・再現性", "",
          f"- レジストリ scope `{SCOPE}` に**判定 1 件のみ**登録＝**K +1**（多重検定は N={N_TRIALS} で"
          "DSR が吸収・K+=6 の二重補正はしない）。`examples/registry_status.py` で確認可。",
          "- 頑健性（部分標本/除外テール/レジーム/mcap床/ホライズン）は throwaway 診断で別途（K 不変）。",
          f"- seed=20260621・決定的。実行 {secs:.0f}s。"]
    out.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
