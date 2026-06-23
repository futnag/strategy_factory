"""因果的（PIT・オンライン）レジーム検知の診断（使い捨て・K不変・判定なし）。

docs/27 は全期間を見て当てる**事後**分析（先読みあり）。本診断は各時点 t で**≤t の
データのみ**を用いてレジーム転換を検知し、docs/27 の“真の”転換点に対する**検出ラグ**
（何か月遅れて気づけたか）と**誤警報**を測る。＝「生運用で実際に気づけたか」の検証。

オンライン手段（すべて ≤t のみ参照）:
  C1 因果ボラ/トレンド・レジーム（timeseries.regime の拡張窓三分位を再利用）
     ＋事後（全期間分位）との対比でラグを可視化
  C2 Page の CUSUM（平均シフト=リターン／ボラシフト=|リターン|・閾値超で警報・リセット）
  C3 拡張窓マルコフ・スイッチングの**フィルタ確率**（因果）vs 平滑確率（事後）
  C4 ベイズ・オンライン変化点検知 BOCPD（Adams-MacKay・Student-t 予測・run-length 事後）
  C5 トレーリング・ドローダウン・トリガ（弱気相場のリアルタイム検知）

すべて data/ のローカルキャッシュのみ（APIキー不要・オフライン）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest_system.timeseries.regime import efficiency_ratio, expanding_tertile  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"

# docs/27 で確定した“真の”構造転換（事後・複数手段の合意）
TRUE_BREAKS = ["2018-10", "2020-03", "2020-12", "2022-09"]


def topix_monthly_returns():
    df = pd.read_parquet(DATA / "jquants/indices/code_0000.parquet")
    s = df.set_index("Date")["C"].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.resample("ME").last().pct_change().dropna()


# ---------------------------------------------------------------- C2 Page CUSUM
def page_cusum(x: pd.Series, k_sd=0.5, h_sd=4.0, min_periods=18):
    """両側 Page CUSUM。基準平均/分散は ≤t 拡張窓（因果）。警報で位置を記録＆リセット。"""
    mu = x.expanding(min_periods=min_periods).mean().shift(1)
    sd = x.expanding(min_periods=min_periods).std().shift(1)
    s_hi = s_lo = 0.0
    alarms = []
    for t, v in x.items():
        m, d = mu.get(t), sd.get(t)
        if not np.isfinite(m) or not np.isfinite(d) or d == 0:
            continue
        z = (v - m) / d
        s_hi = max(0.0, s_hi + z - k_sd)
        s_lo = min(0.0, s_lo + z + k_sd)
        if s_hi > h_sd or s_lo < -h_sd:
            alarms.append(t)
            s_hi = s_lo = 0.0
    return alarms


# ------------------------------------------------------- C4 BOCPD (Adams-MacKay)
def bocpd(x, hazard_mean=36, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=None):
    """Bayesian Online Changepoint Detection（Student-t 予測・NIG 共役・定数ハザード）。

    各 t で run-length（最後の変化点からの経過）事後を更新。MAP run-length が急落した
    月を変化点として返す（オンライン＝≤t のみ）。
    """
    from scipy.stats import t as student_t
    x = np.asarray(x, float)
    n = len(x)
    if beta0 is None:
        beta0 = float(np.var(x)) if np.var(x) > 0 else 1.0
    H = 1.0 / hazard_mean
    # NIG パラメータ（各 run-length 仮説ごと）
    mu = np.array([mu0]); kappa = np.array([kappa0])
    alpha = np.array([alpha0]); beta = np.array([beta0])
    R = np.array([1.0])                       # run-length 事後（r=0..）
    maps = np.zeros(n, dtype=int)
    for i in range(n):
        xi = x[i]
        # 予測（Student-t）：自由度 2α, location μ, scale sqrt(β(κ+1)/(ακ))
        scale = np.sqrt(beta * (kappa + 1.0) / (alpha * kappa))
        pred = student_t.pdf(xi, df=2 * alpha, loc=mu, scale=scale)
        growth = R * pred * (1.0 - H)         # 継続
        cp = np.sum(R * pred * H)             # 変化点（r=0 へ）
        R = np.concatenate([[cp], growth])
        R /= R.sum()
        maps[i] = int(np.argmax(R))
        # 事後更新（NIG）— 既存仮説を1ステップ更新し、先頭に prior を追加
        mu_n = (kappa * mu + xi) / (kappa + 1.0)
        kappa_n = kappa + 1.0
        alpha_n = alpha + 0.5
        beta_n = beta + (kappa * (xi - mu) ** 2) / (2.0 * (kappa + 1.0))
        mu = np.concatenate([[mu0], mu_n])
        kappa = np.concatenate([[kappa0], kappa_n])
        alpha = np.concatenate([[alpha0], alpha_n])
        beta = np.concatenate([[beta0], beta_n])
        # 裾の刈り込み（数値安定）
        if len(R) > 200:
            R, mu, kappa, alpha, beta = (a[:200] for a in (R, mu, kappa, alpha, beta))
            R /= R.sum()
    return maps


# ------------------------------------------ C3 expanding Markov filtered vs smoothed
def markov_causal_vs_expost(ret, warmup=48, refit_every=3):
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    y = (ret * 100.0).astype(float)
    n = len(y)

    def turb_index(values, fp):                   # 高分散レジーム = 波乱
        v = []
        for k in range(fp.shape[1]):
            w = fp[:, k]
            m = np.average(values, weights=w)
            v.append(np.average((values - m) ** 2, weights=w))
        return int(np.argmax(v))

    # 事後（平滑）＝全期間でフィット（先読みあり・基準）
    mod_full = MarkovRegression(y.values, k_regimes=2, trend="c",
                                switching_variance=True)
    res_full = mod_full.fit(em_iter=100, search_reps=20, disp=False)
    sp = np.asarray(res_full.smoothed_marginal_probabilities)
    if sp.shape[0] != n:
        sp = sp.T
    tt = turb_index(ret.values, sp)
    smoothed = pd.Series(sp[:, tt], index=y.index)

    # 因果（フィルタ）＝拡張窓で params 推定（≤t）→ filtered prob[t]
    filt = pd.Series(np.nan, index=y.index)
    params, turb = None, None
    for i in range(warmup, n):
        sub = y.iloc[:i + 1]
        try:
            if params is None or (i - warmup) % refit_every == 0:
                r = MarkovRegression(sub.values, k_regimes=2, trend="c",
                                     switching_variance=True
                                     ).fit(em_iter=25, search_reps=6, disp=False)
                params = r.params
                fp = np.asarray(r.filtered_marginal_probabilities)
                if fp.shape[0] != len(sub):
                    fp = fp.T
                turb = turb_index(ret.iloc[:i + 1].values, fp)
            else:
                m = MarkovRegression(sub.values, k_regimes=2, trend="c",
                                     switching_variance=True)
                fp = np.asarray(m.filter(params).filtered_marginal_probabilities)
                if fp.shape[0] != len(sub):
                    fp = fp.T
            filt.iloc[i] = fp[-1, turb]
        except Exception:                          # noqa: BLE001
            continue
    return smoothed, filt


# ------------------------------------------------------------------ lag scoring
def _entries(flag: pd.Series):
    """False→True の立ち上がり月のリスト。"""
    f = flag.fillna(False).astype(bool)
    return list(f[(f) & (~f.shift(1).fillna(False))].index)


def lag_report(alarms_by_method: dict, breaks):
    bks = [pd.Timestamp(b) + pd.offsets.MonthEnd(0) for b in breaks]
    print(f"\n{'手段':<22}" + "".join(f"{b:%Y-%m}".rjust(10) for b in bks)
          + "  誤警報")
    for name, al in alarms_by_method.items():
        al = sorted(pd.Timestamp(a) for a in al)
        cells = ""
        near = set()
        for b in bks:
            cand = [a for a in al if (b - pd.offsets.MonthEnd(1)) <= a
                    <= b + pd.offsets.MonthEnd(12)]
            if cand:
                lag = (min(cand).to_period("M") - b.to_period("M")).n
                cells += f"{f'{lag:+d}mo':>10}"
                near.add(min(cand))
            else:
                cells += f"{'—':>10}"
        false = sum(1 for a in al if all(
            abs((a.to_period('M') - b.to_period('M')).n) > 3 for b in bks))
        print(f"{name:<22}{cells}  {false}")


def main() -> int:
    print("=" * 80)
    print("因果的レジーム検知（オンライン・≤t のみ）— 検出ラグと誤警報")
    print("=" * 80)
    ret = topix_monthly_returns()
    print(f"TOPIX 月次 {ret.index.min():%Y-%m}〜{ret.index.max():%Y-%m} ({len(ret)}か月)")
    print("事後“真の”転換点（docs/27）:", ", ".join(TRUE_BREAKS))

    alarms = {}

    # --- C1 因果ボラ/トレンド・レジーム（拡張窓三分位 vs 事後分位）-------------
    vol = ret.rolling(6).std() * np.sqrt(12)
    caus_hi = (expanding_tertile(vol, min_periods=24) == 2)        # ≤t 分位（因果）
    expost_hi = (vol > vol.quantile(0.66))                          # 全期間分位（事後）
    print("\n[C1] ボラ・レジーム HIGH 突入月")
    print("  因果(拡張窓三分位):", "  ".join(f"{d:%Y-%m}" for d in _entries(caus_hi)))
    print("  事後(全期間分位)  :", "  ".join(f"{d:%Y-%m}" for d in _entries(expost_hi)))
    alarms["C1 因果ボラHIGH"] = _entries(caus_hi)

    er = efficiency_ratio((1 + ret).cumprod(), window=6)
    caus_trend = (expanding_tertile(er, min_periods=24) == 2)
    alarms["C1 因果トレンド強"] = _entries(caus_trend)

    # --- C2 Page CUSUM -------------------------------------------------------
    cm = page_cusum(ret, k_sd=0.5, h_sd=4.0)
    cv = page_cusum((ret - ret.expanding(18).mean().shift(1)).abs().dropna(),
                    k_sd=0.5, h_sd=4.0)
    print("\n[C2] Page CUSUM 警報")
    print("  平均シフト(リターン):", "  ".join(f"{d:%Y-%m}" for d in cm) or "—")
    print("  ボラシフト(|残差|)  :", "  ".join(f"{d:%Y-%m}" for d in cv) or "—")
    alarms["C2 CUSUM平均"] = cm
    alarms["C2 CUSUMボラ"] = cv

    # --- C3 Markov フィルタ(因果) vs 平滑(事後)------------------------------
    print("\n[C3] マルコフ波乱状態：フィルタ(因果) vs 平滑(事後)")
    try:
        smoothed, filt = markov_causal_vs_expost(ret)
        s_in = _entries(smoothed > 0.5)
        f_in = _entries(filt > 0.5)
        print("  平滑(事後)突入:", "  ".join(f"{d:%Y-%m}" for d in s_in) or "—")
        print("  フィルタ(因果)突入:", "  ".join(f"{d:%Y-%m}" for d in f_in) or "—")
        alarms["C3 Markovフィルタ"] = f_in
    except Exception as e:                                          # noqa: BLE001
        print("  （Markov 失敗）", e)

    # --- C4 BOCPD ------------------------------------------------------------
    maps = bocpd(ret.values, hazard_mean=36)
    mser = pd.Series(maps, index=ret.index)
    # MAP run-length が直近から急落した月＝変化点
    drops = mser.index[(mser.shift(1) > 6) & (mser <= 1)]
    print("\n[C4] BOCPD 変化点（MAP run-length 急落）:",
          "  ".join(f"{d:%Y-%m}" for d in drops) or "—")
    alarms["C4 BOCPD"] = list(drops)

    # --- C5 ドローダウン・トリガ --------------------------------------------
    cum = (1 + ret).cumprod()
    dd = cum / cum.cummax() - 1.0
    trig = _entries(dd < -0.10)
    print("\n[C5] ドローダウン<-10% 突入:", "  ".join(f"{d:%Y-%m}" for d in trig) or "—")
    alarms["C5 DD<-10%"] = trig

    # --- 検出ラグ表 ----------------------------------------------------------
    print("\n" + "=" * 80)
    print("検出ラグ（各“真の”転換点に対し、最初の警報が何か月遅れか／—=見逃し）")
    print("=" * 80)
    lag_report(alarms, TRUE_BREAKS)
    print("\n注：±1か月以内=ほぼ即時、+数か月=遅行、—=この窓では検知できず（見逃し）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
