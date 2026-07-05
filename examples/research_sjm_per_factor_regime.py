"""sjm_per_factor_regime — Stage-0 K=0 診断（red-team 2026-07-05 反映）.

per-factor regime 切替（SJM 型・2状態ジャンプモデル）が固定等ウエイト合成を上回るかの
**実行前キルチェック**。K=0（記述スキャン・パラメータグリッド無し・judge 未使用）。

PRE-FIXED（チューニング禁止・prereg docs/61 で固定）:
  factor set  = FF Japan {Mkt-RF, SMB, HML} 月次超過リターン（外部・1990-07+・後知恵選抜なし）
  model       = 2状態ジャンプモデル・特徴 z[ret], z[roll6 vol]・ジャンプペナルティ LAMBDA=8（固定）
  inference   = オンライン拡張窓（regime_t は data<=t のみ＝因果）・配分は t+1（execution_lag=1）
  allocation  = 各因子: 現 regime が高平均状態なら1・else 0 → 稼働因子で等ウエイト再正規化（全offは現金）
  fixed/常時オン = 各因子 1/3 を常時保有（＝改良の分母・H-5 の買い持ち基準）
  controls    = mom オーバーレイ（trailing-12m>0・H-1 統制）／placebo（regime ラベルをシード固定でシャッフル・H-18）

Stage-0 キル（red-team 義務化）:
  (1) H-11: 因子毎の in-sample regime 遷移数 ~1 なら「1ブレイク過適合」＝棄却候補
  (2) H-14: rho(switched, fixed) > 0.9 かつ switched Sharpe が fixed 超でない → DSR 張り付き＋エッジ無し＝棄却
"""
import numpy as np
import pandas as pd

LAMBDA = 8.0        # ジャンプペナルティ（事前固定・非チューニング）
VOL_WIN = 6         # ボラ特徴の窓（月）
MIN_TRAIN = 120     # 初期学習（10年）後に最初の regime を推定
MOM_WIN = 12        # mom 統制のトレーリング窓
SEED = 12345
FACTORS = ["Mkt-RF", "SMB", "HML"]
ANN = np.sqrt(12.0)


def jump_model_2state(X, lam=LAMBDA, n_iter=15):
    """2状態ジャンプモデル（座標降下）. X:(T,d) 標準化済み. 状態ラベル(T,)∈{0,1} を返す."""
    T = X.shape[0]
    s = (X[:, 0] > 0).astype(int)           # 初期化: 第1特徴（return）の符号
    for _ in range(n_iter):
        mu = np.array([X[s == k].mean(0) if (s == k).any() else X.mean(0) for k in range(2)])
        cost = ((X[:, None, :] - mu[None, :, :]) ** 2).sum(2)   # (T,2)
        dp = np.zeros((T, 2)); back = np.zeros((T, 2), int)
        dp[0] = cost[0]
        for t in range(1, T):
            for k in range(2):
                trans = dp[t - 1] + lam * (np.arange(2) != k)
                j = int(trans.argmin()); back[t, k] = j; dp[t, k] = cost[t, k] + trans[j]
        s_new = np.zeros(T, int); s_new[-1] = int(dp[-1].argmin())
        for t in range(T - 1, 0, -1):
            s_new[t - 1] = back[t, s_new[t]]
        if (s_new == s).all():
            s = s_new; break
        s = s_new
    return s


def features(ret_window):
    """拡張窓の [z(return), z(roll6 vol)] を返す（rolling は因果）."""
    r = ret_window
    vol = pd.Series(r).rolling(VOL_WIN).std().bfill().values
    zr = (r - r.mean()) / (r.std() + 1e-9)
    zv = (vol - vol.mean()) / (vol.std() + 1e-9)
    return np.column_stack([zr, zv])


def online_regime_good(ret):
    """オンライン因果 regime: regime_t は data<=t のみ使用. 高平均状態=good を 1/0 で返す（<MIN_TRAIN は NaN）."""
    r = ret.values.astype(float); T = len(r)
    good = np.full(T, np.nan)
    for t in range(MIN_TRAIN, T):
        X = features(r[: t + 1])
        s = jump_model_2state(X)
        m0 = r[: t + 1][s == 0].mean() if (s == 0).any() else -np.inf
        m1 = r[: t + 1][s == 1].mean() if (s == 1).any() else -np.inf
        gstate = 0 if m0 >= m1 else 1
        good[t] = 1.0 if s[t] == gstate else 0.0
    return pd.Series(good, index=ret.index)


def full_sample_transitions(ret):
    """記述カウント（look-ahead 可＝データ構造の性質）: in-sample regime 遷移回数."""
    X = features(ret.values.astype(float))
    s = jump_model_2state(X)
    return int((np.diff(s) != 0).sum())


def combo_from_signal(rets, sig):
    """sig:(T,F) の 0/1 稼働フラグ → 稼働因子の等ウエイト再正規化リターン（execution_lag=1）."""
    w = sig.shift(1)                    # t の配分は t-1 の regime（T+1 執行）
    wsum = w.sum(axis=1)
    w = w.div(wsum.replace(0, np.nan), axis=0).fillna(0.0)   # 全off は現金(0)
    return (w * rets).sum(axis=1)


def sharpe(x):
    x = x.dropna()
    return float(x.mean() / (x.std() + 1e-12) * ANN)


def main():
    df = pd.read_parquet("data/external_factors/ff_japan_3factors.parquet")
    rets = df[FACTORS].astype(float)
    rets.index = rets.index.to_timestamp()
    print(f"factor panel: {rets.shape[0]} months  {rets.index.min():%Y-%m}..{rets.index.max():%Y-%m}")

    # --- regime シグナル（SJM online・因果） ---
    good = pd.DataFrame({f: online_regime_good(rets[f]) for f in FACTORS})
    eval_idx = good.dropna().index                    # 全因子で regime 確定した評価期間
    rets_e = rets.loc[eval_idx]
    good_e = good.loc[eval_idx]

    # --- 統制シグナル ---
    mom = (rets.rolling(MOM_WIN).sum() > 0).astype(float).loc[eval_idx]     # H-1: trailing-12m>0
    fixed = pd.DataFrame(1.0, index=eval_idx, columns=FACTORS)   # 常時オン=等ウエイト

    # H-18 placebo: regime ラベルをシャッフル（露出割合は保存・timing 破壊）を N シードで分布評価
    def placebo_series(seed):
        rng = np.random.default_rng(seed)
        plac = good_e.copy()
        for f in FACTORS:
            plac[f] = rng.permutation(good_e[f].values)
        return combo_from_signal(rets_e, plac)
    plac_srs = np.array([sharpe(placebo_series(SEED + i)) for i in range(50)])

    series = {
        "fixed_EW(常時オン)":  combo_from_signal(rets_e, fixed),
        "switched_SJM(主)":    combo_from_signal(rets_e, good_e),
        "switched_MOM(H-1統制)": combo_from_signal(rets_e, mom),
        "placebo_shuffle(H-18)": placebo_series(SEED),
    }

    print(f"\n評価期間: {eval_idx.min():%Y-%m}..{eval_idx.max():%Y-%m}  ({len(eval_idx)} months)")
    print("\n== 系列 Sharpe(ann) / 平均月次(bps) ==")
    for name, s in series.items():
        print(f"  {name:24s} SR={sharpe(s):+.3f}  mean={s.mean()*1e4:+6.1f}bps  vol={s.std()*ANN*100:4.1f}%")

    # --- Stage-0 キル①: 遷移数 ---
    print("\n== Stage-0 キル①（H-11 遷移数・full-sample 記述） ==")
    trans = {f: full_sample_transitions(rets[f]) for f in FACTORS}
    for f in FACTORS:
        yrs = rets.shape[0] / 12.0
        print(f"  {f:8s} 遷移 {trans[f]:3d} 回  (~{trans[f]/yrs:.2f}/年)")
    med_tr = int(np.median(list(trans.values())))
    kill1 = med_tr <= 1
    print(f"  → median 遷移 {med_tr} 回  H-11 過適合フラグ={'YES(棄却候補)' if kill1 else 'no'}")

    # --- Stage-0 キル②: F5 エッジ + H-18 placebo 同値 + H-14 rho + H-1 ---
    print("\n== Stage-0 キル②（F5 エッジ / H-18 placebo / H-14 rho / H-1 mom） ==")
    sw, fx = series["switched_SJM(主)"], series["fixed_EW(常時オン)"]
    rho = float(sw.corr(fx))
    dev_frac = float((good_e.shift(1).fillna(1.0) != fixed).any(axis=1).mean())   # 配分が固定と乖離する月の割合
    sw_sr = sharpe(sw); fx_sr = sharpe(fx)
    mom_sr = sharpe(series["switched_MOM(H-1統制)"])
    edge = sw_sr - fx_sr
    plac_mean, plac_p95 = float(plac_srs.mean()), float(np.percentile(plac_srs, 95))
    print(f"  rho(switched, fixed) = {rho:+.3f}   乖離月割合 = {dev_frac*100:.0f}%")
    print(f"  SR switched {sw_sr:+.3f} vs fixed {fx_sr:+.3f}   エッジΔSR(vs常時オン) = {edge:+.3f}  [F5]")
    print(f"  placebo(shuffle) SR: mean {plac_mean:+.3f} / p95 {plac_p95:+.3f}  (switched が p95 内 = timing 情報なし)  [H-18]")
    print(f"  switched {sw_sr:+.3f} vs MOM統制 {mom_sr:+.3f}  ΔSR = {sw_sr - mom_sr:+.3f}  (負=素朴momに劣後)  [H-1]")
    f5   = edge <= 0.0                       # 常時オン未満＝de-risk が有害
    h18  = sw_sr <= plac_p95                 # シャッフル placebo と区別不能＝timing 情報ゼロ
    h14  = (rho > 0.9) and (edge <= 0.0)     # 固定に張り付き＋エッジ無し
    print(f"  → F5={'YES' if f5 else 'no'}  H-18(placebo同値)={'YES' if h18 else 'no'}  "
          f"H-14(rho張り付き)={'YES' if h14 else 'no'}")

    print("\n== Stage-0 総合判定 ==")
    kill = kill1 or f5 or h18 or h14
    if kill:
        reasons = [t for t, c in [("H-11 過適合", kill1), ("F5 常時オン未満", f5),
                                  ("H-18 placebo 同値", h18), ("H-14 rho 張り付き", h14)] if c]
        print(f"  KILL（K=0 で Stage-0 棄却）: {' / '.join(reasons)}。グリッド不要＝documented negative。")
    else:
        print("  SURVIVE: 正式 judge_grid（K<=6）へ進む価値あり。")


if __name__ == "__main__":
    main()
