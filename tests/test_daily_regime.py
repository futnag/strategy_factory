"""daily_regime step 0–1 の受け入れテスト（DoD）。

完了定義：
  (A) leak_tests が**意図通り**赤/緑（クリーンは緑、意図的リーク注入は赤）。
  (B) 2契約スキーマが**実データで充足**。
  (C) 橋渡し層の**実行時アサーション**が通過（かつ違反で落ちる）。

釘①：リーク版で『赤』になることを同テスト内で必ず確認する（何も捕まえない緑は偽の安心）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.tseries.offsets import BDay

from invest_system.research.daily_regime import asof_bridge, leak_tests, tiers, weekly_anchor
from invest_system.research.daily_regime import data as drdata
from invest_system.research.daily_regime.contracts import (
    ContractViolation,
    LookAheadError,
    validate_anchor,
    validate_tiers,
)
from invest_system.research.sector_regime.data_loader import get_data_root


# ---------------------------------------------------------------- helpers
def _valid_anchor() -> pd.DataFrame:
    fridays = pd.date_range("2024-01-05", periods=4, freq="W-FRI")
    return validate_anchor(pd.DataFrame({
        "week_end_date": fridays,
        "available_from": fridays + BDay(1),
        "anchor_id": "0050",
        "p_bear": [0.1, 0.2, 0.7, 0.3],
        "p_neutral": [0.2, 0.3, 0.2, 0.4],
        "p_bull": [0.7, 0.5, 0.1, 0.3],
        "regime": ["bull", "bull", "bear", "neutral"],
        "anchor_source": ["sector", "sector", "sector", "market"],
        "n_constituents": [50, 50, 50, 3],
    }))


def _ar1(n: int, phi: float, seed: int, scale: float = 1.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, scale, n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(x, index=idx)


# ====================================================== (A) null 生成器の正しさ
def test_block_shuffle_preserves_autocorr_better_than_iid():
    x = _ar1(2400, phi=0.6, seed=1)
    a0 = x.autocorr(1)
    blk = leak_tests.block_shuffle(x, block=40, seed=7).autocorr(1)
    rng = np.random.default_rng(7)
    iid = pd.Series(rng.permutation(x.to_numpy()), index=x.index).autocorr(1)
    assert abs(blk - a0) < abs(iid - a0)         # ブロックは自己相関を残す
    assert abs(iid) < 0.1                          # IID は壊す


def test_phase_randomize_preserves_acf():
    x = _ar1(4000, phi=0.6, seed=2)
    sur = leak_tests.phase_randomize(x, seed=3)
    assert abs(sur.autocorr(1) - x.autocorr(1)) < 0.2


# ====================================================== (A) プラセボがリークを捕まえる
def test_placebo_flags_future_leak_not_noise():
    r = _ar1(1500, phi=0.3, seed=10, scale=0.01)
    leak = leak_tests.leaky_future_signal(r)                       # 完全な先読み
    p_leak = leak_tests.placebo_pvalue(leak, r, null="block", block=20, n=200, seed=0)
    assert p_leak < 0.05                          # 赤：エッジが null で説明されない＝本物（=リーク）

    # 校正：独立ノイズ信号の p は帰無下で一様 → 複数シードの平均は 0.5 付近（過剰検出しない）。
    # 単一ドローの p は 5% で偶発的に小さくなるため、平均で校正を確認する（非フレーキー）。
    ps = [
        leak_tests.placebo_pvalue(
            _ar1(1500, phi=0.3, seed=100 + k, scale=0.01).reindex(r.index),
            r, null="block", block=20, n=100, seed=k,
        )
        for k in range(10)
    ]
    assert float(np.mean(ps)) > 0.3               # 緑：ノイズは null と区別できない（校正済み）


# ====================================================== (C) アンカー日足ラグ：緑/赤
def test_anchor_daily_pit_green_and_red():
    anchor = _valid_anchor()
    didx = pd.bdate_range("2024-01-01", "2024-02-15")
    canon = asof_bridge.anchor_daily_panel(anchor, didx, "p_bull", "0050")
    asof_bridge.assert_anchor_daily_pit(anchor, canon, "p_bull", "0050")   # 緑

    leaky = leak_tests.leaky_anchor_daily_panel(anchor, didx, "p_bull", "0050")
    with pytest.raises(LookAheadError):                                     # 赤
        asof_bridge.assert_anchor_daily_pit(anchor, leaky, "p_bull", "0050")


# ====================================================== (C) ティア遡及再付番：緑/赤
def test_tier_pool_index_pit_green_and_red():
    dates = pd.bdate_range("2024-01-01", periods=6)
    mem = pd.DataFrame({
        "A": ["T1", "T1", "T1", "T2", "T2", "T2"],   # 途中でティア変化
        "B": ["T1", "T1", "T1", "T1", "T1", "T1"],
    }, index=dates)
    asof = dates[-1]

    idx_t1 = tiers.tier_pool_index(mem, "T1", asof)
    tiers.assert_pool_index_pit(mem, "T1", asof, idx_t1)        # 緑
    idx_t2 = tiers.tier_pool_index(mem, "T2", asof)
    tiers.assert_pool_index_pit(mem, "T2", asof, idx_t2)        # 緑

    leaky = leak_tests.leaky_pool_index(mem, "T2", asof)        # A の as-of=T2 を過去へ遡及
    with pytest.raises(LookAheadError):                         # 赤
        tiers.assert_pool_index_pit(mem, "T2", asof, leaky)


# ============================================ ティア帰属修正：平滑＋スティッキー＋スパイク頑健
def test_tier_smoothed_sticky_and_spike_robust():
    # 20銘柄：売買代金が code 順に増加（流動性順位は安定）。C19 が最流動（=T1）。
    dates = pd.bdate_range("2022-01-01", periods=400)
    n = 20
    rng = np.random.default_rng(0)
    ret = rng.normal(0.0, 0.01, (400, n))
    codes = [f"C{i:02d}" for i in range(n)]
    turnover = pd.DataFrame(
        np.tile((np.arange(n) + 1) * 1e8, (400, 1)), index=dates, columns=codes
    ).astype(float)                                   # 一定・code 順に増加（全て ADV 床超）

    # 最流動 C19 に一過性ストレススパイク（巨大 |ret|・極小出来高）を refit 直前窓へ注入
    spike = [175, 176, 177]
    ret[spike, n - 1] = 0.30
    adj_close = pd.DataFrame(100.0 * np.exp(np.cumsum(ret, axis=0)), index=dates, columns=codes)
    turnover.iloc[spike, n - 1] = 1e6

    tlong = tiers.assign_tiers(
        adj_close, turnover, n_tiers=3, adv_floor=5e7, smooth_window=60, refit_every=20,
    )
    mem = tiers.membership_panel(tlong)

    # (1) 平滑キー＝中央値はスパイクを無視 → C19 は最流動 T1 のまま（誤再ランクしない）
    assert mem.loc[dates[177], "C19"] == "T1"
    assert mem.loc[dates[180], "C19"] == "T1"          # スパイク窓を含む refit 後も T1

    # (2) 対比：当日生 Amihud だと C19 はスパイク日に最非流動ティアへ誤再ランクされる
    raw = tiers._daily_amihud(adj_close, turnover).loc[dates[177]]
    naive_tier = "T1" if raw.rank(pct=True).loc["C19"] <= 1 / 3 else (
        "T2" if raw.rank(pct=True).loc["C19"] <= 2 / 3 else "T3")
    assert naive_tier == "T3"                          # 生値なら誤って非流動扱い（修正が回避する害）

    # (3) スティッキー：refit ブロック内（180→199）でティアは不変
    block = mem.loc[dates[180]:dates[199], "C19"].dropna()
    assert block.nunique() == 1


# ============================================ ② per-stock Student-t HMM：合成回復
def test_t_hmm_recovers_vol_regime_online():
    from invest_system.research.daily_regime import detector as drdet
    rng = np.random.default_rng(0)
    n = 600
    s, states = 0, []
    for _ in range(n):                                   # 持続的な2レジーム（低/高ボラ）
        if rng.random() < 0.02:
            s = 1 - s
        states.append(s)
    states = np.array(states)
    r = rng.normal(0.0, np.where(states == 0, 0.005, 0.02))
    idx = pd.bdate_range("2015-01-01", periods=n)
    rv = pd.Series(r, index=idx).rolling(10).std().bfill()
    absr = pd.Series(np.abs(r), index=idx).rolling(5).mean().bfill()
    feat = pd.DataFrame({"rv": rv, "absr": absr})

    out = drdet.walk_forward_t_hmm(
        feat, n_states=2, refit_every=30, window=300, warmup=150,
        n_init=2, align_col=0, min_dwell=3, random_state=0,
    )
    m = out["prob_stressed"].notna()
    truth = pd.Series(states, index=idx)[m]
    ps = out["prob_stressed"][m]
    # 整列後 stressed=高ボラ → 真の高ボラ区間で filtered の stressed 確率が明確に高い
    assert ps[truth == 1].mean() > ps[truth == 0].mean() + 0.2
    # state_filtered は離散ラベルで NaN を含まない区間がある（オンライン稼働）
    assert out["state_filtered"].notna().sum() > 100


# ============================================ step 3：方式A 注入の機構＋PIT
def test_method_a_injection_mechanism_and_identity():
    from invest_system.research.daily_regime.detector import _standardize_past, fit_best
    rng = np.random.default_rng(0)
    n = 400
    s, states = 0, []
    for _ in range(n):
        if rng.random() < 0.02:
            s = 1 - s
        states.append(s)
    states = np.array(states)
    r = rng.normal(0.0, np.where(states == 0, 0.005, 0.02))
    idx = pd.bdate_range("2016-01-01", periods=n)
    rv = pd.Series(r, index=idx).rolling(10).std().bfill()
    absr = pd.Series(np.abs(r), index=idx).rolling(5).mean().bfill()
    X = _standardize_past(np.column_stack([rv.values, absr.values]))
    m = fit_best(X, n_states=2, dof=4.0, sticky_kappa=10.0, n_init=2, random_state=0)
    m.align_by(m.means_[:, 0])
    base = m.filtered_proba(X)
    assert np.allclose(m.filtered_proba_injected(X, np.zeros((n, 2))), base, atol=1e-9)  # lam=0 恒等
    tilt = np.zeros((n, 2)); tilt[150:250, 1] = 2.0; tilt[150:250, 0] = -2.0              # bearish 注入
    inj = m.filtered_proba_injected(X, tilt)
    assert inj[150:250, 1].mean() > base[150:250, 1].mean() + 0.05                        # stressed↑


def test_weekly_log_tilt_is_pit_before_available_from():
    from invest_system.research.daily_regime import multiscale
    fridays = pd.date_range("2024-01-05", periods=3, freq="W-FRI")
    anchor = validate_anchor(pd.DataFrame({
        "week_end_date": fridays, "available_from": fridays + BDay(1), "anchor_id": "S",
        "p_bear": [0.7, 0.7, 0.7], "p_neutral": [0.2, 0.2, 0.2], "p_bull": [0.1, 0.1, 0.1],
        "regime": "bear", "anchor_source": "sector", "n_constituents": 50,
    }))
    didx = pd.bdate_range("2024-01-01", "2024-02-01")
    tilt = multiscale.weekly_log_tilt(anchor, didx, "S", n_states=2, lam=1.0)
    before = didx < (fridays[0] + BDay(1))
    assert np.allclose(tilt[before], 0.0)            # available_from 前は注入ゼロ（PIT）
    assert tilt[~before, 1].max() > 0.0              # 以降は bear→stressed 事前↑


# ============================================ step 5：本物の週足方向性HMM の PIT（意図的リーク→赤）
def test_real_weekly_anchor_pit_and_leak():
    from invest_system.research.daily_regime import asof_bridge, leak_tests, weekly_anchor
    idx = pd.bdate_range("2018-01-01", periods=600)
    rng = np.random.default_rng(0)
    codes = [f"C{i:02d}" for i in range(20)]
    adj = pd.DataFrame(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, (600, 20)), 0)),
                       index=idx, columns=codes)
    s33 = pd.Series(["1000"] * 10 + ["2000"] * 10, index=codes)
    anchor = weekly_anchor.build_weekly_anchor_hmm(
        adj, s33, n_states=2, warmup=52, window=104, refit_every=8, n_init=2)
    validate_anchor(anchor)                                  # スキーマ＋ available_from>week_end_date
    assert len(anchor) > 0
    sec = anchor["anchor_id"].iloc[0]
    didx = pd.bdate_range(idx[0], idx[-1])
    canon = asof_bridge.anchor_daily_panel(anchor, didx, "p_bear", sec)
    asof_bridge.assert_anchor_daily_pit(anchor, canon, "p_bear", sec)        # 緑：available_from ラグ
    leaky = leak_tests.leaky_anchor_daily_panel(anchor, didx, "p_bear", sec)
    with pytest.raises(LookAheadError):                                      # 赤：week_end_date 漏れ
        asof_bridge.assert_anchor_daily_pit(anchor, leaky, "p_bear", sec)


# ============================================ PEAD 並行：決算窓フラグ（発表後・PIT）
def test_earnings_window_flag_post_announcement():
    from invest_system.research.daily_regime import event_mask
    didx = pd.bdate_range("2024-01-01", periods=20)
    dd = pd.DataFrame({"Code": ["X"], "DiscDate": [didx[5]]})
    flag = event_mask.earnings_window_flag(dd, didx, codes=["X"], window=3)
    assert not bool(flag["X"].iloc[5])                                # 発表当日は窓外（跨がない）
    assert bool(flag["X"].iloc[6]) and bool(flag["X"].iloc[7]) and bool(flag["X"].iloc[8])  # 翌3営業日
    assert not bool(flag["X"].iloc[9])                               # 窓終了
    assert not bool(flag["X"].iloc[4])                               # 発表前 False


# ============================================ (a)ループ：ファンダ PIT（期末日アラインで赤・恒久）
def test_fundamental_pit_disclosure_vs_period_end_leak():
    from invest_system.research.daily_regime import fundamentals_pit as fp
    fund = pd.DataFrame({"Code": ["X"], "DiscDate": [pd.Timestamp("2024-06-20")],
                         "CurFYEn": [pd.Timestamp("2024-03-31")], "EPS": [100.0]})   # 期末3月・開示6月
    rebal = pd.bdate_range("2024-04-01", "2024-08-01", freq="W-FRI")
    pit = fp.asof_field(fund, rebal, "EPS", lag_days=1)              # DiscDate アンカー
    leak = fp.leaky_period_end_asof(fund, rebal, "EPS")             # 期末日アンカー（リーク）
    pre = rebal[rebal < pd.Timestamp("2024-06-20")][-1]            # 開示前
    post = rebal[rebal > pd.Timestamp("2024-06-21")][0]
    assert pd.isna(pit["X"].get(pre))                              # PIT：開示前は見えない
    assert pit["X"].get(post) == 100.0                             # PIT：開示後に見える
    assert leak["X"].get(pre) == 100.0                             # リーク：期末後すぐ＝未公表をトレード
    fp.assert_fundamental_pit(pit, fund, "EPS", lag_days=1)        # 緑
    with pytest.raises(LookAheadError):                           # 赤：期末日アラインは開示前に値が出る
        fp.assert_fundamental_pit(leak, fund, "EPS", lag_days=1)


# ============================================ step 4：§2 PEAD 適用除外（両側）＋レジーム依存コスト
def test_pead_exemption_both_sided():
    from invest_system.research.data_view import AsOf
    from invest_system.research.daily_regime.strategy import ConstantLong, RegimeAwareStrategy
    idx = pd.bdate_range("2024-01-01", periods=5)
    code = "X"
    close = pd.DataFrame({code: [100.0, 101, 102, 103, 104]}, index=idx)
    stressed = pd.DataFrame({code: [0.9] * 5}, index=idx)             # 高ストレス
    earn_in = pd.DataFrame({code: [True] * 5}, index=idx)
    earn_out = pd.DataFrame({code: [False] * 5}, index=idx)
    t = idx[-1]
    pead = RegimeAwareStrategy(ConstantLong(code), event_driven=True, filter_thresh=0.6, size_floor=0.2)

    asof_in = AsOf({"close": close.loc[:t], "prob_stressed": stressed.loc[:t],
                    "earn_window": earn_in.loc[:t]}, t)
    w_in = pead.target_weights(asof_in)
    assert code in w_in.index and w_in[code] > 0.0          # (a) 決算窓でフィルタが新規を止めていない
    assert abs(w_in[code] - 0.2) < 1e-9                      # (b) 同窓でサイジングがレジーム値で縮小

    asof_out = AsOf({"close": close.loc[:t], "prob_stressed": stressed.loc[:t],
                     "earn_window": earn_out.loc[:t]}, t)
    assert pead.target_weights(asof_out).get(code, 0.0) == 0.0   # 窓外＋高ストレス→フィルタ適用（ゼロ）

    value = RegimeAwareStrategy(ConstantLong(code), event_driven=False, filter_thresh=0.6)
    assert value.target_weights(asof_in).get(code, 0.0) == 0.0   # 赤の対比：非イベントは窓内でも殺される
    # → pead は通す(>0)／value は殺す(=0)。「除外すべきを除外・残すべきを残す」両立＝§2 が効いている。


def test_regime_cost_is_amihud_dependent_and_conservative():
    from invest_system.research.daily_regime import cost as drcost
    idx = pd.bdate_range("2020-01-01", periods=300)
    a = pd.Series(1.0, index=idx); a.iloc[-30:] = 5.0               # 末尾に低流動スパイク
    panel = drcost.regime_cost_panel(pd.DataFrame({"X": a}), base_bps=10.0, impact_coef=20.0)
    assert abs(panel["X"].iloc[100] - 10.0) < 1e-6                  # 平常は base
    assert panel["X"].iloc[-1] > panel["X"].iloc[100] + 1.0        # 低流動でコスト増（レジーム依存）
    assert (panel["X"] >= 10.0 - 1e-9).all()                        # ≥ base（保守・過小評価しない）


def test_daily_only_equals_multiscale_zero_tilt():
    """ベースライン統制（釘3）：日足のみ ≡ 方式A の週足tilt=0（lam_inject=0）。"""
    from invest_system.research.daily_regime import multiscale, pooling
    rng = np.random.default_rng(1)
    n = 400
    idx = pd.bdate_range("2016-01-01", periods=n)
    s, states = 0, []
    for _ in range(n):
        if rng.random() < 0.03:
            s = 1 - s
        states.append(s)
    states = np.array(states)
    feat_panel = {}
    for i in range(3):
        r = rng.normal(0.0, np.where(states == 0, 0.005, 0.02))
        rv = pd.Series(r, index=idx).rolling(10).std().bfill()
        absr = pd.Series(np.abs(r), index=idx).rolling(5).mean().bfill()
        feat_panel[f"S{i}"] = pd.DataFrame({"rv": rv, "absr": absr})
    mem = pd.DataFrame("T1", index=idx, columns=[f"S{i}" for i in range(3)])
    fridays = pd.date_range(idx[0], idx[-1], freq="W-FRI")
    anchor = validate_anchor(pd.DataFrame({
        "week_end_date": fridays, "available_from": fridays + BDay(1), "anchor_id": "SEC",
        "p_bear": 0.6, "p_neutral": 0.2, "p_bull": 0.2, "regime": "bear",
        "anchor_source": "sector", "n_constituents": 50,
    }))
    kw = dict(n_states=2, refit_every=40, window=250, warmup=150, n_init=2,
              align_col=0, min_dwell=3, random_state=0)
    base = pooling.walk_forward_pooled_t_hmm("S0", feat_panel, mem, **kw)
    ms0 = multiscale.walk_forward_multiscale("S0", feat_panel, mem, anchor, "SEC", lam_inject=0.0, **kw)
    common = base["prob_stressed"].dropna().index.intersection(ms0["prob_stressed"].dropna().index)
    assert len(common) > 50
    assert np.allclose(base["prob_stressed"].loc[common].values,
                       ms0["prob_stressed"].loc[common].values, atol=1e-9)


# ============================================ #1#3 放出安定性ハーネス（枯渇時に pooling 検出）
def test_emission_stability_harness_detects_pooling_when_starved():
    from invest_system.research.daily_regime import pooling_eval as pe
    from invest_system.research.daily_regime.detector import _standardize_past
    rng = np.random.default_rng(0)

    def gen(n):                                          # 2クラスタ真値（重なりあり＝小標本で不安定）
        z = rng.random(n) < 0.5
        return np.where(z[:, None], rng.normal(0.0, 1.0, (n, 2)), rng.normal(2.2, 1.2, (n, 2)))

    n_refits, short, n_pool = 10, 40, 30
    ps_iter = [(i, _standardize_past(gen(short))) for i in range(n_refits)]              # per-stock：枯渇窓
    pl_iter = [(i, _standardize_past(np.vstack([_standardize_past(gen(short)) for _ in range(n_pool)])))
               for i in range(n_refits)]                                                 # pooled：ティア・スタック
    ts_ps = pe.track_emissions(ps_iter, n_states=2, dof=4.0, n_init=2, random_state=1, align_col=0)
    ts_pl = pe.track_emissions(pl_iter, n_states=2, dof=4.0, n_init=2, random_state=1, align_col=0)
    s_ps, s_pl = pe.stability_summary(ts_ps), pe.stability_summary(ts_pl)

    assert s_pl["gap_cv"] < s_ps["gap_cv"]                       # プールで分離 gap が安定
    assert s_pl["gap_cv"] <= pe.GAP_CV_RATIO * s_ps["gap_cv"]    # 事前登録マージン（≥20%低減）
    assert s_pl["degeneracy_rate"] <= s_ps["degeneracy_rate"]    # 退化も悪化しない
    # verdict：枯渇帯（2銘柄想定）で採用、同等同士なら非採用（基準の機械適用）
    assert pe.verdict({"a": s_ps, "b": s_ps}, {"a": s_pl, "b": s_pl})["adopt_pooling"] is True
    assert pe.verdict({"a": s_pl, "b": s_pl}, {"a": s_pl, "b": s_pl})["adopt_pooling"] is False


# ============================================ 保留LL の PIT 釘（窓外定義を1歩間違えると赤）
def test_heldout_block_pit_guard_green_and_red():
    from invest_system.research.daily_regime import pooling_eval as pe
    idx = pd.bdate_range("2020-01-01", periods=10)
    pe.assert_future_block(idx[4], idx[5:])                  # 緑：全て r より後
    with pytest.raises(LookAheadError):                      # 赤：≤r を含む（窓外の定義ミス＝リーク）
        pe.assert_future_block(idx[4], idx[3:])


# ============================================ 保留LL の PIT ゲート：標準化リークが LL を膨張
def test_heldout_ll_standardization_leak_inflates():
    """窓外を**未来込み統計**で z 化すると LL が膨張＝指標はリーク感応（ゆえに ≤r 統計が必須）。"""
    from invest_system.research.daily_regime import pooling_eval as pe
    from invest_system.research.daily_regime.detector import fit_t_mixture
    rng = np.random.default_rng(0)
    fit = np.where(rng.random((300, 2)) < 0.5, rng.normal(0.0, 1.0, (300, 2)), rng.normal(2.5, 1.0, (300, 2)))
    fut = np.where(rng.random((120, 2)) < 0.5, rng.normal(0.0, 2.5, (120, 2)), rng.normal(2.5, 3.0, (120, 2)))  # 未来はボラ拡大
    mu_p, sd_p = fit.mean(0), fit.std(0)                                  # ≤r（PIT）
    alld = np.vstack([fit, fut])
    mu_l, sd_l = alld.mean(0), alld.std(0)                                # 未来込み（LEAK）
    res = fit_t_mixture((fit - mu_p) / sd_p, n_states=2, dof=4.0, n_init=2, random_state=1)
    ll_pit = pe.mixture_loglik(*res, (fut - mu_p) / sd_p, 4.0).mean()
    ll_leak = pe.mixture_loglik(*res, (fut - mu_l) / sd_l, 4.0).mean()
    assert ll_leak > ll_pit + 0.05      # 未来統計でz化＝リークはLLを膨張させる（PIT 窓分割が load-bearing）


# ============================================ ③ pooled walk-forward：スモーク（合成・回復）
def test_pooled_walk_forward_runs_and_recovers():
    from invest_system.research.daily_regime import pooling
    rng = np.random.default_rng(1)
    n = 500
    idx = pd.bdate_range("2016-01-01", periods=n)
    s, states = 0, []
    for _ in range(n):
        if rng.random() < 0.03:
            s = 1 - s
        states.append(s)
    states = np.array(states)
    feat_panel = {}
    for i in range(3):                                   # 同一ティアの3銘柄（共有ボラ構造）
        r = rng.normal(0.0, np.where(states == 0, 0.005, 0.02))
        rv = pd.Series(r, index=idx).rolling(10).std().bfill()
        absr = pd.Series(np.abs(r), index=idx).rolling(5).mean().bfill()
        feat_panel[f"S{i}"] = pd.DataFrame({"rv": rv, "absr": absr})
    membership = pd.DataFrame("T1", index=idx, columns=[f"S{i}" for i in range(3)])

    out = pooling.walk_forward_pooled_t_hmm(
        "S0", feat_panel, membership, n_states=2, refit_every=40, window=250,
        warmup=150, n_init=2, align_col=0, min_dwell=3,
    )
    m = out["prob_stressed"].notna()
    truth = pd.Series(states, index=idx)[m]
    ps = out["prob_stressed"][m]
    assert ps[truth == 1].mean() > ps[truth == 0].mean() + 0.2   # pooled も回復
    assert (out.loc[out["refit"], "pool_size"] >= 2).any()       # プールが成立


# ============================================ multi-tilt 最適化：数値同一性（回帰）
def test_pooled_multi_matches_single_versions():
    from invest_system.research.daily_regime import pooling
    rng = np.random.default_rng(1)
    n = 400
    idx = pd.bdate_range("2016-01-01", periods=n)
    s, states = 0, []
    for _ in range(n):
        if rng.random() < 0.03:
            s = 1 - s
        states.append(s)
    states = np.array(states)
    feat_panel = {}
    for i in range(3):
        r = rng.normal(0.0, np.where(states == 0, 0.005, 0.02))
        rv = pd.Series(r, index=idx).rolling(10).std().bfill()
        absr = pd.Series(np.abs(r), index=idx).rolling(5).mean().bfill()
        feat_panel[f"S{i}"] = pd.DataFrame({"rv": rv, "absr": absr})
    mem = pd.DataFrame("T1", index=idx, columns=[f"S{i}" for i in range(3)])
    tilt = np.zeros((n, 2)); tilt[100:200, 1] = 1.5; tilt[100:200, 0] = -1.5
    kw = dict(n_states=2, refit_every=40, window=250, warmup=150, n_init=2,
              align_col=0, min_dwell=3, random_state=0)
    single0 = pooling.walk_forward_pooled_t_hmm("S0", feat_panel, mem, **kw)
    singleT = pooling.walk_forward_pooled_t_hmm("S0", feat_panel, mem, log_tilt=tilt, **kw)
    multi = pooling.walk_forward_pooled_multi("S0", feat_panel, mem, {"z": None, "t": tilt}, **kw)
    for single, name in [(single0, "z"), (singleT, "t")]:
        a = single["prob_stressed"].dropna()
        b = multi[name]["prob_stressed"].dropna()
        common = a.index.intersection(b.index)
        assert len(common) > 50
        assert np.allclose(a.loc[common].values, b.loc[common].values, atol=1e-9)


# ============================================ ① _align_states 一般化（後方互換）
def test_align_states_orders_by_key_and_matches_legacy_top():
    from invest_system.research.sector_regime.detectors import _align_states, _high_vol_state_index
    states = np.array([0, 0, 1, 1, 2, 2])
    key = np.array([0.1, 0.1, 0.9, 0.9, 0.5, 0.5])     # state1 高 / state0 低 / state2 中
    order = _align_states(states, key, ascending=True)
    assert order == [0, 2, 1]                            # calm→stressed（昇順）
    assert order[::-1] == [1, 2, 0]                      # 降順（bull→…）も得られる
    # 非タイの最上位は既存ボラ整列と一致（後方互換）
    assert order[-1] == _high_vol_state_index(states, key)


# ====================================================== (C) AsOf 橋渡し：スライス＆smoothed遮断
def test_asofview_slices_and_rejects_smoothed():
    idx = pd.bdate_range("2024-01-01", periods=4)
    close = pd.DataFrame({"X": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    regime = pd.DataFrame({"X": [0.1, 0.2, 0.3, 0.4]}, index=idx)

    view = asof_bridge.build_regime_view({"close": close}, {"prob_stressed": regime})
    leak_tests.assert_pit_slicing(view, "prob_stressed", idx[2])           # 緑（≤t）
    assert view.asof(idx[2]).frame("prob_stressed").index.max() == idx[2]

    with pytest.raises(LookAheadError):                                    # smoothed 登録禁止
        asof_bridge.build_regime_view({"close": close}, {"state_smoothed": regime})


# ====================================================== 契約バリデータが不正を弾く
def test_contract_validators_reject_violations():
    good = _valid_anchor()
    with pytest.raises(ContractViolation):
        validate_anchor(good.drop(columns=["p_bull"]))            # 列不足
    bad_sum = good.copy(); bad_sum.loc[0, "p_bull"] = 0.9          # Σ≠1
    with pytest.raises(ContractViolation):
        validate_anchor(bad_sum)
    no_lag = good.copy(); no_lag["available_from"] = no_lag["week_end_date"]
    with pytest.raises(ContractViolation):
        validate_anchor(no_lag)                                   # ラグ無し

    dates = pd.bdate_range("2024-01-01", periods=3)
    tlong = pd.DataFrame({
        "date": dates, "code": "A", "tier": ["T1", "T2", "T1"],
        "assigned_through": dates + BDay(1),                      # assigned_through > date
    })
    with pytest.raises(ContractViolation):
        validate_tiers(tlong, n_tiers=3)
    bad_lbl = tlong.copy(); bad_lbl["assigned_through"] = bad_lbl["date"]
    bad_lbl["tier"] = ["T1", "T9", "T1"]                          # 不正ラベル
    with pytest.raises(ContractViolation):
        validate_tiers(bad_lbl, n_tiers=3)


# ====================================================== (B) 2契約スキーマ：実データ充足
def _have_real_data() -> bool:
    try:
        root = get_data_root()
    except Exception:
        return False
    return (root / "processed" / "equities" / "wide" / "close.parquet").exists() \
        and (root / "jquants" / "equities_master.parquet").exists()


@pytest.mark.skipif(not _have_real_data(), reason="実データ（data/）未配置")
def test_contracts_satisfied_on_real_data():
    panels = drdata.load_daily_panels(start="2025-01-01")
    assert not panels.close.empty and not panels.turnover.empty

    # 契約2: TierMembership（Amihud 主・ADV ゲート・平滑＋スティッキー）
    tlong = tiers.assign_tiers(
        panels.adj_close, panels.turnover,
        n_tiers=3, adv_floor=5e7, smooth_window=60, refit_every=20,
    )
    validate_tiers(tlong, n_tiers=3)                              # 例外なし＝充足
    assert len(tlong) > 0
    assert {"T1", "T2", "T3"} & set(tlong["tier"].unique())      # ティアが実在
    assert (pd.to_datetime(tlong["assigned_through"]) <= pd.to_datetime(tlong["date"])).all()
    # スティッキー：変化は refit 境界でのみ＝入替率は [0,1]、変化日数は refit 回数程度
    mem = tiers.membership_panel(tlong)
    to = tiers.refit_turnover(mem)
    assert ((to >= 0.0) & (to <= 1.0)).all()
    assert len(to) <= mem.shape[0] // 20 + 2                      # 毎日は変わらない

    # 契約1: WeeklyDirectionalAnchor（業種指数 S33 代理・スタブ regime）
    s33 = drdata.load_s33_map()
    anchor = weekly_anchor.build_weekly_anchor(panels.adj_close, s33)
    validate_anchor(anchor)                                       # 例外なし＝充足
    assert len(anchor) > 0
    assert set(anchor["anchor_source"].unique()) <= {"sector", "market"}
    # 値幅/stale フラグ・橋渡しが実データで動く
    assert drdata.limit_flag(panels).shape == panels.close.shape


# ---------------------------------------------------------------- 契約：フィールド欠落は例外
def test_strategy_missing_field_raises():
    """フィールド欠落（配線ミス）は silent fallback せず KeyError（値欠損の優雅な劣化と区別）。"""
    from invest_system.research.data_view import AsOf
    from invest_system.research.daily_regime.strategy import (
        ConstantLong,
        RegimeAwareStrategy,
        ValueReversalPrimary,
    )
    idx = pd.bdate_range("2024-01-01", periods=3)
    asof = AsOf({"close": pd.DataFrame({"X": [100.0, 101.0, 102.0]}, index=idx)}, idx[-1])
    with pytest.raises(KeyError):                     # regime_field 不在
        RegimeAwareStrategy(ConstantLong("X"), filter_thresh=0.6).target_weights(asof)
    with pytest.raises(KeyError):                     # eps_field 不在
        ValueReversalPrimary("X").target_weights(asof)
