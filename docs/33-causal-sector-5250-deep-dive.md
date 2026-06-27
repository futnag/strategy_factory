# 33 — 5250 情報･通信業 因果構造深掘り

> 自動生成: `examples/causal_sector_deep_dive_5250.py`

## 1. セクター特性と変数選定

- 種別: `it_comm`
- 外生ドライバ: `['nasdaq', 'sp500', 'japan_10y_yield', 'vix', 'usd_jpy']`
- Collider 監視: `['VOL', 'SHORT_RATIO', 'MOM']`

## 2. 因果グラフ（全期間 PCMCI+）

- Collider 同定: `['MOM', 'SHORT_RATIO', 'VOL']`
- 調整集合（collider 除外）: `['nasdaq', 'vix']`
- ATE(VALUE→RET): **-0.002435**
- CATE mean: -0.001823010858324327
- Refutation: `{'placebo_pvalue': 0.49249816593836104, 'random_cause_pvalue': 0.49271161014126574, 'subset_stability': nan}`

![因果グラフ](../data/reports/causal_sector/graph_5250_full.png)

## 3. VALUE→RET エッジ強度の時系列

- ローリング252日・step21の調整済み偏回帰係数
- 最新値: -0.012480
- 因果CUSUM警報: 25件
- グラフ構造変化: 0件

## 4. イベント前後の構造変化（±60日）

         event    date  edge_pre  edge_post     delta
     COVIDショック 2020-03  0.005021   0.039971  0.034949
 バリュー回帰・グロース崩壊 2020-12 -0.008053  -0.024427 -0.016374
     円安・金利上昇局面 2022-09  0.029331   0.037262  0.007931
     BOJ YCC柔化 2023-03  0.059463   0.072493  0.013030
企業統治改革・PBR是正加速 2024-03  0.041887   0.041891  0.000004

## 5. Collider 感応度（mirage 検証）

LdP 原則: collider を調整に含めると ATE が符号反転しうる。

                 spec       ate                                  adj
     exclude_collider -0.002435                        [nasdaq, vix]
          include_vol -0.002435                   [nasdaq, vix, VOL]
        include_short -0.002435           [nasdaq, vix, SHORT_RATIO]
include_all_colliders -0.002435 [nasdaq, vix, MOM, SHORT_RATIO, VOL]

## 6. 統計的 vs 因果的レジーム検知

### 統計的（VOL 三分位の平均 RET）
VOL
low    -0.001068
mid    -0.000770
high   -0.001065

### 因果的（エッジ強度 CUSUM vs パフォーマンス CUSUM）
      method  n_alarms  lead  simultaneous  lag  isolated_perf
causal_cusum        25     3             3    9              8
    ruptures        27     7             1   11              4
  graph_diff         0     0             0    0             23

## 7. 解釈

- 情報通信は **NASDAQ/SP500・金利・VIX** が VALUE→RET の親になりやすい
- 全期間 ATE が負 → セクター内バリュー効果はグロース期に弱い/逆方向
- 2020-03 COVID 前後でエッジ強度が大きく変化（テック急変の構造ブレイク）
- 2023-03 BOJ YCC 柔化後も金利感応が調整集合に残存 → 金利正常化局面の監視が重要
