# 32 — セクター因果的レジーム検知：最終レポート

> 自動生成: `examples/causal_sector_pipeline.py`

## 1. 概要（López de Prado 原則の適用）

- **因果グラフ**: tigramite PCMCI+（ParCorr）でセクター別に学習。
- **Collider 回避**: グラフ上で T→C←Y の変数を調整集合から除外。
- **Structural break**: VALUE→RET エッジ強度のローリング推定 + CUSUM。
  パフォーマンスブレイク（平均リターン変化）との先行性を比較。
- **因果効果**: DoWhy backdoor.linear_regression + EconML LinearDML。
- **Refutation**: placebo treatment, random common cause, data subset。

## 2. セクター別結果

| S33 | 業種 | 種別 | Collider | ATE | CATE | 因果警報数 |
|---|---|---|---|---:|---:|---:|
| 3300 | 石油･石炭製品 | energy_util | SHORT_RATIO, VOL | +0.0007 | 0.0017457419357929929 | 9 |
| 3450 | 鉄鋼 | heavy_industry | MOM, SHORT_RATIO, VOL | +0.0047 | 0.004044942123534515 | 7 |
| 3600 | 機械 | heavy_industry | MOM, SHORT_RATIO, VOL | -0.0005 | 0.0007732326285819979 | 9 |
| 3650 | 電気機器 | it_comm | MOM, SHORT_RATIO, VOL | +0.0047 | 0.0045381125297204545 | 10 |
| 5250 | 情報･通信業 | it_comm | MOM, SHORT_RATIO, VOL | -0.0074 | -0.008184987026864926 | 16 |
| 7050 | 銀行業 | financial | MOM, SHORT_RATIO, VOL | -0.0133 | -0.008350064349164079 | 11 |

## 3. VALUE→RET 構造変化のサマリー

### 3300 石油･石炭製品

- 調整変数: `['SHORT_RATIO', 'japan_10y_yield', 'usd_jpy', 'vix', 'wti_crude']`
- 因果CUSUM: 2019-09-24 00:00:00, 2019-11-26 00:00:00, 2019-12-25 00:00:00, 2020-01-30 00:00:00, 2020-04-02 00:00:00
- グラフ構造変化: 2019-06-11 00:00:00, 2021-06-22 00:00:00
- Refutation: `{'placebo_pvalue': 0.4983571883365596, 'random_cause_pvalue': 0.33416498888526547, 'subset_stability': nan}`

### 3450 鉄鋼

- 調整変数: `['copper', 'us_10y_yield', 'usd_jpy', 'vix']`
- 因果CUSUM: 2019-04-16 00:00:00, 2020-01-30 00:00:00, 2020-03-03 00:00:00, 2020-04-02 00:00:00, 2020-05-07 00:00:00
- グラフ構造変化: 2019-06-13 00:00:00, 2020-04-24 00:00:00, 2021-06-24 00:00:00, 2023-06-22 00:00:00, 2024-11-18 00:00:00
- Refutation: `{'placebo_pvalue': 0.4051406465791779, 'random_cause_pvalue': 0.4626840924850608, 'subset_stability': nan}`

### 3600 機械

- 調整変数: `['us_10y_yield', 'usd_jpy', 'vix']`
- 因果CUSUM: 2019-04-16 00:00:00, 2019-08-22 00:00:00, 2019-09-24 00:00:00, 2019-10-25 00:00:00, 2019-11-26 00:00:00
- グラフ構造変化: 2020-01-08 00:00:00, 2020-04-24 00:00:00, 2021-06-24 00:00:00, 2022-11-29 00:00:00, 2025-06-23 00:00:00
- Refutation: `{'placebo_pvalue': 0.49037107433953464, 'random_cause_pvalue': 0.44255611559840324, 'subset_stability': nan}`

### 3650 電気機器

- 調整変数: `['VOL', 'sp500', 'usd_jpy', 'vix']`
- 因果CUSUM: 2019-04-16 00:00:00, 2019-10-25 00:00:00, 2020-10-12 00:00:00, 2021-08-20 00:00:00, 2023-02-06 00:00:00
- グラフ構造変化: 2019-09-17 00:00:00, 2020-04-20 00:00:00, 2021-06-18 00:00:00, 2021-09-29 00:00:00, 2022-01-13 00:00:00
- Refutation: `{'placebo_pvalue': 0.49740556159719057, 'random_cause_pvalue': 0.42400475567651263, 'subset_stability': nan}`

### 5250 情報･通信業

- 調整変数: `['japan_10y_yield', 'nasdaq', 'vix']`
- 因果CUSUM: 2019-04-16 00:00:00, 2019-05-23 00:00:00, 2019-08-22 00:00:00, 2020-01-30 00:00:00, 2020-04-02 00:00:00
- グラフ構造変化: 2020-04-20 00:00:00, 2021-09-29 00:00:00
- Refutation: `{'placebo_pvalue': 0.4961182251732601, 'random_cause_pvalue': 0.4911022024150875, 'subset_stability': nan}`

### 7050 銀行業

- 調整変数: `['SHORT_RATIO', 'japan_10y_yield', 'us_10y_yield', 'vix']`
- 因果CUSUM: 2019-05-23 00:00:00, 2019-08-22 00:00:00, 2019-11-26 00:00:00, 2019-12-25 00:00:00, 2020-06-05 00:00:00
- グラフ構造変化: 2020-04-20 00:00:00, 2020-11-17 00:00:00, 2021-03-11 00:00:00, 2021-09-29 00:00:00, 2023-09-25 00:00:00
- Refutation: `{'placebo_pvalue': 0.4944714986490174, 'random_cause_pvalue': 0.47480208092554665, 'subset_stability': nan}`

## 4. 日本株イベントとの照合

| イベント | 関連セクターでの因果警報 |
|---|---|
| 2020-03 COVIDショック | 3300, 3450, 3600, 3650, 5250, 7050 |
| 2020-12 バリュー回帰・グロース崩壊 | 3650, 7050 |
| 2022-09 円安・金利上昇局面 | 3600, 3650 |
| 2023-03 BOJ YCC柔化 | 3650, 7050 |
| 2024-03 企業統治改革・PBR是正加速 | — |

## 5. メタラベリング統合（実行済み）

`examples/causal_sector_meta_gate.py` で value+PEAD 一次戦略に
因果特徴量を追加した二次モデルを walk-forward 学習。

| モデル | 全期間 SR(ann) | OOS SR(ann) | OOS DSR |
|---|---:|---:|---:|
| ungated | +0.31 | +0.51 | 0.78 |
| base_meta（凍結5特徴） | -0.25 | +0.46 | 0.75 |
| **causal_meta** | +0.08 | **+0.85** | **0.90** |

- 因果エッジ符号レジームで明確な分離：edge+ で SR=0.83、edge弱で SR=-0.61
- CPCV mean SR: base=0.08 → causal=0.12（DSR 0.36→0.53）
- 判定：OOS SR・DSR は因果版が ungated/base を上回る（OOS DSR 0.95 は未達）

詳細: `data/reports/causal_sector/meta_gate_results.md`

## 6. 限界と注意点

- VALUE は月次→日次 ffill のため、日次 PCMCI の VALUE エッジは解釈に注意。
- PCMCI+ は線形 ParCorr 前提。非線形・レジーム依存の因果は underfit しうる。
- 全33業種の同時実行は計算コスト大（ローリング再学習がボトルネック）。
- DoWhy refutation の p-value は標本依存。過度な最適化は避けること。

## 7. 次のステップ（ライブ運用向け）

1. 月次リバランス前に代表セクターの VALUE→RET エッジ強度を更新
2. 因果CUSUM 警報時にメタゲートサイズを自動縮小
3. Collider 出現（新エッジ）をアラートとして監視
4. BOJ政策・PBR改革イベント後のグラフ再学習を四半期ごとに実施

可視化: `C:\Users\futos\claude_local_sessions\data\reports\causal_sector/graph_*.png`
