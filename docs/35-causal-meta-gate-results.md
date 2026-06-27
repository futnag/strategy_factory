# 35 — 因果メタゲート正式判定結果

> scope=`value_pead_causal_meta_gate`  事前登録: docs/34

**K=3**  DSR閾値=0.95

# 判定レポート: value_pead_causal_meta_gate
- 仮説: value+PEADのOOS失速は因果構造状態に依存し、セクター因果エッジ特徴量を加えたメタモデルで改善する
- 試行数 K（この scope の累計）= **3**, 試行間SR分散 V[SR]=0.0067
- 判定基準: DSR ≥ 0.95
- 補助診断（表示専用・DP18）: PBO(CSCV)=0.09（IS最良がOOSで中央値以下になる確率・ノイズ≈0.5）, MinBTL(K=3, SR_ann=1)≈0.7年（これより短い標本ではノイズの最良が年率SR 1 を超えうる）

| strategy | SR(ann) | PSR(>0) | **DSR** | 頑健 | minTRL(月) | 回転 | maxDD | 容量 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| value+pead_lt | +0.31 | 0.84 | **0.59** | 0.46 | 327 | 0.58 | -19.4% | ¥8億 |
| value+pead_lt|causal_meta | +0.08 | 0.60 | **0.31** | 0.47 | 5120 | 0.51 | -19.4% | ¥8億 |
| value+pead_lt|base_meta | -0.26 | 0.21 | **0.06** | 0.41 | ∞ | 0.36 | -17.3% | ¥8億 |

- 頑健＝補助頑健性スコア（表示専用・judge.robustness_score）。**判定は DSR のみ**（DP18）。
## 判定: ❌ FAIL — 最良 value+pead_lt でも DSR=0.59 < 0.95
- 最良の内訳: SR(ann)=+0.31, PSR(>0)=0.84, minTRL=327か月
- サブ期間: 2016-07..2019-10:-0.42  2019-11..2023-01:+0.53  2023-02..2026-04:+0.60
- K=3 試行に対しデフレート済み。**試行を増やすほど基準は上がる**（＝判定器自体のp-hack不能）。

HTML: `data\reports\value_pead_causal_meta_gate.html`