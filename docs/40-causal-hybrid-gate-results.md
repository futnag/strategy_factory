# 40 — ハイブリッド因果メタゲート判定結果

> scope=`value_pead_causal_hybrid_gate`  事前登録: docs/39
> adverse月: 19  縮小適用: 8

**K=4**

# 判定レポート: value_pead_causal_hybrid_gate
- 仮説: base_metaを維持し因果は不利レジーム時のみ0.5x縮小すればOOS改善と全期間DSRを両立する
- 試行数 K（この scope の累計）= **4**, 試行間SR分散 V[SR]=0.0060
- 判定基準: DSR ≥ 0.95
- 補助診断（表示専用・DP18）: PBO(CSCV)=0.00（IS最良がOOSで中央値以下になる確率・ノイズ≈0.5）, MinBTL(K=4, SR_ann=1)≈1.1年（これより短い標本ではノイズの最良が年率SR 1 を超えうる）

| strategy | SR(ann) | PSR(>0) | **DSR** | 頑健 | minTRL(月) | 回転 | maxDD | 容量 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| value+pead_lt | +0.31 | 0.84 | **0.53** | 0.46 | 327 | 0.58 | -19.4% | ¥8億 |
| value+pead_lt|causal_all33 | -0.15 | 0.33 | **0.09** | 0.40 | ∞ | 0.47 | -17.3% | ¥8億 |
| value+pead_lt|base_meta | -0.26 | 0.21 | **0.05** | 0.41 | ∞ | 0.36 | -17.3% | ¥8億 |
| value+pead_lt|hybrid_all33 | -0.26 | 0.21 | **0.05** | 0.41 | ∞ | 0.36 | -17.3% | ¥8億 |

- 頑健＝補助頑健性スコア（表示専用・judge.robustness_score）。**判定は DSR のみ**（DP18）。
## 判定: ❌ FAIL — 最良 value+pead_lt でも DSR=0.53 < 0.95
- 最良の内訳: SR(ann)=+0.31, PSR(>0)=0.84, minTRL=327か月
- サブ期間: 2016-07..2019-10:-0.42  2019-11..2023-01:+0.53  2023-02..2026-04:+0.60
- K=4 試行に対しデフレート済み。**試行を増やすほど基準は上がる**（＝判定器自体のp-hack不能）。

HTML: `data\reports\value_pead_causal_hybrid_gate.html`