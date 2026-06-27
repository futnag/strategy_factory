# OOS特化ハイブリッド因果メタゲート 事前登録（FROZEN）

**事前登録日**: 2026-06-23
**ステータス**: FROZEN
**関連**: docs/34/37/38（純因果メタゲートは OOS↑・IS↓）、docs/25（base_meta）
**判定 scope**: `value_pead_causal_hybrid_gate`

---

## 0. 仮説

> docs/37/38 の純因果メタゲート（causal_all33）は OOS SR を改善するが、
> walk-forward 学習が IS 期間で過剰縮小し全期間 DSR を毀損した。
> **ハイブリッド**＝一次サイズは base_meta のまま、因果特徴量は
> 「不利レジームの downside 保護」にのみ使えば、
> **OOS SR 改善を維持しつつ全期間 DSR を ungated 以上に保てる**。

**経済的合理性**: 因果シグナルは常にサイズ決定に使うべきではない（LdP: 監視≠毎月売買）。
IT/金融で VALUE→RET が構造的に負（docs/36）かつ横断エッジも負かつ
多数セクターが不安定な「三重不利」の月だけ露出を半減すれば、
value+PEAD 脚衝突の損失を削りつつ、有利月は base_meta に委ねる。

## 1. 一次モデル（FROZEN）

docs/25/37 と同一。変更なし。

## 2. ハイブリッドゲート（FROZEN）

`invest_system/research/causal_sector/hybrid_gate.py::fit_hybrid_causal_gate`

### 2.1 ベース層
`fit_meta_gate(primary_net, base_feat_5)` — docs/25 凍結5特徴、warmup=36, embargo=1

### 2.2 因果オーバーレイ（ルールベース・ML なし）
当月 `size_hybrid = size_base`、ただし **adverse 月**のみ:
`size_hybrid = size_base × 0.5`

**adverse 定義（3条件 AND・凍結）:**
1. `causal_edge < 0`
2. `edge_financial < 0`
3. `n_sectors_unstable >= 5`

因果特徴量は `build_causal_meta_features_all33` の8本から上記3列のみ使用。
いずれか欠損 → adverse=False（縮小しない）。

### 2.3 パラメータ（凍結）
- `shrink_mult = 0.5`
- `unstable_thresh = 5`

## 3. 比較戦略（judge_grid・K=4）

| name | 説明 |
|---|---|
| `value+pead_lt` | ungated |
| `value+pead_lt\|base_meta` | docs/25 |
| `value+pead_lt\|causal_all33` | docs/37 純因果 |
| `value+pead_lt\|hybrid_all33` | 本ハイブリッド |

## 4. 判定基準

- OOS = 2024-01〜
- scope=`value_pead_causal_hybrid_gate`

**PASS（AND）** — `hybrid_all33`:
1. OOS SR ≥ ungated OOS SR
2. 全期間 DSR ≥ ungated DSR
3. 全期間 DSR ≥ 0.95

**副次（報告のみ）:**
- 全期間 SR ≥ causal_all33 全期間 SR（IS 毀損の回復）
- OOS SR ≥ causal_all33 OOS SR
- adverse 月の ungated SR < 非adverse 月（分離診断）

## 5. 禁止事項

結果に基づく閾値・shrink_mult・条件の事後変更禁止。