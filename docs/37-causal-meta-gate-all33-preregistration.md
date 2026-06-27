# 全33業種因果メタゲート 事前登録（FROZEN）

**事前登録日**: 2026-06-23
**ステータス**: FROZEN
**関連**: docs/34（代表6業種版）、docs/36（全業種バッチATE）
**判定 scope**: `value_pead_causal_meta_gate_all33`

---

## 0. 仮説（a priori）

> docs/34 の代表6業種因果特徴量は OOS SR を改善したが全期間 DSR は未改善だった。
> **全33業種**に拡張し、業種種別（重工業/IT/金融）のエッジ強度を明示的に入れることで、
> 因果メタゲートの **OOS SR および全期間 DSR** が代表6版・ungated を上回る。

**経済的根拠（docs/36）**:
- 重工業 mean ATE > 0、IT/金融 mean ATE < 0 → kind 別エッジが value+PEAD の脚衝突を予測
- 構造変化頻度が高い業種（金融・食料等）を横断集約に含めると `n_sectors_unstable` が精緻化

## 1. 一次モデル（FROZEN・docs/25/34 と同一）

変更なし。

## 2. 特徴量（FROZEN）

### base（5本・docs/25）
`topix_vol`, `topix_mom`, `value_trail`, `dispersion`, `flow_intensity`

### causal_all33 追加（8本）

**横断集約（全33業種・9999除外）:**
1. `causal_edge` — 全業種平均エッジ強度
2. `edge_volatility` — 全業種平均エッジ安定性
3. `edge_chg_3m` — 3か月変化
4. `months_since_break` — 構造ブレイク経過月
5. `n_sectors_unstable` — 不安定業種数

**kind 別（凍結3種・docs/36 の業種経済学）:**
6. `edge_heavy_industry` — 重工業6業種の平均エッジ
7. `edge_it_comm` — IT通信3業種の平均エッジ
8. `edge_financial` — 金融4業種の平均エッジ

**合計13本**。選別・追加禁止。

## 3. 比較戦略（同一 judge_grid）

| name | 特徴量 |
|---|---|
| `value+pead_lt` | ungated |
| `value+pead_lt\|causal_rep6` | base5 + 因果5（代表6業種・docs/34） |
| `value+pead_lt\|causal_all33` | base5 + 因果8（全33業種） |

## 4. 判定基準

- 期間・OOS・warmup・モデル：docs/34 と同一
- scope=`value_pead_causal_meta_gate_all33`、K=3

**PASS（AND）** — `causal_all33` に対し:
1. OOS SR ≥ ungated OOS SR
2. 全期間 DSR ≥ ungated DSR
3. 全期間 DSR ≥ 0.95

**副次（報告のみ）**:
- causal_all33 OOS SR ≥ causal_rep6 OOS SR
- causal_all33 DSR ≥ causal_rep6 DSR

## 5. 実行後禁止

結果に基づく特徴量・閾値の事後変更禁止。