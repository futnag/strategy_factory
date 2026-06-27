# 因果メタラベル局面ゲート 事前登録（FROZEN）

**事前登録日**: 2026-06-23
**ステータス**: FROZEN（本ファイル確定後は、実行結果を見てから特徴量・モデル・分割・判定基準を変更しない）
**関連**: docs/25-meta-gate-preregistration.md（base メタゲート）、docs/32-causal-sector-final-report.md
**判定 scope**: `value_pead_causal_meta_gate`

---

## 0. 一文の主張（a priori 仮説）

> value+PEAD 合成の OOS 失速は、市場統計量（TOPIX ボラ等）だけでなく、
> **セクター横断の因果構造状態**（VALUE→RET エッジ強度・安定性・構造ブレイク近接度）に
> 依存する。docs/25 の凍結5特徴量に因果特徴量5本を追加した二次メタモデルは、
> **ungated および base_meta 比で OOS Sharpe と多重検定後 DSR を改善する**。

**経済的合理性（López de Prado 2023/2025）**: パフォーマンスブレイクを待つのではなく
因果エッジの structural break を監視すべき。VALUE プレミアムはセクター・金利・テックサイクルで
因果構造が変わる（重工業=コモディティ、IT=NASDAQ）。不利レジーム（edge弱・不安定セクター多数）では
メタゲートがエクスポージャを縮小し、value+PEAD の脚衝突による負けを削る。

## 1. 一次モデル（FROZEN・docs/25 と同一）

- `value_ls` + `pead_lt` 合成 `[0.5, 0.5]`、name=`value+pead_lt`
- ユニバース `point_in_time_universe(top_n=300)`、S33 中立、z 化、`lag_days=1`
- costs 15bps、participation 0.1。**一次は無改修**。

## 2. 二次（メタ）モデル

`invest_system/research/meta_gate.py::fit_meta_gate`（docs/25 §3.3 と同一プロトコル）。

### 2.1 base_meta 特徴量（凍結5本・docs/25 §3.2）
`topix_vol`, `topix_mom`, `value_trail`, `dispersion`, `flow_intensity`

### 2.2 causal_meta 追加特徴量（凍結5本）
代表6業種（3300,3450,3600,3650,5250,7050）から月次集約：
1. `causal_edge` — VALUE→RET 調整済みエッジ強度（ローリング偏回帰）
2. `edge_volatility` — エッジ強度の60日 rolling std
3. `edge_chg_3m` — エッジ強度の3か月差分
4. `months_since_break` — 直近因果構造ブレイクからの経過月数
5. `n_sectors_unstable` — edge_volatility > 横断中央値のセクター数

※ 10本合計。追加・差替・選別禁止。欠損月はゲート無効（scale=1.0）。

### 2.3 学習プロトコル（FROZEN）
- walk-forward 拡張窓、`warmup=36`, `embargo=1`
- `StandardScaler` → `LogisticRegression(C=1.0)`
- 確率 → `bet_size_from_prob` → clip [0,1]

## 3. 比較戦略（同一 judge_grid 呼び出し）

| name | 説明 |
|---|---|
| `value+pead_lt` | ungated 一次 |
| `value+pead_lt\|base_meta` | 凍結5特徴のみ |
| `value+pead_lt\|causal_meta` | 凍結5 + 因果5 |

## 4. 評価と判定基準（実行前確定）

- 期間：2016-07〜2026-05。保留 OOS = **2024-01〜**
- `judge_grid` scope=`value_pead_causal_meta_gate`、3戦略を同時投入・同一 K でデフレート

**PASS 条件（AND）** — causal_meta に対し:
1. causal_meta OOS SR ≥ ungated OOS SR
2. causal_meta 全期間 DSR ≥ ungated 全期間 DSR
3. causal_meta DSR ≥ **0.95**

**副次比較（報告のみ・PASS 非条件）**:
- causal_meta OOS SR ≥ base_meta OOS SR
- 因果エッジ符号レジームでの regime_breakdown 分離

## 5. 実行後の禁止事項

結果を見てから特徴量・warmup・モデル・OOS 境界・PASS 閾値を変更しない。
負の結果も docs/35 に記録する。