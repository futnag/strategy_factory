# メタラベル局面ゲート 事前登録（FROZEN）

**事前登録日**: 2026-06-22
**ステータス**: FROZEN（本ファイル確定後は、実行結果を見てから特徴量・モデル・分割・判定基準を変更しない）
**関連**: docs/03-research-findings.md（§6 value+PEAD の OOS 局面依存）、[[invest-system-project]]、AFML ch.3.6 / ch.10
**判定 scope**: `value_pead_meta_gate`

---

## 0. 一文の主張（a priori 仮説）

> value+PEAD 合成（現行最良・全 DSR0.91）の OOS 失速は**局面依存**である。各リバランス月 t で
> 「この賭けが翌月勝つ確率」を **t 以前のデータのみ**で学習する二次（メタ）モデルを置き、
> その確率を連続ベットサイズ（AFML snippet 10.1）に変換して一次ウェイトに乗じれば、
> **OOS Sharpe と多重検定後 DSR を ungated 比で改善する**。

**経済的合理性**: PEAD ショート脚は割安・ディストレス銘柄を売るため、value ロングと衝突する
（docs/03 §6.x の OOS 診断＝下方修正ショートが 2024–26 バリュー復権で逆噴射）。崩れは個別銘柄
ではなく**市場局面**（バリュー/グロースの優劣・ボラ・トレンド）に駆動される。局面条件付きで
エクスポージャを絞れば、不利局面での負けを構造的に削れる。これは「方向でなくサイジング・
リスクに ML を」という López de Prado の指針そのもの。

## 1. 既存資産との差分（重複しない理由）

- `RegimeGated`（strategies_meanrev.py）＝**離散レジームラベル×手設定 sizing**。sizing を
  breakdown から選べば in-sample。→ 本ゲートは sizing を**学習**する点が異なる。
- `walk_forward_regime_assignment`（judge.py）＝walk-forward だが**単一レジーム条件付き平均で
  スリーブを離散選択**。→ 本ゲートは**複数特徴量の確率分類→連続サイズ**で、これは AFML
  メタラベルの本体（二次確率モデル→ベットサイズ）に対応する。

## 2. 一次モデル（FROZEN・無改修で流用）

`examples/research_value_pead_longtilt.py` と同一構成：
- `value_ls = CrossSectionalStrategy(value, 0.2, name="value")`
- `pead_lt = CrossSectionalStrategy(pead, 0.2, name="pead_longtilt", long_only=True)`
- `primary = CompositeStrategy([value_ls, pead_lt], [0.5, 0.5], name="value+pead_lt")`
- ユニバース＝`point_in_time_universe(top_n=300, lookback=12, min_obs=6)`、S33 中立、z 化、
  `lag_days=1`、costs 15bps、participation 0.1。**一次は一切変更しない**。

## 3. 二次（メタ）モデル＝局面ゲート（FROZEN）

`invest_system/research/meta_gate.py::fit_meta_gate`。

### 3.1 ラベル
`y_s = 1{ primary_net[s] > 0 }`（一次のネット月次リターンが翌月プラスか）。
これは `get_bins(side=...)` の {0,1} メタラベルと同義（方向はウェイトに織込済）。

### 3.2 特徴量（**5 本に凍結**・全て t 時点で PIT）
1. `topix_vol`     — TOPIX 直近 6 か月の月次実現ボラ（標準偏差）
2. `topix_mom`     — TOPIX 直近 6 か月リターン（トレンド/レジーム）
3. `value_trail`   — value スリーブ自身の直近 12 か月 Sharpe（自己状態＝崩れ検知、≤t-1 の実績）
4. `dispersion`    — 当月ユニバース銘柄の直近 1 か月リターン横断標準偏差（分散の広さ）
5. `flow_intensity`— 海外勢ネットフロー強度（`equities/flows.py`、公表遅延反映済）

※ 特徴量の追加・差替・選別はしない（探索＝p-hacking のため凍結）。欠損は当月ゲート無効（=スケール1.0）。

### 3.3 学習プロトコル（walk-forward・リーク防止）
- 各 t で**拡張窓**：位置 `≤ i-1-embargo`（embargo=1）の決済済みベットのみ訓練。
- `warmup=36`（3 年）未満の訓練数ではゲート無効（スケール NaN→ラッパが 1.0 扱い）。
- モデル＝**標準化＋ロジスティック回帰**（`StandardScaler`→`LogisticRegression(C=1.0)`）。
  単一クラスしか無い窓は基準率 `mean(y)` を確率とする。**モデル族・正則化は固定**。
- 確率 p → サイズ＝`clip(bet_size_from_prob(p), 0, 1)`（AFML snippet 10.1。既存実装を使用）。

### 3.4 適用
`MetaGatedStrategy(primary, scale)`：各 t で `primary` のウェイトに「≤t の最新スケール」を乗じる
（NaN/未確定は 1.0＝フル建玉＝保守的に素の一次に従う）。`AsOf` と `scale.loc[:asof]` で先読み不能。

## 4. 評価と判定基準（**実行前に確定**）

- 期間：2016-07〜2026-05、月次。保留 OOS = **2024-01〜**（IS=2016-07〜2023-12）。
- ゲートは walk-forward ゆえ OOS 予測は OOS 以前＋拡張窓のみ使用＝真の OOS。
- `judge_grid` に **ungated（value+pead_lt）と gated を同一呼び出しで**投入し、同一 scope
  `value_pead_meta_gate` の K で**まとめてデフレート**（ゲート追加＝K+1 で基準が上がる）。

**PASS 条件（AND・事前確定）**:
1. gated OOS SR ≥ ungated OOS SR、かつ
2. gated 全期間 DSR ≥ ungated 全期間 DSR、かつ
3. gated DSR ≥ **0.95**

**事前の期待**: 全研究で誰も DSR0.95 に未達。最有力の結末は「①②は満たすが③未達」。その場合も
**「動的メタ重みは OOS を改善するが単独認定には足りない」という正当な結論**として docs/26 に記録し、
**通すための再調整（特徴量変更・モデル変更・分割変更）は行わない**。

## 5. p-hacking ガードレール（構造的強制）

| リスク | 封じ手 |
|---|---|
| 特徴/モデル/分割の事後変更 | 本 FROZEN doc。`run_phase4_judgment.py` 同様の二重実行ガードで**一度きり**実行 |
| ゲート試行が K に乗らない | ungated と gated を同一 `judge_grid`＝同一 scope の K で同時デフレート |
| 二次モデルのリーク | walk-forward 拡張窓＋embargo＋`AsOf`／`MetaGatedStrategy` の `loc[:asof]`（多重防御） |
| 分離が無いのにゲート | **診断ファースト**：ungated で `regime_breakdown` を見て分離を確認（throwaway・K 不変） |
| 二次モデル自身の過学習 | 特徴 5 本・浅い正則化ロジスティック・確率の OOS 精度を併記 |

## 6. 実行手順
1. （本 doc 凍結済み）
2. コード＋テスト実装・`pytest` green（K 不変）
3. 診断ファースト：ungated `regime_breakdown` で局面分離を確認（無ければ NO-GO）
4. `examples/research_value_pead_meta_gate.py` を**一度だけ**実行 → docs/26 に記録
