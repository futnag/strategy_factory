# 16. 信用/空売り・微細構造ファクター（GKX 特徴量拡充 A）

既にダウンロード済みなのに月次 PIT クロスセクション特徴量として未材化だったデータを GKX 特徴量に
起こす（**A-1 信用/空売り**・**A-2 微細構造**）。価格系と直交しうる需給・流動性情報を足すのが狙い。
Phase 4（ML推定器）の入力＝**K不変・判定なし・診断 throwaway・EDINET 非依存**。

実装：`equities/holdings_factors.py`（公表日アンカー long ビルダー）／`data/feature_store.py`
（`build_holdings_features`・`build_microstructure_features`）。既存 `margin.py`・
`microstructure.py`・`fundamentals.point_in_time`・標準化ユーティリティを**再利用**。
テスト：`tests/test_factor_expansion.py`。関連：[03 §5](03-research-findings.md)、
[15 価格・流動性](15-price-liquidity-factors.md)。

---

## 1. A-1 信用/空売り（公表日アンカー）

最重要規律：**基準日でなく「公開され利用可能になった日」で ≤t 採用**（EDINET 提出日アンカーと同一
思想）。`available_date` を付与し `point_in_time(date_col="available_date", lag_days=0)` で月末 as-of。

| 公表日ラグ（事前固定・取引カレンダー） | 根拠 |
|---|---|
| 週次信用残高：基準日(金曜) **+2 取引営業日** | JPX は申込日(週末)の**翌週第2営業日**に公表（月=+1,火=+2）。取引カレンダー（Silver 日次の実取引日）で数える＝祝日週でズレない。短い週（営業日≤2）はレコード無し＝欠測。 |
| 大量空売り残高(short_positions)：**DiscDate** をそのまま | DiscDate が公表日（T+2 開示）。基準日 CalcDate は使わない。 |
| 業種別空売り比率：**+1 取引営業日** | 日次・翌営業日に利用可能（保守）。 |

| 特徴量 | 算式 | 符号 |
|---|---|---|
| margin_imbalance | (信用買残−売残)/(買残+売残) | **raw（符号中立）** |
| short_to_long | 信用売残/買残 | **raw（符号中立）** |
| margin_balance_change | 信用残合計(買+売)の前週比 | raw |
| short_interest | 対SO空売り残高比率（報告者合算）に**負号** | 負号（空売りアノマリー 高SI→低リターン） |
| days_to_cover | 空売り残株数 / trailing20日平均出来高 | raw |
| sector_short_ratio | S33 業種の空売り金額/総売り金額 → 構成銘柄へ展開 | raw |

信用需給（margin_imbalance/short_to_long）は文献でも符号が係争的なため **raw 維持**（ハード反転
しない）。実証符号は §3 の throwaway IC で確認する。sector_short_ratio は S33 を構成銘柄へ
ブロードキャスト（`industry_momentum` と同方式・sector-PIT 限界は §4）。

---

## 2. A-2 微細構造（調整済 OHLC）

`microstructure.py` の Series 関数を**全銘柄に適用**（dtype 非依存の関数は wide 直接、Roll/VPIN は
列ループ）→ 月末 as-of・float32。**入力は調整済 OHLC**（adj_high/low/open/close）＝分割日の
ジャンプでスプレッド/レンジ推定が壊れない（特に Roll/Corwin-Schultz は終値の日跨ぎ差に依存）。

| 特徴量 | 推定 | 窓 | 符号 |
|---|---|---|---|
| parkinson_vol | 高安レンジ・ボラ | 20 | **負号**（低ボラ） |
| garman_klass_vol | OHLC ボラ | 20 | **負号**（低ボラ） |
| roll_spread | Roll 実効スプレッド | 20 | 正号（非流動性） |
| corwin_schultz_spread | 高安スプレッド推定 | 2日 | 正号（非流動性） |
| vpin | Bulk-Volume 不均衡 | 50 | raw [0,1] |
| rsi | 相対力指数 | 14 | raw [0,100] |

※ `amihud_illiq` は price_factors で材化済みのため**追加しない**。volume は adj 版が Silver に
無いため raw を使用（VPIN は比率なので影響軽微・§4 明記）。

---

## 3. 診断（`examples/diag_factor_expansion.py`・throwaway・K不変）

PIT ユニバース上位500・121月（2016-06〜2026-06）・翌月リターンの月次 Spearman IC：

- **効くもの**：short_interest +0.028（IR **0.35**＝負号化が機能）、parkinson_vol/garman_klass_vol
  +0.031（IR 0.19・低ボラ）。
- **弱い/中立**：margin_imbalance/short_to_long ≈ 0（係争的・符号中立で妥当）、roll_spread/
  corwin_schultz/vpin/days_to_cover/sector_short_ratio ≈ 0。単体 IC は小（ML 素材・判定は Phase 4）。
- **冗長性**：parkinson_vol↔rvol_60 **+0.81**（ともにボラ）、**corwin_schultz↔amihud_illiq +0.06
  （別側面の非流動性＝重複でない・足す価値あり）**。

---

## 4. 既知の限界

- **信用/空売りは sector-PIT スナップショット**：sector_short_ratio の S33→銘柄展開は現時点の業種
  割当を使う（時変業種でない・docs/03 §6.24 と整合）。
- **公表日ラグは固定値**：週次 +2 営業日は JPX 標準だが稀な公表遅延は完全には排除しない（+3 に
  すれば更に安全側だが鮮度低下）。短い週は欠測。
- **volume は分割未調整**（Silver に adj_volume 無し）：VPIN/days_to_cover は raw 出来高を使用。
  VPIN は比率で影響軽微、days_to_cover は分割年に注意。
- **信用需給の符号は未確定**（raw 維持）：margin_imbalance/short_to_long の予測方向は throwaway IC
  で弱く、Phase 4 の正則化に委ねる。
- 流動性/微細構造は小型・低流動性に偏在（取引コスト依存）＝Phase 4 の容量・執行ラグ評価で扱う。

---

## 5. アンチ p-hacking（K不変）

データ基盤整備であり戦略探索ではない。`judge_grid`・永続レジストリは触らず、予測力確認は
`diag_factor_expansion.py` の throwaway 診断に留める。`examples/registry_status.py` で K 不変を確認。
