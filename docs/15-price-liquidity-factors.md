# 15. 価格系・流動性系ファクター（GKX Phase 1）

GKX（Gu, Kelly & Xiu 2020）が最も支配的と示した **価格系**（多ホライズン・モメンタム／短期・
長期リバーサル／各種ボラ／特異ボラ／ベータ／MAX）と **流動性系**（Amihud／回転率／売買代金／
ゼロリターン日）を、**既存の J-Quants 日次データだけ**（EDINET 非依存・追加データコストゼロ）で
供給する Phase 1。Phase 4（ML推定器）の入力であり戦略探索ではない（**K不変・判定なし・診断は
throwaway**）。

実装：`equities/price_factors.py`（純関数）／`data/feature_store.py: build_price_liquidity_features`
（月次PIT材化）。標準化・中立化は `factors.py`（`cross_sectional_zscore`/`cross_sectional_rank`/
`sector_neutralize`）を**再利用**（再実装しない）。テスト：`tests/test_price_factors.py`（9件）。
関連：[03 研究知見](03-research-findings.md) §5、[14 EDINET ファンダ](14-edinet-fundamentals.md)。

---

## 1. 設計・規律

- **入力**：`adj_close`（調整済・モメンタム/ボラ）、`turnover`=Va 売買代金（流動性）、生 `close`＋
  as-of 株数（回転率の時価総額）、市場系列、S33 業種。すべて Silver（J-Quants 日次）由来。
- **市場系列＝等加重**（`adj_close.pct_change().mean(axis=1)`）。`feature_store.build_regime` と同一規約
  で TOPIX 等の外部系列に依存しない（ivol/beta の回帰に使用）。
- **PIT・先読み厳禁**：各 `f[t]` は ≤t の価格・出来高のみ。trailing 窓（rolling/shift）は過去方向。
  「未来改変→≤t 値が不変」を `tests/test_price_factors.py` で assert。
- **符号は「大きいほどロング側＝期待プレミアム方向」に統一**（低ボラ・低ベータ・低回転・MAX・
  歪度・売買代金は負号、Amihud・ゼロ日は正号）。
- **窓は事前固定**（チューニング探索しない）：短期=1M(≈21営業日)、ボラ=60/252、ベータ/特異ボラ
  =252、流動性=60。
- **材化は月次**：日次で計算→暦月の最終営業日値に as-of サンプリング→月次 wide（float32）。
- **3 ビュー**：raw を材化し、`feature_store.price_factor_view(name, view)` で
  raw／zscore／rank([-1,1])／sector_neutral を既存ユーティリティから生成。
- **評価は PIT ユニバース**（`universe.point_in_time_universe`・時変・生存者バイアス排除）上で。

---

## 2. 特徴量カタログ

`MONTH=21`。既存（再利用）：mom(12-1)=`value_quality_size_factors.momentum`、`low_volatility`、
`residual_momentum`、feature_store の `vol_20`/`momentum_12_1`/`reversal_5`。

| 特徴量 | 算式（符号込み） | 窓 | アノマリー出典 |
|---|---|---|---|
| mom_1m_reversal | −(adj/adj₋₁ₘ−1) | 1M | Jegadeesh 1990（短期リバーサル） |
| mom_3m | adj₋₁ₘ/adj₋₃ₘ−1 | 3M(skip1M) | Jegadeesh-Titman 1993 |
| mom_6m | adj₋₁ₘ/adj₋₆ₘ−1 | 6M(skip1M) | 同上 |
| mom_36m_reversal | −(adj₋₁₂ₘ/adj₋₃₆ₘ−1)（**直近12Mスキップ**） | 36M→12M | De Bondt-Thaler 1985 / Chen-Zimmermann LRreversal |
| industry_momentum | S33 業種 等加重 12-1 を構成銘柄へ | 12M(skip1M) | Moskowitz-Grinblatt 1999 |
| rvol_60 / rvol_252 | −年率実現ボラ | 60/252 | Ang ら 2006 / BBW 2011（低ボラ） |
| ivol | −年率 特異ボラ（市場単回帰残差・var(ret)−β²var(mkt) の閉形式） | 252 | AHXZ 2006 |
| beta | −trailing 市場ベータ | 252 | Frazzini-Pedersen 2014（BAB） |
| max_ret | −過去1Mの日次最大リターン | 1M | Bali-Cakici-Whitelaw 2011（MAX） |
| ret_skew | −trailing 日次リターン歪度 | 252 | 特異歪度（宝くじ）プレミアム |
| amihud_illiq | mean(\|ret\|/Va)（**正**） | 60 | Amihud 2002（非流動性） |
| turnover | −mean(Va)/時価総額 | 60 | Datar-Naik-Radcliffe 1998 |
| dollar_volume | −log(mean Va) | 60 | サイズ/流動性代理 |
| zero_ret_days | ゼロリターン日比率（**正**） | 60 | Lesmond-Ogden-Trzcinka 1999 |

特異ボラ/ベータは銘柄ループを避け **var(resid)=var(ret)−β²·var(mkt)** の閉形式でベクトル化。

---

## 3. 診断（`examples/diag_price_factors.py`・throwaway・K不変）

PIT ユニバース（上位500・lookback12M）・121 月（2016-06〜2026-06）・翌月リターンの月次 Spearman IC：

- **弱い正の IC**：ivol +0.029（IR 0.18）、ret_skew +0.022（**IR 0.27**＝最良）、rvol_60 +0.022、
  max_ret +0.019、mom_3m +0.020、mom_6m +0.017、industry_momentum +0.015。低ボラ・低歪度・短中期
  モメンタムが符号どおり弱く効く。
- **負の IC**：amihud_illiq −0.033（IR −0.34）。**上位500の流動ユニバース内では非流動性プレミアムが
  反転**＝Amihud の効果は真の小型に偏在し、流動大型では逆。dollar_volume −0.018、beta −0.021。
- いずれも単体の IC は小さい（**ML 入力としての素材**であり単独戦略ではない。判定は Phase 4）。

**新旧冗長性（時間平均 CS Spearman）**：mom_6m↔momentum_12_1 **+0.63**、rvol_60↔vol_20 **−0.82**
（rvol が −vol＝負号の反映）、mom_1m_reversal↔reversal_5 **+0.39**。＝**相関はあるが重複ではない**
（GKX は相関特徴を同一モデルに入れてよい。明白な重複は把握済み）。

---

## 4. 既知の限界

- **流動性系は小型・低流動性に偏在**：Amihud/回転率/ゼロ日は小型で支配的で、取引コスト・流動性
  フィルタとの相互作用が大きい（実装上の現実性は Phase 4 の容量・執行ラグ評価で扱う）。上記の
  amihud 反転（流動ユニバース内）はこの偏在の表れ。
- **市場系列は等加重**：時価加重や FF-JP ではない（追加データ回避）。ベータ/特異ボラの水準は
  この市場定義に依存。
- **回転率の被覆は株数 as-of に律速**：時価総額に fins_summary の発行済/自己株式を使うため、
  財務未取得の銘柄は NaN（最新月で ~3,600 銘柄）。
- **歪度の窓=252 に固定**（1M 版は MAX が担う）。窓のチューニング探索はしない（p-hacking 回避）。
- **月次サンプリング**：日次で計算し暦月末の最終営業日値を採る。日次・イベント系の用途は別途。
- **業種モメンタムは S33 の現時点スナップショット**（時変業種でない＝sector-PIT 限界・docs/03 §6.24 と
  整合）。業種日次リターンは `fillna(0.0)`（欠損日を 0 リターン扱い）。
- **準完全窓（min_periods=0.8×窓）**：`residual_momentum` の「完全窓要求」より緩い**意図的**選択
  （祝日・上場直後で過度に NaN を出さないため）。コードベース内で窓の厳密さ規約は一様でない。
- **β/ivol の市場系列は自銘柄を含む等加重**（leave-one-out でない）。N≈5,000 で自己包含バイアスは
  ~1/N＝無視可。per-t trailing は先読みを避けるが自己条件付けは避けない（実害は無視可）。
- **共分散の NaN 端点**：窓内に銘柄リターンの散発欠損があると E[r·m]/E[r] と var(mkt) の平均日集合が
  僅かにずれる（ほぼ完全なパネルでは無視可。上場直後・売買停止の多い小型で留意）。
- **命名衝突**：Silver フィールド `turnover`（=売買代金 Va）とファクター `turnover`（=回転率
  Va/時価総額）が同名（コードは使い分け済み）。

---

## 5. アンチ p-hacking（K不変）

データ基盤整備であり戦略探索ではない。`judge_grid`・永続レジストリは触らず、予測力確認は
`diag_price_factors.py` の **throwaway 診断（月次IC・被覆・冗長性相関）**に留める。新規 DSR 判定は
Phase 4 で一度だけ。`examples/registry_status.py` で K 不変を確認できる。
