# 18. ファクター符号規約 ＆ 重複対応表（Phase 4 設計行列の前提）

Phase 4（ML 推定器・一度きりの DSR 判定）の**設計行列の列定義を曖昧さなく確定**するための
リファレンス。GKX 全特徴量は (1) Phase 1 価格/流動性、(2) Phase 2 EDINET 三表、(3) 拡充 A/B
（保有・マイクロストラクチャ・J-Quants ファンダ）にまたがり、**符号規約が混在**し、同種因子が
**二重に存在**する。判定前にここで一意化する（コード変更は伴わない＝K 不変・判定なし）。

> 出典：`equities/{price_factors,factors,edinet_factors,fundamental_factors,events,margin,holdings_factors}.py`、
> `features/microstructure.py`、`data/feature_store.py`（材化時に符号を適用する権威レイヤ）。

---

## 1. 符号規約（house vs raw）

二つの規約が混在する。**設計行列に入れる前に「どちらの規約か」を列ごとに明示**し、
標準化（zscore/rank）後に向きを揃える。

- **house 符号**：「大きいほどロング側＝期待プレミアム方向」に**設計時点で符号付け済み**。
  - 宣言：`price_factors.py:12-13`「符号は『大きいほどロング側＝期待プレミアム方向』に統一
    （低ボラ・低ベータ・低回転・MAX・歪度は負号、Amihud・ゼロ日は正号）」、
    `factors.py:34-37`「値の向きは『大きいほど割安/高品質/小型』になるよう符号付け」。
- **raw 符号**：会計上/統計上の自然な向き。**向き付けは推定器/ランク側に委ねる**。
  - 宣言：`edinet_factors.py:14-15`「符号は raw（会計上の自然な向き）。プレミアム方向は各
    docstring に明記（ML 入力なので向き付けは推定器/ランク側に委ねる）」。
- **mixed**：`fundamental_factors.py:14-16` は SUE・予想改訂・52週高値・安定度・Dimson β を
  house、成長系（sales/profit/equity growth・sustainable_growth）を raw とする。

### 1.1 モジュール別 規約一覧

| モジュール | 規約 | 備考（負号で house 化している因子） |
|---|---|---|
| `price_factors.py` | **house** 全部 | 低ボラ系 `rvol_60/252`,`ivol`,`beta`,`max_ret`,`ret_skew`,`turnover`,`dollar_volume`,`mom_1m_reversal`,`mom_36m_reversal` に負号。`amihud_illiq`,`zero_ret_days` は正号 |
| `factors.py`（value/quality/size） | **house** | `size=-log(mcap)`,`low_volatility=-std`。`accruals=(CFO-NP)/TA`（高い=高品質） |
| `edinet_factors.py` | **raw** | プレミアム方向は docstring：`asset_growth`/`net_share_issuance`/`accruals(=profit-cfo)` は**低い=プレミアム**、`gross_profitability`/`roic` は高い=高品質 |
| `fundamental_factors.py` | **mixed** | house：`sue_*`,`forecast_revision`,`roe_stability`,`margin_stability`,`high_52w`,`seasonality`,`dimson_beta(=-β)`。raw：`sales/profit/equity_growth`,`sustainable_growth` |
| `events.py` | house 寄り | `surprise`,`fcst_revision`,`div_revision`,`buyback`,`cons_score` は正＝好材料。`delay_days` は raw（正=遅延） |
| `margin.py` | **raw**（中立） | `margin_imbalance`,`short_to_long`,`short_interest`,`sector_short_ratio` |
| `holdings_factors.py`／材化 | 材化時に符号 | `feature_store.py:234` で `short_interest` のみ**負号**（空売りアノマリー）。他は raw |
| `features/microstructure.py`／材化 | 材化時に符号 | `feature_store.py:286-294` で `parkinson_vol`,`garman_klass_vol` に**負号**（低ボラ=ロング）。`roll/corwin` スプレッドは正（非流動） |

**規律**：raw 因子（EDINET・成長・margin）は Phase 4 で**ランク/zscore 後に推定器が向きを学習**
する前提。house 因子と raw 因子を**同一の単純合成（等加重）に混ぜない**。線形プローブや
等加重コンポジットを作る場合は、raw 因子に下表「プレミアム方向」の符号を掛けてから合成する。

### 1.2 raw 因子のプレミアム方向（合成時に掛ける符号）

| raw 因子 | プレミアム方向 | 合成時の符号 |
|---|---|---|
| `asset_growth`（EDINET） | 低い=プレミアム（investment 軸） | **−** |
| `net_share_issuance`（EDINET, 分割調整後） | 低い（希薄化少）=プレミアム | **−** |
| `accruals`（EDINET=`profit-cfo`/TA） | 低い=高品質 | **−** |
| `gross_profitability`,`roic`,`ebitda_margin` | 高い=高品質 | ＋ |
| `leverage`,`rd_intensity` | 仮説中立 | 推定器に委ねる |
| `sales/profit/equity_growth`,`sustainable_growth` | 仮説中立（成長） | 推定器に委ねる |
| `margin_imbalance`,`short_to_long` 等 | 仮説中立 | 推定器に委ねる |

---

## 2. 重複・近重複の対応表（Phase 4 でどれを採るか）

「同じ経済量を測る」列が複数モジュールに散在する。**設計行列では原則どちらか一方**を採り、
両方入れる場合は多重共線性・二重計上を承知の上で（推定器が正則化前提のときのみ）。

| # | 重複 | 判定 | 出典 | Phase 4 の既定（要ユーザ確定＝docs/17） |
|---|---|---|---|---|
| 1 | **accruals**：J-Quants vs EDINET | 同概念・**符号が逆** | `factors.py:60` `(CFO-NP)/TA`（house, 高=良） ⇔ `edinet_factors.py:80` `(profit-cfo)/TA`（raw, 低=良） | **どちらか一方**。両方入れるなら符号統一必須（ほぼ −1 倍関係）。既定：EDINET 版（生ライン由来・基準明示）を採用候補 |
| 2 | **CF/P**：`cf_yield` vs `cf_to_price` | **同一式** | `factors.py:49` `CFO/mcap` ＝ `edinet_factors.py:105` `cfo/mcap` | 一方のみ。`fcf_yield(=（cfo+cfi)/mcap)`・`fcf_yield_3y` は別物（CFI 込み）として残す |
| 3 | **成長 axis**：`equity_growth` vs `asset_growth` | 近接・**別軸として保持** | `fundamental_factors.py:70`（Eq 成長, J-Quants） ⇔ `edinet_factors.py:72`（総資産成長, EDINET） | 別物として両方可（資産 vs 自己資本）。docs/17 に明記済み |
| 4 | **モメンタム**：`momentum`(12-1) vs `momentum_12_1` | **同一** | `factors.py:65` ＝ `feature_store.py:97` | 一本化。`mom_3m/6m`(1m-skip)・`residual_momentum`・`industry_momentum` は別物 |
| 5 | **リバーサル**：`mom_1m_reversal`(21d) vs `reversal_5`(5d) | 同型・窓違い | `price_factors.py:47` ⇔ `feature_store.py:98` | 窓が違う＝別特徴として許容（短期 vs 超短期）。`mom_36m_reversal` は長期 |
| 6 | **希薄化**：`net_share_issuance`(EDINET) vs `buyback`(J-Quants) | 同軸・別ソース | `edinet_factors.py:76`（発行株数YoY） ⇔ `events.py:92`（`diff(TrShFY/ShOutFY)` 自己株比） | 別ソースの同軸。両方可（相関確認は Part 2-8）。**注：株数ソースは EDINET と J-Quants `ShOutFY` の二系統**（`AvgSh` は未使用） |
| 7 | **実現ボラ**：4 実装 | 重複多数 | `price_factors.realized_vol`(`rvol_60/252`) / `factors.low_volatility` / `feature_store.vol_20` / microstructure `parkinson_vol`,`garman_klass_vol` | 推定器に渡すのは**窓・推定法ごとに代表 1 本**に整理。close-to-close と OHLC レンジ系は別物として最大 2 系統 |
| 8 | **Amihud 非流動性**：2 実装 | **完全同一式** | `price_factors.py:158` ＝ `microstructure.py:30`（`mean(\|ret\|/Va)`） | 既に二重計上回避済み（`feature_store.py:278`「amihud は price_factors で材化済みのため追加しない」）。設計行列も 1 本 |
| 9 | **ベータ**：`beta` vs `dimson_beta` | 当期 vs ラグ込み | `price_factors.py:135`（当期・−β） ⇔ `fundamental_factors.py:111`（Dimson 和・−β, 薄商い） | 両方可（薄商い銘柄で差）。docstring が「整合」を明記 |
| 10 | **予想改訂 / サプライズ**：events vs fundamental_factors | 同概念・正規化違い | `events.forecast_revision`(`:13`)/`earnings_surprise`(`:29`) ⇔ `fundamental_factors.forecast_revision_raw`(`:49`)/`sue_*`(`:66-67`) | 正規化（÷価格 vs pct, FY-vs-interim マッチ）が異なる＝一方を主、他方は頑健性チェック扱い |

**スコープ確認（重複ではない＝因子を定義しないモジュール）**：`frictions.py`（執行制約・コスト）、
`stability.py`（Sharpe 診断）、`index_events.py`／`flows.py`（市場/部門レベル）、`tob_*`（イベント
裁定診断）。クロスセクション因子は持たない（Part 2-7 のコード再利用衝突チェック対象外）。

---

## 3. 株数ソースの注意（net_share_issuance / mcap / buyback）

| 用途 | ソース | フィールド | 備考 |
|---|---|---|---|
| 時価総額・回転率 | J-Quants | `ShOutFY`−`TrShFY` | `factors.py:41`,`feature_store.py:159-162` |
| `net_share_issuance` | EDINET | `shares_outstanding`（`TotalNumberOfIssuedSharesSummaryOfBusinessResults`） | `edinet_factors.py:76`。**分割調整後**（`attach_split_cf`） |
| `buyback` | J-Quants | `TrShFY/ShOutFY` の差分 | `events.py:92` |

`AvgSh`（期中平均株数）はリポジトリ未使用。EDINET と J-Quants の株数は**期末基準**で概ね一致
するはずだが、定義（自己株控除・単元）差を Part 2-8 の重複整合で確認する。
