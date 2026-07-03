# 56. 流動性変化のアンダーリアクション（delta_liquidity）— 事前登録

**scope=`delta_liquidity`・K=4（グリッド4セル・extra_trials=0）・2026-07-03 事前登録**
（起源: /research-scout 2026-07-03 調査＝research_loop/IDEAS I-29・BACKLOG 昇格。
サイクル `2026-07-03-delta-liquidity`）

---

## §0 既知の失敗型の構造的回避（LESSONS §1 との突合）

| 失敗型 | 本設計での回避 |
|---|---|
| F1 直近regime共変 | 流動性の変化＝value/PBR改革と機構が別。サブ期間符号を副次基準に |
| F2 OOS負 | OOS(2024-01+)設計不使用。方向は**日本主標本**の査読済み証拠で固定（H-15 適合） |
| F3 符号逆 | ⚠ 原論文 abstract が**成分依存の符号**を明記（短期変化→正・長期変化→負）＝
  成分ごとに方向を事前固定（§2.1）。abstract に忠実・全文非入手のため操作的定義は §2.4 で当方が固定 |
| F4/F7 代理・redux | 静的流動性**水準**（amihud/低ボラ系）・短期リバーサル・サイズとの残差化セルを
  主セル化＋残差セル生存を副次基準（§2.6）。**一階差分軸**は registry 52+ scope に無い |
| F5 de-risk | XS ドルニュートラル＝該当なし |
| F8 cost死 | 月次・top500 流動ユニバース・15bps＋貸株115bps。ただし「流動性が激変した銘柄」は
  マイクロストラクチャ不安定＝コスト感応を診断で必須表示 |
| F9 データ制約 | 日次 wide（adj_close・turnover）から Amihud を直接構築＝完全ローカル |
| H-12/H-14 | 分位L/S（各脚~100銘柄）＝breadth 十分。既存スリーブの改良でない新規軸 |

## §1 実現可能性
- Amihud 非流動性: `data/processed/equities/wide/` の adj_close・turnover から
  illiq_d = |r_d| / max(turnover_d, floor) を日次構築し月次平均（Amihud 2002 標準形）。
  2016-06〜2026-06 全期間で構築可能（features に amihud_illiq が既存＝データ実在の傍証。
  本検証では窓とfloorを制御するため raw から再構築する）。
- ユニバース: `point_in_time_universe`（月次売買代金中央値・top500・lookback12）。
- ΔLT に12ヶ月履歴が必要＝戦略リターンは 2017-07 頃開始（n≈107ヶ月）。

## §2 事前登録（固定・実行前コミット）

### §2.1 仮説（方向は日本主標本 Iwanaga & Hirose 2023 abstract で成分別に事前固定）
流動性 L ≡ −log(Amihud非流動性)（大＝流動的）。**銘柄固有**の変化（市場共通成分を除去）について:
- **H1（短期変化＝アンダーリアクション・ドリフト）**: 直近1ヶ月の固有流動性変化 ΔL_ST が
  **大きい**銘柄は翌月**アウトパフォーム**（正方向）。
- **H2（長期変化＝期待リターン・チャネル）**: 過去12ヶ月の固有流動性変化 ΔL_LT が
  **大きい**銘柄は翌月**アンダーパフォーム**（負方向＝耐久的な流動性改善は非流動性プレミアムの
  低下＝以後の期待リターン低下）。

### §2.2 経済的根拠
流動性が変わると銘柄が担う非流動性プレミアムが変わるが、再価格付けは即時に完結しない。
短期の変化は過小反応のドリフトを生み（H1）、長期の水準シフトは要求リターン自体を変える（H2）。
原論文（日本株）は attention 仮説を棄却し illiquidity ベースの機構を支持。

### §2.3 データと PIT 規律（チェックリスト確認済み）
- L は ≤t の日次データのみ（当月= t の月内日次平均）。月末決定→**翌営業日始値**約定
  （fill 価格ビュー・DP17・docs/54/55 と同一機構）。
- ユニバース: `point_in_time_universe` top500。上場廃止はパネル脱落処理。
- `limit_lock_flags` 適用・adv=trailing 売買代金・participation=0.1・貸株115bps。
- OOS(2024-01+)は設計・セル選択に不使用（診断のみ）。

### §2.4 シグナル構築＝固定グリッド（4セル・K=4）＝操作的定義（当方で事前固定）
- 月次 Amihud: A_m = mean_d∈m(|r_d|/max(turnover_d, ¥1百万))・L_m = −log(A_m)。
- **固有化**: 各月、L の変化から当月のクロスセクション平均を控除（市場共通成分の除去）。
- ΔL_ST = L_t − L_{t−1}（固有化後）・ΔL_LT = L_t − L_{t−12}（固有化後）。
- resid 版: シグナルを [直近1ヶ月リターン, L_t 水準, log(trailing ADV)] に日次クロスセクション
  回帰した残差（`cross_sectional_residualize`）＝リバーサル・流動性水準・サイズ代理の除去。

| # | strategy_id | シグナル | 方向 | 分位 |
|---|---|---|---|---|
| 1 | `dliq_st_q20` | ΔL_ST（生） | ＋（H1） | 0.2 |
| 2 | `dliq_st_resid_q20` | ΔL_ST 残差化（**H1 主セル**） | ＋ | 0.2 |
| 3 | `dliq_lt_q20` | −ΔL_LT（生） | 高ΔLTをショート（H2） | 0.2 |
| 4 | `dliq_lt_resid_q20` | −ΔL_LT 残差化（**H2 主セル**） | 同上 | 0.2 |

### §2.5 judge 配線
`judge_grid(scope="delta_liquidity", costs_bps=15, adv=trailing売買代金, participation=0.1,
no_buy/no_sell=limit_lock_flags, short_borrow_bps=115, registry=default_registry())`＋
fill 価格ビュー（月次）。

### §2.6 合格基準（事前固定）
1. **DSR ≥ 0.95**（唯一の主基準）
2. 副次（PASS の追加条件）: (i) 最良セル net SR>0 (ii) サブ期間符号 2/3 以上正
   (iii) **最良セルと同方向の残差化セルの net SR>0**（生のみ正なら水準/リバーサル代理）
   (iv) 容量 ≥ ¥5,000万
3. 未達は機械的に FAIL。事後救済なし（成分定義・窓・分位の変更再実行禁止）。

### §2.7 既知の限界（正直に）
- 原論文の全文非入手＝成分の操作的定義（1ヶ月/12ヶ月差・クロスセクション demean）は当方の
  忠実近似（明示的逸脱）。原論文は spread 分解等より精密な可能性。
- H1 と H2 は同一データの別ホライズン＝完全独立な2仮説ではない（scope 内 K=4 で会計）。
- 流動性激変銘柄のコスト実現性（診断で回転・コスト感応を表示）。
- n≈107ヶ月・K=4＝H-4 域（SR~0.5 では認定不能）。仮説別の符号記録は §5 に必ず残す。

## §3 実装ステージ（同一サイクル内で許される段階＝これのみ）
1. **Stage 1（本判定）**: §2.4 の4セルを judge_grid で1回判定。
2. 診断（throwaway・K不変）: gross/net・IS/OOS(2024-01)・pre/post2020・年次SR・コスト感応
  （0/15/30/50bps）・ρ̄（シグナル vs 短期リバーサル/L水準/サイズ/momentum）・
  H1/H2 の符号一致確認・分位の単調性。

## §4 出典
- Iwanaga & Hirose (2023) "Liquidity changes and decomposition in the Japanese equity market"
  PBFJ 81, 102115（abstract 確認済＝日本主標本・成分依存の符号）
- Iwanaga & Hirose (2022) "Liquidity shock and stock returns in the Japanese equity market" PBFJ 75
- Amihud (2002) の非流動性尺度。既試との区別: 静的な amihud 水準（features/低ボラ系・❌）とは
  一階差分軸で別・短期リバーサル（❌）とは残差化＋副次基準(iii)で区別
- 調査: research_loop/surveys/2026-07-03-crypto-event-jpnative.md（I-29）

## §5 結果（実行後に記入）
（未実行）
