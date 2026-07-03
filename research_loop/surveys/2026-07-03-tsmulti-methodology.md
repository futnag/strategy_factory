# 調査レポート 2026-07-03 — 時系列・マルチアセット／方法論（検証・ポートフォリオ構築）

**survey_id**: 2026-07-03-tsmulti-methodology（/research-scout 初回・E2E検証を兼ねる）
**テーマ**: ①TSMOM/トレンド/マルチアセット ②マクロ・レジーム条件付け＋日本固有 ③検証方法論 ④ポートフォリオ構築
**方法**: 4並列サブエージェント（WebSearch＋WebFetch・arXiv q-fin/stat・SSRN・JF/JFE/RFS/JBF/
Management Science/JPM/実務系）。全候補は abstract の実在・一致を fetch で確認（SSRN 403 は
著者ページ・RePEc・出版社ミラーで代替確認）。主宰側でも昇格候補の中核3本
（arXiv:2504.10914・arXiv:2510.23150・JEF ジャンプテール）を再検証済み。

## 結果サマリ

- 検証済み候補 **20件** → IDEAS 追記 **I-4〜I-23**（💡15・⬆5・🗑1 ※⬆は I-4/5/6→trend_structure、
  I-11→jump_tail_beta_xs、I-9/I-12 は既存 BACKLOG 項目の出典アンカー）
- BACKLOG 昇格 **2件**: `trend_structure`（トレンド設計の三つ巴：単一EMA vs バーベル vs 本番12-1 vs S字）・
  `jump_tail_beta_xs`（N225 OTMプット→個別株テールβ XS）
- 既存項目の強化 **2件**: `governance_event_value` に D'Ercole-Wagner-Yamada (JCF 2026)、
  `sjm_per_factor_regime` に Shu-Mulvey (arXiv:2410.14841) の出典アンカー
- 棄却 **1件**: 株主還元アナウンスメント・ドリフト（I-10・F7 既試redux＝shareholder_return §6.17）

## 昇格判断の要旨

1. **trend_structure**: 3論文（単一EMA最適性・バーベル・S字減衰）の主張が部分対立しており、
   同一ユニバース・同一コスト・本番12-1をベースラインにした1回のグリッド判定で3つの文献仮説を
   同時に裁ける＝サイクル効率が最良。データ完備・パラメータは論文値で事前固定＝小データ安全。
   momentum 代理は自明なので「本番スリーブへの増分」を副次基準化。
2. **jump_tail_beta_xs**: オプション情報の XS 利用は registry 初の軸（既試の VRP・レジームゲートと別物）。
   options_225 が 2016-06 から完備でデータ適合が高い。低ボラ/BAB への直交化と OTM 板の疎さが要注意
   （§1 実現可能性スキャンで先に確認する設計を BACKLOG に明記）。

## 方法論候補（judge_grid の枠外・platform 改善＝人間ゲート）

高レバレッジ順: I-14 階層ベイズ縮小（JKP・判定意味論の変更＝表示専用併走から）／
I-15 e-backtesting（Phase 2 監視のキルスイッチ原理化・意味論不変）／I-16 厳密 IS/OOS Sharpe 分布
（PSR/minTRL の drop-in 改善＋スパニング・ゲート基礎）／I-21 RTL 測定（最安・即効）／
I-17 複製率・I-18 合成テストベッド・I-19/I-20 結合/ボラ管理。
→ これらは research-cycle でなく「platform 改善提案」として人間が優先度判断（proposals/ 経由でなく
IDEAS に保持。着手する場合は throwaway 診断 or 別枠の事前登録比較で）。

## Near-miss（拾わなかった主なもの・理由）

- ネットワーク・モメンタム（arXiv 2501.07135）: 60+先物ユニバース前提＝11資産では薄すぎ
- 通貨ベーシス・モメンタム／VIX先物系／単一株オプションXSスキュー（Tian-Wu MS 2024）: データ制約
  （先物カーブ・単一株オプションなし）
- ターン・オブ・マンス指数先物: 日次イベント×数bps/日＝30bpsコスト床で cost death（H-3）
- Bayesian Parametric Portfolio Policies（arXiv 2602.21173）: 機械的ファクターブレンドの再輸入＝
  house finding と衝突
- Busseti-Ryu-Boyd DD制約Kelly (2016): 窓外だが I-22 より単純な先手として脚注価値
- 政策保有解消（cross-shareholding unwinding）: **検証可能な学術アンカー不在＝文献ギャップ**。
  ローカルの EDINET 保有＋ToSTNeT データでオリジナル研究が可能な空白（将来の自前仮説候補）

## カバレッジの空白（次回テーマ候補）

- 小口スケールでの戦略容量配分（文献は機関水準のみ）
- TDnet メタデータのリターン信号・季節性×フローの日本研究（2022+ の検証可能アンカーなし）
- クリプト（bitbank 軸）は今回未調査＝次回スコープ候補
