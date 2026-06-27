# `data/` — データ格納ディレクトリ（索引）

> **このディレクトリは git 管理外**です（`.gitignore` の `/data/*`）。唯一の例外がこの
> `data/README.md`。市場データ・EDINET取得物・M&A Online md は**絶対にコミットしない**
> （各 ToS・容量）。コミット可はメタ（名簿・様式コード表）のみ。

**完全なデータカタログ（どこに・何が・どんな形式で・どう読むか）は正本を参照:**

➡️ [`docs/24-data-catalog.md`](../docs/24-data-catalog.md)

## クイック早見（詳細は上記正本へ）

| 何が | どこ | 読み方 |
|---|---|---|
| 全銘柄日次株価 | `jquants/daily/` | `panel.load_daily_panel("AdjC")` |
| Date×Code wide | `processed/equities/wide/` | `store.load_wide("adj_close")` |
| 派生特徴量 | `features/` | `feature_store.load_feature(name)` |
| 財務 | `jquants/{fins_summary,statements}/` | `fundamentals.load_fundamentals()` |
| EDINET 書類一覧/生 | `edinet/{list,docs}/` | `sources/edinet.py` |
| TOB/大量保有 | `edinet/{tob_deals,large_holdings}.parquet` | 直接 read |
| クロスアセット/マクロ | `investers/`, `supplemental/` | `external.load_external_prices/load_macro` |
| 研究行列（GKX） | `phase4/`, `phase4b/` | 直接 read（`X/y/universe/macro/meta`） |
| 試行レジストリ（DSR） | `research_trials.db` | `invest_system/validation/` |
| TOB履歴（手動） | `manual/maonline/` | `parsed/*.csv` |
| TDnet 適時開示（前向き） | `tdnet/` | `tdnet.load_tdnet()` |
| ToSTNeT 超大口（前向き） | `jpx_tostnet/` | `jpx_tostnet.load_tostnet()` |

**鉄則**: ① 生パス直読みよりローダ経由 ② PIT規律（提出日アンカー・`adj_*`は派生・`_displayed`は
非安全） ③ 銘柄コードは英字混じり5桁文字列（`130A0`） ④ `_empty`列はデータ無しマーカー。
詳細・スキーマ・略語グロッサリは [`docs/24-data-catalog.md`](../docs/24-data-catalog.md)。
