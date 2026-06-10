# strategy_factory (invest-system)

López de Prado の金融機械学習フレームワーク（検証の厳格さ・過学習対策）を土台に、Ernest Chan の平均回帰／統計的裁定（柱D）を戦略仮説面で補完する**ハイブリッド投資システム**。
**Linux / macOS / Windows で動作**（純 Python：numpy / pandas / scipy / scikit-learn /
statsmodels。OS 依存コードなし、パスは相対 / `pathlib`、改行は LF 固定）。

設計ドキュメント：
- [`docs/01-knowledge-base.md`](docs/01-knowledge-base.md) — 統合ナレッジベース（原理リファレンス。§11 に Ernie Chan 平均回帰／統計的裁定 補完）
- [`docs/02-system-design.md`](docs/02-system-design.md) — システム設計書 v0.2
- [`docs/03-research-findings.md`](docs/03-research-findings.md) — 研究知見・検証ファクトリ・データ資産（最新・最重要）

## 実装済みモジュール

| 層 | モジュール | 内容 | KB |
|----|-----------|------|----|
| L1-L2 | `invest_system/data/` | bitbank 取込（公開API・キー不要）、ドル/インバランスバー | §3.2 |
| L3-L4 | `invest_system/features/` | 分数階差分（メモリ保持定常化）、因果フィルタ（コライダー除去） | §3.1, §7 |
| L3-L4 | `invest_system/timeseries/` | 平均回帰・共和分（半減期/Hurst/CADF/Johansen/Kalman）＝柱D | §11 |
| L5/L7 | `invest_system/labeling/` | トリプルバリア、メタラベリング＋ベットサイジング | §4 |
| L6 | `invest_system/sampling/` | サンプル独自性（非IID重み付け）、逐次ブートストラップ | §4.3 |
| L8 | `invest_system/validation/` | パージング/エンバーゴ、CPCV、DSR、試行レジストリ | §5 |
| L8 | `invest_system/equities/frictions.py` | 日本固有の執行フリクション（値幅制限の張り付き・貸借銘柄マスク・ボラ連動コスト）＝DP15/17 | §5 |
| — | `invest_system/backtest/` | purged CPCV バックテスト（Sharpe を分布で評価） | §5.2 |
| L9 | `invest_system/portfolio/` | ノイズ除去(RMT)、最小分散、HRP、NCO、フラクショナル・ケリー（DP16） | §6 |
| — | `invest_system/research/strategies_tsmom.py` | マルチアセット時系列モメンタム（MOP2012型・柱E候補＝保留、[`docs/03`§6.15]） | — |

> **柱D（時系列・統計的裁定／Ernie Chan 補完）＝実装済み**：`invest_system/timeseries/` ＋
> `research/strategies_meanrev.py` の `CointegratedPairs`/`JohansenBasket`/`LinearMeanReversion`、
> 検証 `examples/research_meanrev_pairs.py`。原理 [`docs/01`§11]、設計 [`docs/02`§5.2・DP12-14]、検証 [`docs/03`§6.6]。

## セットアップ

### Linux / macOS (bash)
```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install numpy pandas scipy statsmodels scikit-learn pytest
```

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install numpy pandas scipy statsmodels scikit-learn pytest
```

> 依存は `pyproject.toml` の `[project.dependencies]` にも記載。実データ取得（bitbank
> 公開ローソク足）は標準ライブラリのみで動作し、APIキーや `ccxt` は不要。

## テスト

```bash
# Linux / macOS
.venv/bin/python -m pytest -q
```
```powershell
# Windows (PowerShell)
.\.venv\Scripts\python.exe -m pytest -q
```

## デモの実行

各デモはリポジトリ root から実行する（`pyproject.toml` の `pythonpath` で `invest_system`
を解決。各スクリプトも自前で root を import パスへ追加するため単体実行可）。

```bash
# Linux / macOS（UTF-8 がデフォルトなので追加設定不要）
.venv/bin/python examples/end_to_end_demo.py
```
```powershell
# Windows (PowerShell)
$env:PYTHONUTF8 = "1"; .\.venv\Scripts\python.exe examples\end_to_end_demo.py
# 日本語が文字化けする場合は先に: chcp 65001
```

| デモ | 内容 |
|------|------|
| `validation_harness_demo.py` | 多重検定の罠（ノイズ最良戦略を DSR が偽物判定） |
| `frac_diff_demo.py` | 分数階差分（メモリ vs 定常性のトレードオフ） |
| `triple_barrier_demo.py` | トリプルバリア・ラベリング（利確/損切/時間切れ） |
| `uniqueness_demo.py` | サンプル独自性（実効サンプル数・逐次ブートストラップ） |
| `causal_filter_demo.py` | 因果フィルタ（コライダーバイアスによる符号反転） |
| `meta_labeling_demo.py` | メタラベリング（補正的AI が Precision を改善） |
| `portfolio_demo.py` | ポートフォリオ（ノイズ除去・HRP・NCO で Markowitz 安定化） |
| `end_to_end_demo.py` | 合成データで全部品を連結（特徴量→ラベル→CPCV→DSR） |
| `bitbank_e2e.py` | 実 BTC/JPY データで End-to-End（ネットワーク必要） |

## 日本株・検証ファクトリ（J-Quants）

任意の戦略の有効性を、蓄積データと検証メソッドで厳密に裁く**検証ファクトリ**
（`invest_system/research/` ＋ `invest_system/equities/`）。偽陽性を排する設計＝PITデータビュー・
実行ラグ/コスト/容量の現実モデル・**永続レジストリでの大域デフレートDSR**。詳細と研究知見は
[`docs/03-research-findings.md`](docs/03-research-findings.md)。

- **データ（J-Quants Standard・全営業日 by-date ミラー）**：株価四本値・財務サマリー・信用/空売り・
  指数(TOPIX含む)・投資部門別・日経225オプション(IV等)。市場データは利用規約によりコミットせず
  `data/`（gitignore）にのみ保持。一括取得は手元ターミナルで `examples/download_jquants.py`
  （Claude 非経由・進捗表示・中断再開）。`.env` に `J_QUANTS_API_KEY` が必要。
  ```powershell
  $env:J_QUANTS_MIN_INTERVAL="0.7"; .\.venv\Scripts\python.exe examples\download_jquants.py --all
  ```
- **代表的な検証スクリプト**：`research_value_pead_longtilt.py`（最有力候補 value＋ロングティルトPEAD）、
  `research_breadth_factors.py`（value×momentum×quality×low-vol の多ファクター breadth）、
  `research_value_pead_realism.py`（執行現実性チェック＝値幅制限・貸株コスト・ケリー、[`docs/03`§6.12]）、
  `research_value_pead_timing.py`（執行タイミング＝T+1始値・ボラ連動スリッページ、[`docs/03`§6.13]）、
  `research_index_events.py`（日経225入替イベント＝検証済み発表日付きイベント表
  `equities/index_events.py` で裁定、[`docs/03`§6.14]）、
  `research_tsmom_multiasset.py`（マルチアセットTSMOM・11資産・T+1始値、[`docs/03`§6.15]）。
- 全ユニバースの財務 as-of は `equities/fundamentals.py` の `fundamentals_panel()`／`load_fundamentals()`
  （全件ミラーを先読みなしで組立）。ファクターは `equities/factors.py`（value/quality/size/momentum/
  配当利回り/低ボラ/アクルーアル）。
- **柱D（実装済み・Chan 補完）**：共和分ペアの平均回帰が第3の独立アルファ源。`timeseries/` ツールキット
  ＋`CointegratedPairs` で、取得済み日足ミラー＋業種マスタから業種内ペアを CADF ゲート→全候補を試行計上
  →DSR デフレート（`examples/research_meanrev_pairs.py`・[`docs/03`§6.6]）。

## コマンド対応表（PowerShell ↔ bash）

| 操作 | Windows (PowerShell) | Linux / macOS (bash) |
|------|----------------------|----------------------|
| venv 作成 | `python -m venv .venv` | `python3 -m venv .venv` |
| venv の python | `.\.venv\Scripts\python.exe` | `.venv/bin/python` |
| 依存インストール | `.\.venv\Scripts\python.exe -m pip install <pkgs>` | `.venv/bin/python -m pip install <pkgs>` |
| テスト | `.\.venv\Scripts\python.exe -m pytest -q` | `.venv/bin/python -m pytest -q` |
| デモ実行 | `$env:PYTHONUTF8="1"; .\.venv\Scripts\python.exe examples\<x>.py` | `.venv/bin/python examples/<x>.py` |
| UTF-8 出力（化け対策） | `chcp 65001`（+ `$env:PYTHONUTF8="1"`） | 不要（既定で UTF-8） |
| venv 有効化（任意） | `.\.venv\Scripts\Activate.ps1` | `source .venv/bin/activate` |

> `git` 系コマンドは Windows / Linux で共通。

## クロスプラットフォーム方針
- コードは OS 非依存（純 Python＋科学計算ライブラリ）。`os.system` 等のシェル呼び出しなし。
- ファイルパスは相対 / `pathlib`。絶対パスのハードコードなし。
- 改行コードは `.gitattributes` で LF に固定。
- venv 内 python の差（`Scripts\python.exe` vs `bin/python`）以外、運用差はほぼ無い。
