# invest-system

López de Prado の金融機械学習フレームワークに基づく投資システム。
設計は [`docs/`](docs/) を参照：

- [`docs/01-knowledge-base.md`](docs/01-knowledge-base.md) — 統合ナレッジベース（原理リファレンス）
- [`docs/02-system-design.md`](docs/02-system-design.md) — システム設計書 v0.2

## 現在の実装範囲：検証ハーネス（L8）

「戦略より先に検証基盤を作る」（設計書 DP7 / DP10）という方針に従い、
最初に過学習を構造的に封じる検証ハーネスを実装している。

| モジュール | 内容 | KB |
|------------|------|----|
| `invest_system/validation/dsr.py` | PSR / E[max SR] / DSR / minTRL | §5.3-5.4 |
| `invest_system/validation/purge_embargo.py` | パージング＆エンバーゴ | §5.1 |
| `invest_system/validation/cpcv.py` | 組合せパージ済交差検証 | §5.2 |
| `invest_system/validation/registry.py` | 事前登録ゲート付き試行レジストリ | §2, §5.3 |

## セットアップ

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy pandas scipy pytest
```

pytest は `pyproject.toml` の `pythonpath=["."]` で `invest_system` を解決する
（個別パッケージ化は将来対応）。

## テスト

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
