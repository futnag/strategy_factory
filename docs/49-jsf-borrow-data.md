# 49 — JSF（日本証券金融）貸借・逆日歩データ（借株コストの実値化＋踏み上げ信号）

**作成日**: 2026-06-28
**前提**: `docs/24`（データカタログ）, `docs/48`（ストップ高リバーサル・main）, `equities/frictions.py`,
`research/engine.py`（`short_borrow_bps`）
**実装**: `invest_system/data/sources/jsf.py`, `examples/update_jsf.py`, `tests/test_jsf.py`（緑）

---

## 0. 位置づけと限界（先に読む）

これは docs/03 の「独立α探索」（9連敗）ではない。**プラットフォームの借株現実化**＝既存スリーブの
**ショート執行コストを実値化**する基盤データ。`engine` の `short_borrow_bps` 既定 ~115bps は制度貸株料の
保守仮定にすぎず、**逆日歩（品貸料）= 借株逼迫銘柄に課される 円/株/日 の追加コスト**を反映できていない。

**重要な限界（正直化）**: 本データでも **ストップ高リバーサル（docs/48）の判定は変わらない**。
docs/48 の診断が示した死因は**取引コスト（≈片道25bps損益分岐）であって借株は二次的**だった：

| lu_rev_h5 | net年率SR |
|---|---|
| 貸株 **0bps**（コスト30bps） | −0.02 ←借株ゼロでも負 |
| コスト 0bps（貸株300bps） | +0.98 |

逆日歩はロック銘柄で 300bps を**上回る**方向 → ショートは更に悪化こそすれ救済されない。
**よって本データの価値は limit-reversal ではなく**：
1. **value L/S 等・全ショート戦略の実コスト化**（旗艦の執行現実性＝docs/06 / `research_value_pead_realism.py` の精度向上）。
2. **逆日歩スパイク＝踏み上げ/クラウディング信号**（ロング簿の左尾回避オーバーレイ・将来の事前登録素地）。

---

## 1. データと出所

- **発行元**: 日本証券金融（JSF, https://www.jsf.co.jp/）。制度信用の**貸借取引貸株残高・融資残高・差引・
  逆日歩（品貸料率）・規制措置（貸株注意喚起／申込停止／増担保）**を**日次公表**。
- **公式 API なし**。リポジトリ規律に従い**手動DLのみ・自動スクレイピングはしない**（M&A Online と同じ。
  ToSTNeT/TDnet は JPX/TDnet の静的公開ページを薄く取得しているが、JSF はサイト構造／ToS を確認するまで
  ネット取得関数を**同梱しない**）。手動DLした生ファイルを `ingest_dir` で parquet 化する。
- **バックフィル限定・前向き蓄積**（ToSTNeT/TDnet と同型）。
- **代替の公式ソース**: **J-Quants Premium の貸借**（契約時）。CANON スキーマに合わせれば手動DL分と合流可能。
- **PIT規律**: 取引日 T の逆日歩は **T+1 に確定・公表**。バックテストでは公表ラグ（≤t-1）で参照する
  （`load_jsf` は `Date`=基準日で返す。ラグ適用＝呼び出し側。`borrow_cost_bps_panel` 注記参照）。

---

## 2. スキーマとローダ（`sources/jsf.py`）

正準スキーマ `CANON_COLUMNS`:
`Date, Code, loan_balance(融資残), lending_balance(貸株残), net_balance(差引), premium_rate(逆日歩 円/株/日),
regulation(規制), source`。Code は英字混じり5桁文字列（4桁は末尾0補完）。

| 関数 | 役割 |
|---|---|
| `parse_jsf_csv(content, default_date=)` | 手動DL生CSV/TSV → 正準DataFrame（**純関数・ネット不要**・ヘッダ自動検出・日英別名対応） |
| `ingest_dir(raw_dir, cache_dir)` | 生ファイル群 → `data/jsf/{YYYYMMDD}.parquet`（**冪等**・ファイル名先頭8桁を既定日付） |
| `load_jsf(start, end)` | 蓄積を1本に（`_empty` 除外） |
| `borrow_cost_bps_panel(jsf, close)` | 逆日歩 → **年率bps** の借株コスト wide（床=base_bps=115）＝engine ショート控除用 |
| `loan_lending_ratio(jsf)` | 貸借倍率=融資残/貸株残（<1＝貸株超過＝クラウディング） |
| `squeeze_flags(jsf)` | 逆日歩>0＝借株逼迫＝踏み上げリスク bool wide |

逆日歩→年率bps 換算: `日次率 = 逆日歩(円/株/日) / 株価`、`年率bps = 日次率 × ann_days(245) × 1e4 + base_bps`。

取得: `examples/update_jsf.py --raw <手動DLディレクトリ>`（冪等・ネット不要）。

---

## 3. engine 統合（次の一手・本ブランチでは未改変）

現状 `engine.backtest` の借株は**スカラ** `short_borrow_bps`（全ショートに一律・`short_gross` に按分）:
```python
borrow_per_period = short_borrow_bps / 1e4 / _ann_factor(dates)
net = r - cost - borrow_per_period * short_gross
```
**per-name 借株**を効かせる最小拡張（後方互換）＝`short_borrow_bps` がスカラ **または** Date×Code パネルを
受け、パネル時は各 t で `Σ_{i∈short} |w_i| × panel.loc[t,i]/1e4 / ann` を控除する（~10行）。これにより
`borrow_cost_bps_panel(load_jsf(), load_wide("close"))`（公表ラグ適用後）を直結できる。
**本ブランチではコア engine を改変せず**、データ層＋シグナル＋テストのみを提供（実データが前向きに
貯まってから wire する＝現時点では実 JSF データが無く統合は時期尚早）。

---

## 4. 検証計画（将来・事前登録の素地）

1. **旗艦の借株現実化**（最優先・α探索でない）: value↔PEAD switch のショート脚コストを 115bps 仮定 → 実逆日歩で
   再評価し、`docs/06` の執行現実性レンジを締める。判定は既存 scope の再走でなく**realism 診断**。
2. **逆日歩クラウディング・オーバーレイ**（任意・要事前登録）: `squeeze_flags`/`loan_lending_ratio` でロング簿から
   踏み上げ・割高貸株銘柄を除外 or 縮小。**事前注意**: 空売り残XS（`short_interest` DSR0.09）・margin系は
   全て FAIL 済（docs/03 §6.20）＝独立αとしての期待値は低い。あくまで**左尾回避**として小さく検証する。
3. データが前向きに数四半期貯まるまで判定は保留（前向き蓄積データの宿命）。

---

## 5. 出典・参考
- 日本証券金融 貸借取引関連データ: https://www.jsf.co.jp/
- J-Quants（Premium 貸借・代替公式ソース）: https://jpx-jquants.com/
- 関連: `docs/48`（ストップ高リバーサル＝借株でなくコストで死亡）, `docs/47`（C1前向き＋alt-dataパイプライン）,
  `tostnet_monitoring_plan.md`（前向き蓄積データの同型先例）。
