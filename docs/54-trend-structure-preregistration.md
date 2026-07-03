# 54. トレンド設計の構造比較（trend_structure）— 事前登録

**scope=`trend_structure`・K=6（グリッド6セル・extra_trials=0）・2026-07-03 事前登録**
（起源: /research-scout 2026-07-03 調査＝research_loop/IDEAS I-4/I-5/I-6・BACKLOG 昇格。
サイクル `2026-07-03-trend-structure`）

---

## §0 既知の失敗型の構造的回避（LESSONS §1 との突合）

| 失敗型 | 本設計での回避 |
|---|---|
| F1 直近regime共変 | 対象はマルチアセット時系列＝日本株 value/PBR 改革と機構が別。サブ期間符号を副次基準に |
| F2 OOS負 | OOS(2024-01+)は設計不使用・診断のみ。3論文の方向は**他市場データで固定**（ローカル覗き見なし） |
| F3 符号逆 | H1-H3 の方向は文献値で事前固定（下記）。逆なら FAIL |
| F4 momentum代理 | **代理であることは自明の前提**＝判定は「本番型ベースラインへの増分」（副次基準）に設計 |
| F7 既試redux | 既試 `tsmom_multiasset` は lookback{3,6,12}+blend のみ。**設計構造**（EMA速度・バーベル・非線形写像）は未試行 |
| F8 cost死 | 月次リバランス＋先物/FX 実勢コスト（§2.5）。速い脚（EMA60）の回転はコスト sweep で露出 |
| F9 データ制約 | 既存ミラー（data/investers・本番と同一11資産）で完結。§1 で健全性確認 |
| H-11 複雑度 | 全セルのパラメータ（半減期・ラダー・関数形）は**論文値で事前固定**＝推定なし |

## §1 実現可能性
- データ: `load_external_prices(KEYS)` の日次 close/open（本番 tsmom_multiasset と同一の11資産:
  nk225_fut, sp500, nasdaq_comp, gold, silver, platinum, wti, copper, usdjpy, eurjpy, audjpy。
  natgas はロール痕で既に事前除外済み＝docs/03 §6.15）。2016〜2026-06。
- 実行スクリプトの冒頭で `_health`（行数・期間・|r|>8% 跳ね数）を表示し確認する。
- 既知のデータ限界（継続足のロール痕・WTI 2020-04）は §6.15 の記載を継承。

## §2 事前登録（固定・実行前コミット）

### §2.1 仮説（方向は他市場文献で事前固定）
- **H1（単一EMA最適性）**: 適切な速度の単一EMA（半減期≈78営業日）は 12ヶ月ルックバックの
  サイン型と同等以上（Valeyre 2025: 一時間尺度ドリフト過程なら単一EMAで十分・R²≈0.98）。
- **H2（バーベル）**: 短期＋長期の2速度バーベルは5速度等加重ラダーと同等以上
  （Etienne et al. 2025: 中間~125日は冗長）。
- **H3（S字非線形）**: トレンド→ポジションの写像を線形サインから S字（極端の減衰）に替えると
  改善（Moskowitz et al. 2025: 非線形 SR 0.83-0.84 vs 線形 0.70・DD局面に利得集中）。

### §2.2 経済的根拠
トレンドの情報は（a）一つの支配的タイムスケールに集中し（Valeyre の理論適合）、（b）極端な
読みではクラウディング/オーバーシュートにより信頼度が低下する（S字）。既存の 12-1 サインは
この2点を無視した設計＝改善余地が理論的に特定されている。3論文の主張は部分対立しており、
同一ユニバース・同一コストの一発比較が最も情報量が多い。

### §2.3 データと PIT 規律（チェックリスト確認済み）
- 意思決定は月末終値（PIT・直近既知値 ffill≤7日）、約定は**翌営業日始値**（DP17・fill価格ビュー
  ＋Replay・engine lag=0＝実現 open→open）。本番 tsmom_multiasset と同一機構。
- シグナルは ≤t の日次終値のみ（EMA・12M リターン・63日ボラ）。
- OOS(2024-01+)は設計・セル選択に不使用（診断のみ）。
- 上場廃止/ユニバース変動なし（固定11資産）。kaiten 系結果は不使用。

### §2.4 シグナル構築＝固定グリッド（6セル・K=6）

共通: 資産ごとの生シグナル s∈[-1,1] → ウェイト w = s×(vol_target/σ_63d)/N（vol_target=0.10・
floor=0.05・N=11＝本番 `tsmom_weights` と同一正規化）。z ≡ r_12M/σ_ann（12ヶ月リターンを
63日年率ボラで標準化）。EMA は日次終値・半減期表記。

| # | strategy_id | シグナル s | 検定対象 |
|---|---|---|---|
| 1 | `tsmom_12m` | sign(r_12M) | ベースライン（本番族の複製・統制） |
| 2 | `ema_single` | sign(P − EMA_hl78) | H1 |
| 3 | `ema_barbell` | ½[sign(P−EMA_hl60) + sign(P−EMA_hl500)] | H2 |
| 4 | `ema_ladder` | mean over hl∈{20,60,125,250,500} of sign(P−EMA_hl) | H2 の統制（冗長ラダー） |
| 5 | `scurve_tanh` | tanh(z) | H3（飽和型） |
| 6 | `scurve_bell` | z·exp(−z²/4) / 0.858（max正規化・極端で反り返る型） | H3（反転型） |

### §2.5 judge 配線
`judge_grid(scope="trend_structure", costs_bps=5.0, registry=default_registry())`＋fill価格ビュー。
**コスト床15bps未満の正当化**: 先物/FX 実勢（片道1-2bp＋ロール償却）＝本番 tsmom_multiasset
（docs/03 §6.15・DP17）の確立済み前提を同一ユニバースに適用。診断で 10bps/15bps 感応を必須表示。
容量基準は適用外（指数先物/FX＝ADV 制約なし・adv 不使用）。

### §2.6 合格基準（事前固定）
1. **DSR ≥ 0.95**（唯一の主基準・scope 累計 K でデフレート）
2. 副次（PASS の追加条件）:
   (i) 最良セルが `tsmom_12m`（ベースライン）のネット SR を上回る
   (ii) 最良セルのサブ期間（judge 3分割）符号が 2/3 以上正
3. **仮説別の記録**（PASS/FAIL とは独立に §5 へ）: H1= ema_single vs tsmom_12m、
   H2= ema_barbell vs ema_ladder、H3= max(scurve) vs tsmom_12m の各ネット SR 差の符号。
4. 未達は機械的に FAIL。事後救済なし（セル追加・関数形変更・半減期調整の再実行禁止）。

### §2.7 既知の限界（正直に）
- ベースライン超えでも「本番採用」は自動でない（本番は blend 型・採用提案は人間ゲート＝§A-5）。
- 継続足のロール痕（§6.15 継承）＝P&L はキャリー分歪む。WTI 2020-04 寄与を診断表示。
- n_obs≈120ヶ月・K=6＝DSR≥0.95 には SR_ann ~1.2 級が必要（tsmom_multiasset は 0.86 止まり）。
  「FAIL だが H1-H3 の符号は判明」という結果でも設計知見として価値がある（§2.6-3）。
- S字の z 標準化はボラ推定に依存（63日窓は事前固定・感応チェックはしない＝K節約）。

## §3 実装ステージ（同一サイクル内で許される段階＝これのみ）
1. **Stage 1（本判定）**: §2.4 の6セルを judge_grid で1回判定。
2. 診断（throwaway・K不変）: gross/net・IS/OOS(2024-01)・pre/post2020・年次SR・コスト感応
  （5/10/15bps）・回転率比較・H1/H2/H3 の対比較表・本番 tsmom_blend との相関・
  最良セル vs ベースラインの月次差分系列の符号検定・WTI 2020-04 寄与。

## §4 出典
- Valeyre (2025) "Breaking the Trend: How to Avoid Cherry-Picked Signals" arXiv:2504.10914
- Etienne, Ohana, Benhamou, Guez, Setrouk & Jacquot (2025) "Revisiting the Structure of Trend
  Premia: When Diversification Hides Redundancy" arXiv:2510.23150
- Moskowitz, Sabbatucci, Tamoni & Uhl (2025) "Nonlinear Time Series Momentum" SSRN 5933974
- 先行 scope: `tsmom_multiasset`（docs/03 §6.15・K=4）／調査: research_loop/surveys/2026-07-03。

## §5 結果（実行後に記入）
（未実行）
