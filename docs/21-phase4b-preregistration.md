# 21. Phase 4b 事前登録メモ（小型寄りユニバース・独立した新判定）

> **STATUS: FROZEN（2026-06-21・Handoff_phase4b_smallcap_universe.md に基づき凍結）**
> OOS 結果を見る前に本設定を凍結した。`examples/run_phase4b_judgment.py` を**一度だけ**実行する。
> 前判定（docs/19/20・scope `phase4_gkx_judgment`）には一切触れない。

## 0. 位置づけ

Phase 4（FAIL・DSR=0.905）の**事後修正ではない**。独立した新仮説：

> mcap 床を ¥100億→¥30億に下げた小型寄りユニバースでは、GKX 型 ML エッジが認定基準（DSR≥0.95）を越えるか。

新スコープ `phase4b_gkx_judgment` で **K+=1**（判定 1 件のみ）。

---

## 1. 前判定との差分表（唯一の変更点）

| 項目 | Phase 4（docs/19 FROZEN） | Phase 4b（本書 FROZEN） |
|---|---|---|
| **mcap 床** | **¥100億**（1e10） | **¥30億**（3e9）← **唯一の変更** |
| 株価床 | ¥100 | ¥100（同一） |
| ADV 床 | 60日 ¥5000万 | 60日 ¥5000万（同一） |
| 普通株フィルタ | あり | あり（同一） |
| 超過リターン基準 | 本番ユニバース断面平均 | **新ユニバース断面平均**（従属変更・必須） |
| 特徴量セット | 60列凍結 | **同一**（新規追加なし） |
| 標準化 | winsorize 1/99% → rank_and_fill | **同一** |
| マクロ | 5系列＋交互作用（線形のみ） | **同一** |
| 推定器 | OLS/Lasso/EN/RF/HGBRT/MLP（N=6） | **同一** |
| CPCV | k=2, n_splits=6, φ=5 | **同一** |
| purge/embargo | gap_before=1, gap_after=12 | **同一** |
| OOS 期間 | 2018-01+ | **同一** |
| コスト | 片道 15bps・デシル L/S | **同一** |
| 廃止補完 | last-price | **同一** |
| 選択規則 | 平均パス Sharpe 最大の族 | **同一** |
| デフレート | N=6, V[SR]=パス間分散 | **同一** |
| 合否閾値 | DSR≥0.95 ∧ net>0 ∧ 封筒−100%>0 | **同一** |
| レジストリ scope | `phase4_gkx_judgment`（不変） | `phase4b_gkx_judgment`（新規） |

ユニバース実装：`design_matrix.production_universe(preset="smallcap_30b")`。
キャッシュ：`data/phase4b/`。

---

## 2. 凍結された入力（Phase 4 と同一・mcap 床除く）

- **ユニバース**：株価≥¥100・時価総額≥**¥30億**・60日 ADV≥¥5000万・普通株・PIT。
- **リターン/ラベル**：1M 先・トータル・超過（**新ユニバース断面平均控除**）・close[t]→close[t+1]・廃止 last-price。
- **特徴量**：60列（docs/18 重複解消済み・Phase 4 と同一）。
- **推定器・検証・判定**：docs/19 §2–§5 と完全同一。

---

## 3. 小型固有診断（throwaway・K 不変・判定選択に不使用）

判定前に `examples/phase4b_preflight_diag.py` で実測し、docs/22 に併報：

1. コスト感応度（15bps vs 30bps）
2. 廃止封筒幅（last-price ↔ −100%）の増幅
3. 因子被覆劣化（EDINET 系）
4. 容量曲線（AUM 別 Sharpe）
5. 全上場 vs 新ユニバース（追加銘柄数・除外テール）

---

## 4. 規律

1. mcap 床 **¥30億のみ**（¥50億等の複数床は走らせない）。
2. 床以外は一切変更しない。FAIL でも事後変更なし。
3. 前判定スコープ・docs/19/20 は不変。
4. 新スコープで K+=1 のみ。