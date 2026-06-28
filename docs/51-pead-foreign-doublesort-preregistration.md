# 51 — PEAD × 外国人持株：size 制御つき二重ソート/Fama-MacBeth（market-wide）事前登録

**作成日**: 2026-06-29
**前提**: docs/50（所有条件付きPEAD＝FAIL・foreign≈size が**標本で反転**＝breadth不足）／オフライン所有抽出器
（4,387社・docs/50 §8.1）。
**実装**: `examples/research_pead_foreign_doublesort.py`。

---

## 0. 動機（前回の未解決点を market-wide で決着）
docs/50 で「PEAD は低外国人/高個人で持続・高外国人で反転」は頑健に再現したが、**foreign% が size の代理を超えるか**は
top-300 の狭い breadth で標本依存に反転した（160≈ / 263 lowF> / 306 lowADV>）。今や所有が市場全域（4,387社）に
あるので、**breadth を確保し size を直接制御**して決着させる。**新規母集団・新規手法＝新規スコープ**（前回の FAIL は不変）。

## 1. 仮説（falsifiable・走らせる前に固定）
- **H1（二重ソート）**: 各 size 分位の中で、**低外国人ほど PEAD が大きい**（高外国人で消失/反転）が **size 各層で一貫**。
- **H2（Fama-MacBeth・決定的）**: 月次クロスセクション回帰
  `r₍t+1₎ = a + b·SUE + c·(SUE×foreign_rank) + d·(SUE×size_rank) + e·foreign_rank + f·size_rank`
  において、**係数 c < 0 が、size 交互作用 d を制御しても有意**（FM/Newey-West t）。
  ＝外国人比率は **size とは独立に** PEAD を弱める。c が非有意 or d に吸収されれば「foreign は size の代理」。
- **H3（判定 L/S）**: broad universe で外国人条件付き PEAD が plain/size 条件付きを上回り、**大域デフレートDSR・
  コスト・容量・T+1** で評価。

## 2. データ・ユニバース（PIT・a-priori 固定）
- **ユニバース**: **top-1000 liquid common**（trailing 60日 ADV）∩ 所有あり ∩ sue あり、ADV 下限 ≥¥50M/日。各月末ランク。
- **シグナル**: `feature_store.load_feature("sue_recent")`（surprise yield・月末PIT）をセクター中立 z。
- **size/流動性**: trailing ADV（API非依存）。`size_rank`＝月内クロスセクション pct ランク。
  （注: 厳密な時価総額でなく ADV 代理。mcap は頑健性で後日。）
- **外国人**: 銘柄別 **median foreign_pct**（持続値）。`foreign_rank`＝月内 pct ランク。
- フォワード＝月次 `adj_close.pct_change().shift(-1)`。

## 3. 手法（3本立て）
- **A. 二重ソート**: 各月 size 3分位 × foreign 3分位の各バケットで PEAD＝(SUE上位tercile − 下位tercile) の翌月平均。
  全月平均で 3×3 行列＋ size 各層の foreign 単調性。
- **B. Fama-MacBeth**: 上式を月次 OLS → 係数時系列の平均＋FM t（焦点＝c＝SUE×foreign）。診断（K不算入）。
- **C. 判定 L/S**: `judge_grid`（`scope="pead_foreign_doublesort"`）で
  `pead_all / pead_lowF / pead_highF / pead_lowADV / pead_xF` を broad universe・execution_lag・costs15bps・adv容量・
  値幅ロックで一回判定（大域デフレートDSR）。

## 4. 合格基準（事前固定）
1. **H1**: size 3層すべてで「低外国人 PEAD > 高外国人 PEAD」。
2. **H2（必須・決定的）**: c<0 かつ |FM t|≥2、size 交互作用 d を入れても保持。← foreign-beyond-size の判定。
3. **H3**: DSR≥0.95（多重検定後）＋サブ期間符号一貫＋容量/コスト現実性。
4. いずれも 2016+ の窓限定（論文 2002-2016 の減衰は再現不可）を明記。

## 5. 失敗型ガード
| 逆風 | 対処 |
|--|--|
| size 代理 | **B の SUE×size 同時投入**で foreign の size 超過分を直接検定（これが本研究の核） |
| breadth 不足（前回の反転） | top-1000 へ拡張＋二重ソートで size 各層を見る |
| capacity/cost | broad は中小型含む＝net・容量を判定に内包・T+1 で執行現実性 |
| 直近集中/過学習 | サブ期間符号一貫＋PBO 表示 |

## 6. 正直な事前予想
- breadth＋size制御で foreign-beyond-size が**分離する**か**size に吸収される**か、A/B が quantile 反転より頑健に決着。
- **認定（DSR≥0.95）は依然厳しい**見込み（2016+ で PEAD は弱い）。本研究の主目的は overlay が「foreign 固有」か
  「size/低流動の言い換え」かを**確定**し、cutoff 設計・本番採否の科学的根拠を与えること。

## 7. 進め方（安価決定的テスト先行）
A＋B（診断・K不算入）を先に出し foreign-beyond-size を判定 → 有望なら C を一回 judge（K計上）。結果は §8 に追記。

---

## 8. 結果（判定済み・2026-06-29）

**前処理の修正**: 特徴量の月末 index（`me`）の **77/121 しか日次パネル index に一致せず**、`reindex(me)` が ~44ヶ月を
NaN 化してユニバースを空にしていた（A/B/C を過小検出力化）。`reindex(me, method="ffill")`（月末以前の最終営業日を採用・
先読みなし）に修正 → uni 日平均 992（≈top1000）、**FM 114ヶ月**の proper-power に。

### A. 二重ソート（size3 × foreign3・PEAD %/月）
低外国人(frgn0) > 高外国人(frgn2) は **3 つの size 層すべてで方向一致**（最大＝大型 size2: +0.45 vs −0.27）。
ただし magnitude は小さく **t は弱い（最大 |t|≈1.4）**＝方向はあるが有意でない。

### B. Fama-MacBeth（114ヶ月・**決定的**）
| 係数 | coef(×1e-4) | FM t |
|---|--:|--:|
| SUE | +29.8 | 1.35 |
| **SUE×foreign（c）** | **−41.5** | **−0.95（n.s.）** |
| **SUE×size（d）** | −1.9 | −0.07（n.s.） |
| foreign（水準） | +151.9 | +4.06 |

→ **H2 棄却**：size を連続制御すると **SUE×foreign は非有意（t=−0.95）**。SUE×size も非有意。
＝**foreign-beyond-size は proper-power では支持されない**。docs/50 の「foreign≈size」を**強化・確定**。
（docs/50 §7.1 の 40ヶ月 t=−1.96 はアラインメント不具合による過小標本の産物だった＝検証規律の実例 #2。）

### C. 判定 L/S（scope=`pead_foreign_doublesort`, K=5）
| strategy | net年率SR | DSR | maxDD | 容量 |
|---|--:|--:|--:|--:|
| pead_lowADV(size) | +0.50 | **0.61** | −72.5% | ¥0.22億 |
| pead_lowF(低外国人) | +0.48 | 0.58 | −34.7% | ¥0.44億 |
| pead_all(基準) | +0.12 | 0.18 | −32.0% | ¥0.42億 |
| pead_highF(対照) | −0.18 | 0.03 | −35.4% | ¥1.09億 |
| pead_xF | −0.22 | 0.02 | −30.3% | ¥0.42億 |

**判定＝❌ FAIL**（最良 pead_lowADV DSR=0.61<0.95・PBO0.46）。条件付けは net SR を ~+0.5 へ引き上げ（PEAD 最良型）だが
未認定。**lowADV ≈ lowF（size≈foreign を再確認）**、高外国人は負。

### 最終結論（foreign-vs-size の決着）
- **foreign% は size を超えて PEAD を予測しない**（FM 決定的・t=−0.95）。＝overlay の効果は **size/低流動の言い換え**であり
  独立した「外国人アルファ」ではない。docs/50 の中心結論を market-wide breadth で**確定**。
- **頑健に残るのは「高外国人＝PEAD 無/負」**（除外の根拠）。PEAD 自体は 2016+ で弱く**認定未達**（DSR0.61）。
- **実務メモ**: 条件付けるなら **foreign 版が size 版より maxDD が大幅に良い**（−35% vs −72%＝低外国人は極小型ほど
  偏らない）。リターンでは同等でも**ドローダウン耐性で foreign 条件付けが優位**＝overlay（高外国人除外）の採用根拠を補強。
- レポート: `data/reports/pead_foreign_doublesort.html` / 実装: `examples/research_pead_foreign_doublesort.py`
