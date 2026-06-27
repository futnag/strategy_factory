# 22. Phase 4b 結果 ― 小型寄りユニバース GKX ML・一度きりの DSR 判定

事前登録 docs/21（FROZEN）に従い **一度だけ** 実行（2026-06-21）。**唯一の変更＝mcap 床 ¥100億→¥30億**（preset `smallcap_30b`）。他は Phase 4 と同一。
CPCV k=2・φ=5・N=6 族デフレート・gap_after=12ヶ月。

---

## 0. Phase 4 との並置

| 項目 | Phase 4（¥10B） | Phase 4b（¥30B） |
|---|---|---|
| 月平均銘柄数 | ~1880 | ~1563 |
| 採用族 | rf | rf |
| 平均パス SR(月) | +0.208 | +0.235 |
| Deflated SR | 0.905 | 0.883 |
| 判定 | FAIL | FAIL |

## 1. 族別 OOS（CPCV φ=5・2018-01+・15bps・廃止 last-price）

| 族 | OOS R² | meanIC | 平均パスSR(月) | 年率SR | パスSR範囲 | 回転率 |
|---|---|---|---|---|---|---|
| ols | -0.1041 | +0.023 | +0.003 | +0.01 | [-0.13,+0.12] | 2.84 |
| lasso | -0.0001 | +0.025 | -0.002 | -0.01 | [-0.10,+0.13] | 1.84 |
| elasticnet | -0.0001 | +0.025 | -0.001 | -0.00 | [-0.10,+0.13] | 1.84 |
| rf | +0.0017 | +0.062 | +0.235 | +0.82 | [+0.13,+0.36] | 1.74 |
| hgbrt | +0.0010 | +0.050 | +0.225 | +0.78 | [+0.16,+0.29] | 2.37 |
| mlp | -0.0133 | +0.049 | +0.125 | +0.43 | [+0.04,+0.22] | 2.56 |

- **線形 vs ML 増分**：線形最良 R²=-0.0001／ML 最良 R²=+0.0017（ML 超え：True）。ネット Sharpe 線形最良=+0.062／ML 最良=+0.334（ML 超え：True）。

## 2. 判定（deflated DSR・一度きり）

- **採用族**：`rf`。
- 平均パス Sharpe = **+0.235/月**（年率 +0.82・n=101ヶ月）。
- **V[SR]（φ=5 パス間分散）= 0.0071**。
- **Deflated Sharpe（N=6）= 0.883**。
- **廃止封筒**：last-price +0.235 ↔ 全−100% +0.226。
- **コスト感応度（採用族・30bps 片道）**：平均パス SR = **+0.173/月**。
- 合否：(i)DSR≥0.95=False ∧ (ii)ネット>0=True ∧ (iii)封筒>0=True → **FAIL**。

## 3. 正直な結論

**FAIL。** 小型寄りユニバースでも認定基準を越えない＝**正当な FAIL**。床以外の変更・再判定はしない（Handoff_phase4b §5）。

## 4. 小型固有診断（throwaway・K 不変）

```
=== Phase 4b 小型ユニバース診断（throwaway・K不変）===

[1] ユニバース規模
  production (¥10B): avg 1421 銘柄/月
  smallcap_30b (¥30B): avg 1563 銘柄/月
  追加銘柄（30Bのみ）: avg +142/月, max +240

[2] 廃止バイアス（等加重月次リターン差 last_price−(−100%)）
  production: mean +0.000%/月, max +0.000%/月 (119 months)
  smallcap_30b: mean +0.000%/月, max +0.000%/月 (119 months)

[3] EDINET 因子被覆（ユニバース内・非NaN率）
  fcf_yield              prod=89.5%  sc=87.6%  Δ=-1.9pp
  roic                   prod=83.4%  sc=82.0%  Δ=-1.4pp
  asset_growth           prod=73.0%  sc=71.0%  Δ=-2.0pp
  accruals               prod=89.5%  sc=87.6%  Δ=-1.9pp
  gross_profitability    prod=78.4%  sc=77.3%  Δ=-1.1pp
  ebitda_margin          prod=81.0%  sc=79.7%  Δ=-1.3pp
  leverage               prod=88.2%  sc=86.4%  Δ=-1.8pp
  net_share_issuance     prod=78.6%  sc=77.6%  Δ=-1.1pp
  rd_intensity           prod=40.2%  sc=38.5%  Δ=-1.7pp
  cf_to_price            prod=89.5%  sc=87.6%  Δ=-1.9pp

[4] 容量概算（participation=5%・デシルL/S近似）
  production: ADV p25=135M p50=351M | cap_est p25=1.4億 p50=3.5億
  smallcap_30b: ADV p25=120M p50=306M | cap_est p25=1.2億 p50=3.1億

→ 判定前診断完了。結果は run_phase4b_judgment.py の docs/22 に併記。
```

## 5. 規律・再現性

- レジストリ scope `phase4b_gkx_judgment` に判定 1 件＝**K +1**。前 scope `phase4_gkx_judgment` は不変。
- seed=20260621・決定的。実行 1404s。
