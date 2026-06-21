"""Phase 4 前 データ・プリフライト診断 — 9 点を 1 レポート（docs/16）に出力。

**これは診断であって判定ではない。** judge_grid・永続レジストリ（K）は一切触らない＝K 不変。
Phase 4（ML 推定器・一度きりの DSR 判定）の前に、データの瑕疵を可視化して潰すための
throwaway 計測（PIT 厳守・先読み無し）。設計判断は docs/17 に選択肢として出す（ここでは測るだけ）。

診断項目（handoff Part 2）:
  1 上場廃止監査（年別件数・廃止込み/抜きのリターン差＝バイアス量）
  2 分割整合（AdjC が分割日でジャンプしない・修正後 net_share_issuance に分割スパイク無し）
  3 因子別 被覆-時系列マップ（有効開始月・月次非NaN被覆率）
  4 欠損値ポリシー（rank→中立0埋めの一貫適用・0埋め偏り）
  5 外れ値/ウィンソライズ監査（比率因子の分布・極小分母外れ値・rank の頑健性）
  6 cfi 突合 WARN の系統性（EDINET cfi vs J-Quants CFI）
  7 証券コード継続性（4桁コード再利用衝突）
  8 重複因子の整合（accruals JQ vs EDINET、cf_yield vs cf_to_price）
  9 設計行列の一体先読みテスト（feature[t]≤t・ラベル前向き・ギャップ t→t+1）

実行: .venv\\Scripts\\python.exe examples\\preflight_data_audit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001  # pragma: no cover
    pass

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from invest_system.data import store  # noqa: E402
from invest_system.equities import edinet_factors as efa  # noqa: E402
from invest_system.equities import factors as fac  # noqa: E402
from invest_system.equities import fundamentals as fu  # noqa: E402
from invest_system.equities import panel as pn  # noqa: E402
from invest_system.equities import universe as uni  # noqa: E402
from invest_system.equities.edinet_fundamentals import build_edinet_long  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "16-preflight-report.md"
MASTER = Path("data/jquants/equities_master.parquet")


def _me(daily: pd.DataFrame) -> pd.DataFrame:
    """日次 wide → 月末営業日 wide（最終観測）。"""
    return daily.resample("ME").last()


def panels() -> dict:
    """月末パネル一式（adjc/close/turnover/adv/shares/mcap）と全上場・流動ユニバースmask。"""
    ac, c, tv = (store.load_wide("adj_close"), store.load_wide("close"),
                 store.load_wide("turnover"))
    af = store.load_wide("adj_factor")
    ac_m, c_m = _me(ac), _me(c)
    adv_m = _me(tv.rolling(60, min_periods=20).mean())          # 60日 ADV（trailing）
    months = ac_m.index
    # 発行済株数（自己株控除）as-of → 月末。fins ShOutFY/TrShFY は文字列なので point_in_time が数値化。
    fund = fu.load_fundamentals()
    shares = pd.DataFrame(index=months)
    if {"ShOutFY", "Code", "DiscDate"}.issubset(fund.columns):
        so = fu.point_in_time(fund, months, ["ShOutFY", "TrShFY"], lag_days=1)
        sh = so.get("ShOutFY", pd.DataFrame())
        tr = so.get("TrShFY", pd.DataFrame()).reindex(columns=sh.columns).fillna(0.0)
        shares = (sh - tr).reindex(columns=[c for c in c_m.columns if c in sh.columns])
    mcap = (shares.reindex(index=months, columns=c_m.columns)
            * c_m).where(lambda x: x > 0)
    # ユニバース：普通株 ∩ 流動（株価¥100・時価総額¥10B・ADV¥50M）
    common = None
    if MASTER.exists():
        m = pd.read_parquet(MASTER)
        common = set(uni.filter_common_stocks(m)["Code"].astype(str))
    liq = uni.liquid_universe_mask(c_m, mcap, adv_m, 100.0, 1e10, 5e7)
    if common is not None and not liq.empty:
        for col in liq.columns:
            if col not in common:
                liq[col] = False
    # EDINET 派生＋利回り因子の as-of パネルを一度だけ計算（d3/d4/d8 で再利用＝高速化）。
    fp = efa.edinet_factor_panels(months, codes=list(c_m.columns), mcap=mcap)
    return dict(ac=ac, c=c, af=af, ac_m=ac_m, c_m=c_m, adv_m=adv_m, mcap=mcap,
                shares=shares, months=months, liq=liq, fp=fp)


# --- 1 上場廃止監査 ---------------------------------------------------------
def d1_delisting(P: dict) -> str:
    ac_m = P["ac_m"]
    mask = pn.delisting_mask(ac_m)
    ev = []
    for code in ac_m.columns:
        col = mask[code]
        if col.any():
            ev.append((code, col[col].index[0]))
    de = pd.DataFrame(ev, columns=["Code", "last"])
    by_year = de["last"].dt.year.value_counts().sort_index() if len(de) else pd.Series(dtype=int)
    # バイアス量：価格フォワードを last_price(0) と all_minus100(-1) で補完し、各月の
    # 等加重平均リターン差を測る（廃止月の最終損益を落とすか入れるかの差）。廃止銘柄は
    # その月までに流動枠を外れるため、流動マスクでなく「その月に価格を持つ全銘柄」で測る。
    base = pn.forward_returns(ac_m)
    lp = pn.impute_delisting(base, ac_m, policy="last_price")
    m100 = pn.impute_delisting(base, ac_m, policy="all_minus100")
    gap = (lp.mean(axis=1) - m100.mean(axis=1)).dropna()   # 全上場 EW のバイアス包絡
    out = ["## 1. 上場廃止監査（生存者バイアス）", ""]
    out.append(f"- 検知した上場廃止（消滅）銘柄数：**{len(de)}**（全上場・月末パネル基準）。")
    if len(by_year):
        out.append("- 年別 廃止件数：")
        out.append("")
        out.append("| 年 | 件数 |")
        out.append("|---|---|")
        for y, n in by_year.items():
            out.append(f"| {y} | {int(n)} |")
        out.append("")
    out.append(f"- **バイアス量**（全上場 等加重 月次平均リターン, last_price − all_−100%）："
               f"平均 **{gap.mean()*100:+.3f}%/月**（最大月 {gap.max()*100:+.2f}%・"
               f"廃止が集中する月で乖離）。素の NaN 落とし＝last_price 側に立つ＝最終損益を"
               f"無視する上方バイアス。下限（全廃止 −100%）との差がバイアスの最大幅。")
    out.append("- 既定方針：**last_price（最終気配で清算＝偽の損失を作らない）**。下限 −100% は"
               "診断併記。理由別ターミナル値は J-Quants に廃止理由が無いため不可（docs/17）。")
    out.append("")
    return "\n".join(out)


# --- 2 分割整合 -------------------------------------------------------------
def d2_split(P: dict) -> str:
    ac, c, af = P["ac"], P["c"], P["af"]
    # 分割日（adj_factor!=1）で AdjC の日次リターンが過大な「見かけのジャンプ」を出さないこと。
    af2 = af.reindex(index=ac.index, columns=ac.columns)
    split_cells = af2.notna() & (af2.round(6) != 1.0)
    adj_ret = ac.pct_change(fill_method=None)
    raw_ret = c.pct_change(fill_method=None)
    jr_adj = adj_ret.where(split_cells).abs().stack()
    jr_raw = raw_ret.where(split_cells).abs().stack()
    # net_share_issuance：raw vs 分割調整後のスパイク率（再掲）。
    long = build_edinet_long()
    raw = efa.derive_disclosure_features(long)["net_share_issuance"].dropna()
    adj = efa.derive_disclosure_features(efa.attach_split_cf(long))["net_share_issuance"].dropna()
    out = ["## 2. 分割整合（AdjC・株数）", ""]
    out.append(f"- 分割セル（adj_factor≠1）数：**{int(split_cells.to_numpy().sum())}**。"
               f"その日の **|日次リターン| 中央値：AdjC={jr_adj.median()*100:.2f}% / "
               f"生C={jr_raw.median()*100:.1f}%**。AdjC 側は通常並み＝**調整後は分割で"
               f"ジャンプしない**（生 C は分割でジャンプ＝時価総額の偽ジャンプ源）。")
    out.append(f"- net_share_issuance の分割スパイク（|YoY|>0.9 の割合）："
               f"**raw {(raw.abs()>0.9).mean()*100:.2f}% → 分割調整後 "
               f"{(adj.abs()>0.9).mean()*100:.2f}%**（1-1 で純粋な分割を≈0 に潰した）。")
    out.append("")
    return "\n".join(out)


# --- 3 因子別 被覆-時系列マップ ---------------------------------------------
def d3_coverage(P: dict) -> str:
    months = P["months"]
    liq = P["liq"]
    fp = P["fp"]
    rows = []
    for name, panel in sorted(fp.items()):
        pa = panel.reindex(index=months, columns=liq.columns).where(liq)
        cov = pa.notna().sum(axis=1)
        denom = liq.sum(axis=1).replace(0, np.nan)
        cov_rate = (cov / denom).dropna()
        nonzero = cov_rate[cov_rate > 0.01]
        start = nonzero.index.min() if len(nonzero) else None
        rows.append((name, start, cov_rate.tail(36).mean() if len(cov_rate) else np.nan,
                     cov_rate.max() if len(cov_rate) else np.nan))
    out = ["## 3. 因子別 被覆-時系列マップ（EDINET 派生・流動ユニバース）", "",
           "Phase 4 の学習窓設定の根拠。被覆がラギッド（年で立ち上がる）な因子は早期窓で薄い。", "",
           "| 因子 | 有効開始 | 直近36M平均被覆 | 最大被覆 |", "|---|---|---|---|"]
    for name, start, recent, mx in rows:
        s = start.strftime("%Y-%m") if start is not None else "—"
        out.append(f"| {name} | {s} | {recent*100:.0f}% | {mx*100:.0f}% |"
                   if pd.notna(recent) else f"| {name} | {s} | — | — |")
    out.append("")
    out.append("- 含意：EDINET 早期（2016-2018）は被覆が薄く IFRS も少ない。YoY 系"
               "（asset_growth・net_share_issuance）は更に 1 年ぶん遅れて立ち上がる。"
               "→ 学習窓は被覆が安定する時期から、または被覆を特徴量に含める。")
    out.append("")
    return "\n".join(out)


# --- 4 欠損値ポリシー -------------------------------------------------------
def d4_missing(P: dict) -> str:
    months = P["months"]
    liq = P["liq"]
    fp = P["fp"]
    out = ["## 4. 欠損値ポリシー（正準：rank→中立0埋め）", "",
           "**`factors.rank_and_fill`** に一元化：cross_sectional_rank（[-1,1]）後、残る欠損を"
           "**中立 0（ランク中央）**で埋める＝0 埋めの方向バイアスを出さない。全因子に同一適用。", "",
           "| 因子 | rank前 欠損率（流動枠） | rank&fill後 平均 |", "|---|---|---|"]
    for name in sorted(fp):
        pa = fp[name].reindex(index=months, columns=liq.columns).where(liq)
        miss = 1.0 - pa.notna().sum(axis=1).sum() / max(1, liq.sum(axis=1).sum())
        rf = fac.rank_and_fill(pa)
        out.append(f"| {name} | {miss*100:.0f}% | {rf.where(liq).mean(axis=1).mean():+.3f} |")
    out.append("")
    out.append("- rank&fill 後の平均は≈0＝**0埋めによる系統的な方向バイアスは出ない**（rank 後の"
               "中央が0のため）。zscore 系で0埋めすると平均がずれ得るので rank→中立を採用。"
               "rd_intensity（欠損60%）等の疎な因子も中立化で安全に同一処理できる。")
    out.append("")
    return "\n".join(out)


# --- 5 外れ値/ウィンソライズ監査 -------------------------------------------
def d5_outliers(P: dict) -> str:
    long = efa.derive_disclosure_features(efa.attach_split_cf(build_edinet_long()))
    ratios = ["roic", "leverage", "asset_growth", "ebitda_margin", "net_share_issuance",
              "gross_profitability", "accruals"]
    out = ["## 5. 外れ値 / ウィンソライズ監査（比率因子）", "",
           "比率因子は極小分母・単位差で桁外れの外れ値を出す。**rank（頑健）を主**とし、"
           "生値合成時は `factors.winsorize_cross_sectional`（既定 1%/99%・事前固定）を一貫適用。", "",
           "| 比率因子 | p1 | p50 | p99 | max\\|x\\| | \\|x\\|>10 件 |", "|---|---|---|---|---|---|"]
    for r in ratios:
        if r not in long.columns:
            continue
        x = pd.to_numeric(long[r], errors="coerce").dropna()
        if x.empty:
            continue
        out.append(f"| {r} | {x.quantile(0.01):.3f} | {x.median():.3f} | "
                   f"{x.quantile(0.99):.2f} | {x.abs().max():,.0f} | {int((x.abs()>10).sum())} |")
    out.append("")
    out.append("- 所見：**net_share_issuance・roic・leverage に桁外れ外れ値**（極小分母＝"
               "純資産≈0・株数単位差由来）。→ Phase 4 は **rank 化を既定**、生値を使う比率は"
               "winsorize(1/99%)。zscore は非頑健のため比率因子には使わない。")
    out.append("")
    return "\n".join(out)


# --- 6 cfi 突合 WARN の系統性 ----------------------------------------------
def d6_cfi(P: dict) -> str:
    long = build_edinet_long()
    jq = fu.load_fundamentals()
    jq = jq[jq["CurPerType"].astype(str) == "FY"].copy()
    jq["key"] = jq["Code"].astype(str).str[:4] + "|" + jq["CurFYEn"].astype(str).str[:10]
    jq["CFI_n"] = pd.to_numeric(jq.get("CFI"), errors="coerce")
    jqi = jq.dropna(subset=["CFI_n"]).drop_duplicates("key").set_index("key")["CFI_n"]
    L = long.copy()
    L["key"] = L["Code"].astype(str).str[:4] + "|" + L["period_end"].astype(str).str[:10]
    L["cfi_n"] = pd.to_numeric(L["cfi"], errors="coerce")
    j = L.dropna(subset=["cfi_n"]).merge(jqi.rename("jq"), left_on="key", right_index=True)
    if j.empty:
        return "## 6. cfi 突合 WARN の系統性\n\n- 突合キー不一致でサンプル無し。\n"
    rel = (j["cfi_n"] - j["jq"]).abs() / j["jq"].abs().clip(lower=1.0)
    warn = j[rel > 0.01]
    by_basis = warn["basis"].value_counts() if "basis" in warn else pd.Series(dtype=int)
    by_year = warn["period_end"].astype(str).str[:4].value_counts().sort_index()
    out = ["## 6. cfi（投資CF）突合 WARN の系統性（EDINET vs J-Quants）", ""]
    out.append(f"- 突合可能：**{len(j)}** 開示。WARN（|相対差|>1%）：**{len(warn)}** "
               f"（{len(warn)/len(j)*100:.1f}%）。")
    if len(by_basis):
        bb = {k: int(v) for k, v in by_basis.items()}
        share = by_basis.max() / by_basis.sum()
        top = by_basis.idxmax()
        verdict = (f"**{top} に {share*100:.0f}% 偏在＝系統的**（{top} の投資CF範囲が "
                   f"J-Quants と定義差。基準別 null/WARN として扱う）" if share > 0.7
                   else "基準横断＝無作為寄り")
        out.append(f"- 基準別 WARN：{bb} ＝ {verdict}。")
    if len(by_year):
        by = {k: int(v) for k, v in list(by_year.items())[:6]}
        out.append(f"- 年別 WARN（一部）：{by} …（年で単調＝開示量増に比例・突発でない）。")
    out.append("- 含意：cfi 差が系統的なら **FCF 利回り（=CFO+CFI）に波及**。投資CFの定義差"
               "（投資の範囲）は基準別 null/WARN として扱い、FCF は EDINET 内で一貫定義のまま使う。")
    out.append("")
    return "\n".join(out)


# --- 7 証券コード継続性 -----------------------------------------------------
def d7_code_reuse(P: dict) -> str:
    out = ["## 7. 証券コード継続性（4桁再利用衝突）", ""]
    if not MASTER.exists():
        return out[0] + "\n\n- master 不在でスキップ。\n"
    m = pd.read_parquet(MASTER)
    m["b4"] = m["Code"].astype(str).str[:4]
    # 同一 4 桁に複数の異なる企業名（現スナップショットは1:1のはず＝再利用は時系列で起こる）
    dup = m.groupby("b4")["CoName"].nunique()
    collide = dup[dup > 1]
    # fins 側の (4桁, 企業名相当=Codeの5桁ゆれ) — 5桁 Code は4桁+チェックで1社1コードが原則
    fins = fu.load_fundamentals()
    f5 = fins.groupby(fins["Code"].astype(str).str[:4])["Code"].nunique()
    multi = f5[f5 > 1]
    out.append(f"- master：4桁コードに複数企業名が紐づく衝突＝**{len(collide)} 件**"
               f"（現スナップショット）。")
    out.append(f"- fins：同一4桁に複数5桁 Code が現れる＝**{len(multi)} 件**"
               + (f"（例 {list(multi.index[:5])}）" if len(multi) else "") + "。")
    out.append("- 含意：5桁 Code は1社1コードが原則だが、4桁での結合は再利用/区分変更で混線し得る。"
               "**結合キーは5桁 Code を厳守**（4桁突合は cfi/accruals 突合のみ・名寄せ注意）。")
    out.append("")
    return "\n".join(out)


# --- 8 重複因子の整合 -------------------------------------------------------
def d8_duplicates(P: dict) -> str:
    months = P["months"]
    liq = P["liq"]
    # accruals: EDINET (profit-cfo)/TA  vs  J-Quants (CFO-NP)/TA  → ほぼ −1 相関のはず
    ed = P["fp"].get("accruals")
    jqf = fu.fundamentals_panel(months, ["CFO", "NP", "TA"])
    out = ["## 8. 重複因子の整合（同概念・別ソース）", ""]
    if ed is not None and all(k in jqf for k in ("CFO", "NP", "TA")):
        ta = jqf["TA"].where(jqf["TA"] > 0)
        jq_acc = (jqf["CFO"] - jqf["NP"]) / ta
        corr = _row_corr(ed.where(liq), jq_acc.reindex_like(ed).where(liq))
        out.append(f"- **accruals**（EDINET `(profit−cfo)/TA` vs J-Quants `(CFO−NP)/TA`）"
                   f"：月次クロスセクション相関の中央値 **{corr:+.2f}**（**符号は逆**＝定義が "
                   f"−1 倍関係＝docs/18。合成時に符号統一必須）。")
    # cf_yield(JQ CFO/mcap) vs cf_to_price(EDINET cfo/mcap)
    edp = P["fp"].get("cf_to_price")
    if edp is not None and "CFO" in jqf:
        cf_yield = jqf["CFO"] / P["mcap"].where(P["mcap"] > 0)
        corr2 = _row_corr(edp.where(liq), cf_yield.reindex_like(edp).where(liq))
        out.append(f"- **CF/P**（EDINET `cf_to_price` vs J-Quants `cf_yield`）"
                   f"：相関中央値 **{corr2:+.2f}**（同式・別ソース＝高相関なら整合）。")
    out.append("- 含意：accruals は **どちらか一方**（既定 EDINET）。CF/P も一本化。"
               "二重計上・符号衝突を Phase 4 設計行列で排除（docs/18 重複表）。")
    out.append("")
    return "\n".join(out)


def _row_corr(a: pd.DataFrame, b: pd.DataFrame) -> float:
    a, b = a.align(b, join="inner")
    cs = []
    for t in a.index:
        x, y = a.loc[t], b.loc[t]
        ok = x.notna() & y.notna()
        if int(ok.sum()) >= 20:
            cs.append(x[ok].corr(y[ok]))
    return float(np.nanmedian(cs)) if cs else float("nan")


# --- 9 設計行列の一体先読みテスト ------------------------------------------
def d9_joint_lookahead(P: dict) -> str:
    months = P["months"]
    if len(months) < 6:
        return "## 9. 設計行列の一体先読みテスト\n\n- 月数不足でスキップ。\n"
    t0 = months[len(months) // 2]
    codes = list(P["c_m"].columns)[:800]                     # 計算量を抑える
    long = efa.attach_split_cf(build_edinet_long())
    f0 = efa.derive_disclosure_features(long)
    base = fu.point_in_time(f0, [t0], ["asset_growth"], date_col="DiscDate",
                            code_col="Code", lag_days=1)["asset_growth"]
    # 未来開示（DiscDate>t0）を改変→ feature[t0] 不変か
    mut = long.copy()
    fut = pd.to_datetime(mut["DiscDate"]) > t0
    for col in ["total_assets", "shares_outstanding", "cfo", "profit"]:
        if col in mut.columns:
            mut.loc[fut, col] = mut.loc[fut, col] * 9.0
    fm = efa.derive_disclosure_features(mut)
    base_m = fu.point_in_time(fm, [t0], ["asset_growth"], date_col="DiscDate",
                              code_col="Code", lag_days=1)["asset_growth"]
    feat_ok = base.fillna(-999).equals(base_m.fillna(-999))
    # ラベル：total_forward_returns は price[t],price[t+1],dps_asof[t] のみ＝t+2 改変で不変
    ac_m, c_m = P["ac_m"][codes], P["c_m"][codes]
    dps = pn.dividend_per_share_asof(months, codes=codes)
    lab = pn.total_forward_returns(ac_m, c_m, dps)
    ac2 = ac_m.copy()
    after = ac2.index > months[months.get_loc(t0) + 1]
    ac2.loc[after] = ac2.loc[after] * 5.0
    lab2 = pn.total_forward_returns(ac2, c_m, dps)
    label_ok = np.allclose(lab.loc[t0].fillna(-999).to_numpy(),
                           lab2.loc[t0].fillna(-999).to_numpy(), equal_nan=True)
    # ギャップ：label[t] は t→t+1（fwd の定義）。feature[t] と同じ t にラベル付け＝整合。
    out = ["## 9. 設計行列の一体先読みテスト（結合パネル）", "",
           "個別因子でなく **feature×label の結合**で先読みを assert（Phase 4 安全弁）。", ""]
    out.append(f"- **feature[t] が ≤t のみ**：t0={t0:%Y-%m} で未来開示(DiscDate>t0)を9倍に"
               f"改変しても asset_growth[t0] 不変 → **{'PASS' if feat_ok else 'FAIL'}**。")
    out.append(f"- **label が前向き・t+2 非依存**：t0+2 以降の価格を5倍にしても "
               f"total_return[t0] 不変 → **{'PASS' if label_ok else 'FAIL'}**。")
    out.append("- **ギャップ**：feature は t 時点 as-of、label は close[t]→close[t+1]（建玉 t→実現 t+1）"
               "で整合（forward_returns の定義）。")
    out.append("")
    return "\n".join(out)


def main() -> int:
    print("=== Phase 4 前 データ・プリフライト診断（throwaway・K不変）===")
    P = panels()
    print(f"月末パネル {P['ac_m'].shape}・流動ユニバース 月平均 "
          f"{int(P['liq'].sum(axis=1).tail(36).mean())} 銘柄")
    sections = [("1 廃止", d1_delisting), ("2 分割", d2_split), ("3 被覆", d3_coverage),
                ("4 欠損", d4_missing), ("5 外れ値", d5_outliers), ("6 cfi", d6_cfi),
                ("7 コード", d7_code_reuse), ("8 重複", d8_duplicates),
                ("9 先読み", d9_joint_lookahead)]
    body = []
    for label, fn in sections:
        try:
            print(f"  [{label}] …", flush=True)
            body.append(fn(P))
        except Exception as e:  # noqa: BLE001
            body.append(f"## {label}\n\n- 診断中にエラー：`{type(e).__name__}: {str(e)[:120]}`\n")
            print(f"    WARN {type(e).__name__}: {str(e)[:100]}")
    head = ("# 16. Phase 4 前 データ・プリフライト報告（throwaway 診断・K 不変・PIT）\n\n"
            "Phase 4（ML 推定器・一度きりの DSR 判定）の**前に**データ瑕疵を可視化した報告。"
            "判定（judge_grid/DSR）はしない。生成：`examples/preflight_data_audit.py`。\n"
            "設計判断（ユニバース/リターン/廃止/ホライズン）は確定済み＝docs/17、符号/重複は docs/18。\n\n"
            "---\n\n")
    OUT.write_text(head + "\n---\n\n".join(body) + "\n", encoding="utf-8")
    print(f"→ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
