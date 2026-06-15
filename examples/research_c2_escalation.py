"""C2 エスカレーション条件の記述統計（事前登録した4条件・docs/11 §後続）。

docs/11 の結論：ブロードな「新規5%→ドリフト」は系統的エッジ無し（中央値超過≈0・テール駆動・
レジーム依存）。仮説：「アクティビストが**入った**」より「**エスカレートした**」シグナルの方が
持続的ドリフトを生み得る。これを **事前に凍結した4条件**（①②③⑤）で一括検証する。

規律（in-sample 選択の回避・docs/11 / HANDOFF）:
- 条件は**検証前に凍結**（このファイルの CONDITIONS が確定版。後から増やさない・条件④は除外）。
- 各条件は purpose フィルタ母集団（コア 359 件・docs/11）の**部分集合**。母集団は不変。
- **厳格 DSR 裁定はしない**。記述統計（中央値 TOPIX 超過・勝率・IQR・サブ期間別・サンプル数）。
- PIT：全条件は**提出日アンカー**で観測可能なもののみ。N<30 は「参考・フォワード待ち」。

凍結4条件（イベント日＝提出日 T+1 始値エントリー）:
  ① 買い増し         : コアfilerの変更報告で保有割合の純増（同一 group×issuer の初回純増）
  ② 複数アクティビスト: 同一issuerに365日以内に2グループ目以降のコア新規（2件目以降の提出日）
  ③ 高保有到達       : 保有割合が初めて ≥10% / ≥20% を超えた報告（新規 or 変更）
  ⑤ スタイル(hard)   : registry style==engagement_hard のコア新規（docs/11 スタイルカットの追認）

usage:
  python examples/research_c2_escalation.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import research_c2_drift as drift                            # noqa: E402
from invest_system.validation.registry import default_registry  # noqa: E402

LVH = "data/edinet/large_holdings.parquet"
CHANGES = "data/edinet/large_holdings_changes.parquet"
HZ = [5, 20, 60]

# ② Phase 2 フォワード事前登録の凍結仕様（docs/12 §3）
SCOPE2 = "c2_escalation_multi"
STRAT2 = "multi_activist_365d"
PARAMS2 = {"window_days": 365, "horizons_td": [20, 60], "ratio_band": [0.05, 0.30],
           "entry": "submit_T+1_open", "metric": "median_topix_excess"}
HYP2 = ("同一 issuer への複数アクティビスト（2グループ目以降・365日以内）の新規大量保有は、"
        "単独参入（中央超過≈0・docs/11）と異なり持続的な TOPIX 超過ドリフトを生む。")
RAT2 = ("2人目の参入は1人目のバリュー命題の独立検証＋経営への圧力増（委任状・株主提案連携、"
        "M&A/非公開化の注目度）。触媒は AGM サイクルで数ヶ月かけ顕在化。選択バイアス（明白な"
        "機会との交絡）は未解明でフォワードで既織り込み度を観察（docs/12 §3.4）。")


def subperiod(year: int) -> str:
    return "2021-2022" if year <= 2022 else "2023-2026"


def core_universe() -> pd.DataFrame:
    """コア母集団（new×重要提案・5–30%）。docs/11 と同一定義。"""
    return drift.load_universe(0.05, 0.30)


def load_changes() -> pd.DataFrame:
    """コアfiler限定の変更報告（010002）。未取得なら空。"""
    if not Path(CHANGES).exists():
        return pd.DataFrame()
    c = pd.read_parquet(CHANGES)
    c = c[c["body_ok"] == True].copy()                       # noqa: E712
    c["submit_dt"] = pd.to_datetime(c["submit_dt"])
    c["sec"] = c["issuer_edinet"].map(drift.edinet_to_sec())
    return c.dropna(subset=["sec"])


def _events(df: pd.DataFrame) -> pd.DataFrame:
    """イベント df（sec/entry_date）→ forward_returns 用の最小スキーマに整形。"""
    e = df.copy()
    e["entry_date"] = pd.to_datetime(e["entry_date"]).dt.normalize()
    e["year"] = e["entry_date"].dt.year
    for c in ("style_grp", "band", "canonical_group"):
        if c not in e.columns:
            e[c] = pd.NA
    return e[["sec", "entry_date", "year", "style_grp", "band", "canonical_group"]]


# --- 4条件のイベント抽出（PIT・提出日アンカー） ---------------------------
def cond_add_on(changes: pd.DataFrame) -> pd.DataFrame:
    """① 買い増し：純増(holding_ratio>prev)の変更報告。group×issuer の初回純増を採る。"""
    if changes.empty:
        return changes
    inc = changes[changes["holding_ratio"] > changes["holding_ratio_prev"]].copy()
    inc["grp"] = inc["canonical_group"].fillna(inc["filer_edinet"])
    inc = inc.sort_values("submit_dt").drop_duplicates(["grp", "issuer_edinet"])
    inc["entry_date"] = inc["submit_dt"]
    return inc


def cond_multi_activist(core: pd.DataFrame) -> pd.DataFrame:
    """② 複数アクティビスト：同一issuerに365日以内に2グループ目以降の新規。2件目以降が event。"""
    c = core.copy()
    c["grp"] = c["canonical_group"].fillna(c["filer_edinet"])
    c = c.sort_values("submit_dt")
    rows = []
    for iss, g in c.groupby("issuer_edinet"):
        seen = []                                            # (group, date)
        for _, r in g.iterrows():
            prior = [d for gp, d in seen if gp != r["grp"]
                     and (r["submit_dt"] - d).days <= 365]
            if prior:                                        # 別グループが既に窓内に居る
                rows.append(r)
            seen.append((r["grp"], r["submit_dt"]))
    out = pd.DataFrame(rows)
    if not out.empty:
        out["entry_date"] = out["submit_dt"]
    return out


def cond_high_ownership(core: pd.DataFrame, changes: pd.DataFrame,
                        thr: float) -> pd.DataFrame:
    """③ 高保有到達：group×issuer で保有割合が初めて thr を超えた報告（新規 or 変更）。"""
    cols = ["canonical_group", "filer_edinet", "issuer_edinet", "submit_dt",
            "holding_ratio"]
    new = core.rename(columns={})[cols] if set(cols) <= set(core.columns) else core[cols]
    parts = [new]
    if not changes.empty:
        parts.append(changes[cols])
    allr = pd.concat(parts, ignore_index=True)
    allr["grp"] = allr["canonical_group"].fillna(allr["filer_edinet"])
    allr["sec"] = allr["issuer_edinet"].map(drift.edinet_to_sec())
    allr = allr.dropna(subset=["sec", "holding_ratio"])
    cross = allr[allr["holding_ratio"] >= thr].sort_values("submit_dt")
    cross = cross.drop_duplicates(["grp", "issuer_edinet"])  # 初回到達のみ
    cross["entry_date"] = cross["submit_dt"]
    return cross


def cond_hard_style(core: pd.DataFrame) -> pd.DataFrame:
    """⑤ スタイル(hard)：engagement_hard のコア新規（docs/11 スタイルカットの追認）。"""
    return core[core["style"] == "engagement_hard"].copy()


# --- 集計（記述統計のみ・中央値主） ----------------------------------------
def summarize(name: str, events: pd.DataFrame, panel: dict, topix) -> list:
    if events is None or events.empty:
        return [{"cond": name, "sub": "—", "N": 0}]
    fr = drift.forward_returns(_events(events), panel, topix)
    fr["sub"] = fr["year"].map(subperiod)
    rows = []
    for sub in ["全体", "2021-2022", "2023-2026"]:
        g = fr if sub == "全体" else fr[fr["sub"] == sub]
        rec = {"cond": name, "sub": sub, "N": len(g)}
        for h in HZ:
            x = g[f"x{h}"].dropna() if f"x{h}" in g else pd.Series(dtype=float)
            rec[f"medx{h}"] = round(x.median() * 100, 2) if len(x) else np.nan
            rec[f"win{h}"] = round((x > 0).mean() * 100, 0) if len(x) else np.nan
            rec[f"n{h}"] = len(x)
        rows.append(rec)
    return rows


def preregister_cond2() -> None:
    """② を Phase 2 フォワードスリーブとして事前登録（冪等・結果は将来 record）。

    status=preregistered ＝ trial_count（K）に算入しない＝a-priori 理論の凍結のみ。
    既に同一 scope/strategy で preregistered があれば再登録しない（重複防止）。
    """
    reg = default_registry()
    exists = reg._conn.execute(
        "SELECT 1 FROM trials WHERE scope=? AND strategy_id=? AND status='preregistered'",
        (SCOPE2, STRAT2)).fetchone()
    if exists:
        print(f"② 既に事前登録済み（scope={SCOPE2}）→ スキップ")
        return
    uid = reg.preregister(scope=SCOPE2, strategy_id=STRAT2, params=PARAMS2,
                          hypothesis=HYP2, economic_rationale=RAT2)
    print(f"② を事前登録（scope={SCOPE2} / uuid={uid[:8]}… / status=preregistered）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preregister", action="store_true",
                    help="② をフォワードスリーブとして registry に事前登録（docs/12 §3）")
    args = ap.parse_args()
    if args.preregister:
        preregister_cond2()
    core = core_universe()
    changes = load_changes()
    print(f"コア母集団 {len(core)} 件 / 変更報告(コアfiler) {len(changes)} 件"
          + ("" if not changes.empty else "  ※未取得＝①③は変更分を欠く"))

    conds = {
        "① 買い増し": cond_add_on(changes),
        "② 複数アクティビスト": cond_multi_activist(core),
        "③ 高保有≥10%": cond_high_ownership(core, changes, 0.10),
        "③ 高保有≥20%": cond_high_ownership(core, changes, 0.20),
        "⑤ hard style": cond_hard_style(core),
    }
    secs = set(core["sec"])
    for e in conds.values():
        if e is not None and not e.empty and "sec" in e:
            secs |= set(e["sec"].dropna())
    start = "20211201"
    panel = drift.price_panel(secs, start, pd.Timestamp.today().strftime("%Y%m%d"))
    topix = drift.topix_series()

    # ベースライン（ブロード母集団・docs/11）
    base = drift.forward_returns(_events(core), panel, topix)
    print("\n=== ベースライン（ブロード・docs/11 再掲） ===")
    for h in HZ:
        x = base[f"x{h}"].dropna()
        print(f"  x{h}: 中央超過 {x.median()*100:+.2f}% / 勝率 {(x>0).mean()*100:.0f}% / n={len(x)}")

    print("\n=== エスカレーション条件（中央TOPIX超過% / 勝率% / n・サブ期間別） ===")
    print(f"  {'条件':22s}{'期間':12s}{'N':>4} "
          + " ".join([f"{'x'+str(h):>16}" for h in HZ]))
    for name, ev in conds.items():
        for r in summarize(name, ev, panel, topix):
            if r["N"] == 0 and r["sub"] == "—":
                print(f"  {name:22s}{'(イベント無し/未取得)':12s}")
                continue
            cells = " ".join(
                [f"{r.get(f'medx{h}'):>7}/{r.get(f'win{h}'):>3.0f}%/{r.get(f'n{h}'):>3}"
                 if pd.notna(r.get(f'medx{h}')) else f"{'—':>16}" for h in HZ])
            tag = "  ※参考(N<30)" if r["N"] < 30 and r["sub"] == "全体" else ""
            print(f"  {name:22s}{r['sub']:12s}{r['N']:>4} {cells}{tag}")
    print("\n注：記述統計のみ（厳格DSRなし・確定事項3）。中央値主（平均はテール駆動・docs/11実証）。"
          "母集団不変・PIT提出日アンカー。Phase2フォワード事前登録の素材。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
