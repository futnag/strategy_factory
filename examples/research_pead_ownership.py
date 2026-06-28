r"""Jinushi (2023) PEAD × 所有構造 の再現検証（ローカルDB）。

論文: Junko Jinushi, "Post-Earnings Announcement Drift and Ownership Structure in the
Modern Japanese Stock Market," The Japanese Accounting Review 13 (2023), 1-36.
主要主張: (H1) 日本株の PEAD は時間とともに減衰。(H2) ただし**外国人持株比率が低い／個人比率が
高い銘柄では減衰しにくく underreaction が持続**する（所有構造による異質性）。

本スクリプトの再現範囲と正直な限界:
- 期間: ローカル株価は **2016-06〜**（論文の 2002-2020 の前半は再現不可）。よって H1 は
  「2016 以降の窓での減衰／既に弱いか」を検証する（論文の後期と接続する位置づけ）。
- 驚き(SUE): リポジトリ既存の `fundamental_factors.disclosure_features` を再利用＝**本決算の
  実績EPS − 会社予想EPS（株価でスケール＝surprise yield）**。日本は会社予想が標準ベンチマーク。
  → イベント＝各社の **本決算発表（DiscDate）**、年1回・PIT。
- ドリフト: 発表日アンカーの **市場調整CAR**（TOPIX 控除）。窓 [+2,+H]（発表日ジャンプを避け +2 から）。
- 所有構造(H2): `data/edinet/ownership_categories.parquet`（EDINET-DB の所有者別状況・FY2014+。
  列 Code, fiscal_year, foreign_pct(=外国法人等+外国個人), individual_pct(=個人その他)）が在れば
  銘柄別の**持続的**所有タイプ（期間中央値）で層別。無ければ H1 のみ実行し取得手順を表示。

実行: $env:PYTHONUTF8="1"; .\.venv\Scripts\python.exe examples\research_pead_ownership.py
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from invest_system.data.store import load_wide  # noqa: E402
from invest_system.equities import fundamental_factors as ff  # noqa: E402
from invest_system.equities import fundamentals as fu  # noqa: E402

OWN_PATH = "data/edinet/ownership_categories.parquet"
HORIZONS = [21, 42, 63]          # 取引日（≈1/2/3ヶ月）の CAR 窓 [+2, +H]
SKIP = 2                          # 発表日 +SKIP からエントリ（ジャンプ回避・T+1約定現実）
MIN_PRICE = 100.0                 # 低位株除外
LIQ_TOPN = 1000                   # PIT 流動ユニバース（trailing 売買代金 上位N）


def _load_topix(dates: pd.DatetimeIndex) -> pd.Series:
    """TOPIX 終値（log）を株価パネル日付に整列。無ければ全銘柄等加重を代用。"""
    cand = glob.glob("data/jquants/indices/code_0000.parquet")
    if cand:
        ix = pd.read_parquet(cand[0])
        s = ix.set_index(pd.to_datetime(ix["Date"]))["C"].sort_index()
        return np.log(s.reindex(dates).ffill())
    return pd.Series(np.nan, index=dates)


def build_events() -> pd.DataFrame:
    """本決算イベント表: [Code, DiscDate, sue_raw]（PIT・先読みなし）。"""
    long = fu.load_fundamentals()
    out_fy, _ = ff.disclosure_features(long)
    ev = out_fy[["Code", "DiscDate", "sue_recent_raw"]].copy()
    ev["Code"] = ev["Code"].astype(str)
    ev["DiscDate"] = pd.to_datetime(ev["DiscDate"])
    ev = ev.dropna(subset=["DiscDate", "sue_recent_raw"])
    return ev


def attach_car_and_sue(ev: pd.DataFrame, adj: pd.DataFrame, close: pd.DataFrame,
                       turn: pd.DataFrame, topix_log: pd.Series) -> pd.DataFrame:
    """各イベントに 市場調整CAR（各H）と surprise-yield SUE・流動性フラグを付与（ベクトル化）。"""
    dates = adj.index
    codes = list(adj.columns)
    cpos = {c: i for i, c in enumerate(codes)}
    L = np.log(adj.to_numpy(dtype="float64"))            # dates × codes（log adj_close）
    Cl = close.reindex(columns=codes).to_numpy(dtype="float64")
    Lm = topix_log.to_numpy(dtype="float64")
    # trailing 60日 平均売買代金（PIT 流動性）と当日順位
    adv = turn.reindex(columns=codes).rolling(60, min_periods=20).mean()
    A = adv.to_numpy(dtype="float64")

    ev = ev[ev["Code"].map(lambda c: c in cpos)].copy()
    cidx = ev["Code"].map(cpos).to_numpy()
    # 発表日の取引日位置（DiscDate 以降の最初の営業日）
    p = dates.searchsorted(ev["DiscDate"].values, side="left")
    n = len(dates)
    inb = p < n
    ev, cidx, p = ev[inb].copy(), cidx[inb], p[inb]

    # surprise yield = (実績−会社予想 EPS) / 発表日終値
    px = Cl[p, cidx]
    ev["price_disc"] = px
    ev["sue"] = ev["sue_recent_raw"].to_numpy() / np.where(px > 0, px, np.nan)

    # PIT 流動性: 発表日の trailing ADV 順位が上位N、かつ価格≥MIN_PRICE
    adv_row = A[p, cidx]
    ev["adv"] = adv_row
    ev["price_ok"] = px >= MIN_PRICE

    # 市場調整 CAR [+SKIP, +H]
    s = p + SKIP
    for H in HORIZONS:
        e = p + H
        ok = (e < n) & (s < n)
        car = np.full(len(p), np.nan)
        si, ei, ci = s[ok], e[ok], cidx[ok]
        stock = L[ei, ci] - L[si, ci]
        mkt = Lm[ei] - Lm[si]
        car[ok] = stock - mkt
        ev[f"car_{H}"] = car
    ev["year"] = ev["DiscDate"].dt.year
    return ev


def pit_liquid_universe(ev: pd.DataFrame) -> pd.DataFrame:
    """各発表月ごとに trailing ADV 上位N かつ価格OK のイベントだけ残す（PIT・先読みなし）。"""
    ev = ev[ev["price_ok"] & ev["adv"].notna()].copy()
    ev["ym"] = ev["DiscDate"].dt.to_period("M")
    keep = []
    for _, g in ev.groupby("ym"):
        thr = g["adv"].nlargest(min(LIQ_TOPN, len(g))).min()
        keep.append(g[g["adv"] >= thr])
    return pd.concat(keep, ignore_index=True) if keep else ev.iloc[0:0]


def _qsort(s: pd.Series, q: int = 5) -> pd.Series:
    """クロスセクション分位（同値で潰れたら rank ベースに退避）。"""
    try:
        return pd.qcut(s, q, labels=False, duplicates="drop")
    except Exception:  # noqa: BLE001
        return pd.qcut(s.rank(method="first"), q, labels=False)


def pead_table(ev: pd.DataFrame, H: int, by: str | None = None) -> pd.DataFrame:
    """SUE分位ごとの平均CAR と Q5−Q1 スプレッド（t値つき）。by を与えると層別。"""
    car = f"car_{H}"
    d = ev.dropna(subset=[car, "sue"]).copy()
    # 各発表月内で SUE 5分位（クロスセクション・PIT）
    d["q"] = d.groupby(d["DiscDate"].dt.to_period("M"))["sue"].transform(lambda s: _qsort(s, 5))
    d = d.dropna(subset=["q"])
    d["q"] = d["q"].astype(int)
    groups = [(None, d)] if by is None else list(d.groupby(by))
    rows = []
    for gname, g in groups:
        qmean = g.groupby("q")[car].mean() * 100
        lo, hi = g[g["q"] == 0][car], g[g["q"] == g["q"].max()][car]
        spread = (hi.mean() - lo.mean()) * 100
        se = np.sqrt(hi.var(ddof=1) / max(len(hi), 1) + lo.var(ddof=1) / max(len(lo), 1))
        t = spread / (se * 100) if se > 0 else np.nan
        row = {"group": gname, "n": len(g),
               "Q1(low SUE)": round(qmean.get(0, np.nan), 2),
               "Q5(high SUE)": round(qmean.get(g["q"].max(), np.nan), 2),
               "Q5-Q1(PEAD)": round(spread, 2), "t": round(t, 2)}
        rows.append(row)
    return pd.DataFrame(rows)


def decay_table(ev: pd.DataFrame, H: int, by: str | None = None) -> pd.DataFrame:
    """年×(層) の PEAD スプレッド（Q5−Q1, %）＝時間減衰の確認。"""
    car = f"car_{H}"
    d = ev.dropna(subset=[car, "sue"]).copy()
    d["q"] = d.groupby(d["DiscDate"].dt.to_period("M"))["sue"].transform(lambda s: _qsort(s, 5))
    d = d.dropna(subset=["q"]).astype({"q": int})
    keys = ["year"] if by is None else ["year", by]
    out = {}
    for gv, g in d.groupby(keys if by else "year"):
        qmax = g["q"].max()
        spread = (g[g["q"] == qmax][car].mean() - g[g["q"] == 0][car].mean()) * 100
        out[gv] = round(spread, 2)
    if by is None:
        return pd.DataFrame({"year": list(out), "PEAD(Q5-Q1)%": list(out.values())})
    s = pd.Series(out)
    return s.unstack().round(2)  # index=year, col=group


def attach_ownership(ev: pd.DataFrame) -> pd.DataFrame | None:
    """所有構造パネルから銘柄別の持続タイプ（期間中央値）を付与。無ければ None。"""
    if not Path(OWN_PATH).exists():
        return None
    own = pd.read_parquet(OWN_PATH)
    own["Code"] = own["Code"].astype(str)
    med = own.groupby("Code")[["foreign_pct", "individual_pct"]].median()
    ev = ev.merge(med, left_on="Code", right_index=True, how="inner")
    # 銘柄別の持続タイプ（中央値で2分割）
    f_med = ev.groupby("Code")["foreign_pct"].first().median()
    i_med = ev.groupby("Code")["individual_pct"].first().median()
    ev["foreign_grp"] = np.where(ev["foreign_pct"] < f_med, "foreign_LOW", "foreign_HIGH")
    ev["indiv_grp"] = np.where(ev["individual_pct"] >= i_med, "indiv_HIGH", "indiv_LOW")
    return ev


def main() -> int:
    print("=== Jinushi(2023) PEAD×所有構造 再現（ローカルDB・2016+）===")
    adj = load_wide("adj_close")
    close = load_wide("close")
    turn = load_wide("turnover")
    topix_log = _load_topix(adj.index)
    print(f"価格パネル: {adj.shape[0]}日 × {adj.shape[1]}銘柄  {adj.index.min():%Y-%m}..{adj.index.max():%Y-%m}"
          f"  TOPIX={'有' if topix_log.notna().any() else '無(等加重代用)'}")

    ev = build_events()
    print(f"本決算イベント(生): {len(ev):,}")
    ev = attach_car_and_sue(ev, adj, close, turn, topix_log)
    ev = pit_liquid_universe(ev)
    print(f"PIT流動ユニバース後イベント: {len(ev):,}  期間 {ev['year'].min()}..{ev['year'].max()}")

    # ---- H1: PEAD 存在 と 時間減衰 ----
    for H in HORIZONS:
        print(f"\n## H1  PEAD（市場調整CAR [+{SKIP},+{H}]日, SUE5分位, %, 全期間）")
        print(pead_table(ev, H).to_string(index=False))
    print(f"\n## H1  PEAD スプレッド(Q5−Q1, %) の年次推移  [+{SKIP},+{HORIZONS[1]}]日 ＝減衰確認")
    print(decay_table(ev, HORIZONS[1]).to_string(index=False))

    # ---- H2-proxy: サイズ/流動性プロキシ（全サンプル・ローカルのみ）----
    # 外国人持株比率は size/流動性と強相関（大型=高外国人、小型=個人主体）。実所有データ取得前に
    # 全サンプルで機構を検証＝小型/個人主体ほど PEAD が強く持続するか（論文 H2 の含意）。
    H = HORIZONS[1]
    ev["adv_grp"] = ev.groupby(ev["DiscDate"].dt.to_period("M"))["adv"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 3,
                          labels=["small_retail", "mid", "large_foreign"]))
    print(f"\n## H2-proxy  サイズ/流動性プロキシ層別 PEAD（[+{SKIP},+{H}]日, %, 全サンプル）")
    print("   （論文の含意: small_retail=低外国人/高個人 ほど PEAD 大・持続）")
    print(pead_table(ev, H, by="adv_grp").to_string(index=False))
    print(f"### 年次減衰（プロキシ層別・PEAD Q5−Q1 %）")
    print(decay_table(ev, H, by="adv_grp").to_string())

    # ---- H2-actual: 実所有構造による異質性（EDINET-DB 取得パネルが在れば）----
    evo = attach_ownership(ev)
    if evo is None:
        print(f"\n## H2  所有構造パネル `{OWN_PATH}` が未作成 → H2 はスキップ。")
        print("   取得手順: EDINET-DB の get_shareholder_categories(edinet_code) で")
        print("   foreign_pct(=外国法人等+外国個人)・individual_pct(=個人その他) を銘柄別に集め、")
        print(f"   列 [Code, fiscal_year, foreign_pct, individual_pct] で {OWN_PATH} に保存。")
        # 取得対象の候補（直近の流動上位銘柄コード）を出力
        recent = ev[ev["year"] >= ev["year"].max() - 1]
        top = (recent.groupby("Code")["adv"].mean().sort_values(ascending=False)
               .head(300).index.tolist())
        Path("data/_pead_universe_codes.txt").write_text("\n".join(top), encoding="utf-8")
        print(f"   → 取得候補 {len(top)} コードを data/_pead_universe_codes.txt に出力。")
        return 0

    print(f"\n## H2  所有構造で層別（銘柄数: foreign {evo['Code'].nunique()}）")
    H = HORIZONS[1]
    for by in ("foreign_grp", "indiv_grp"):
        print(f"\n### 層別 PEAD（{by}・[+{SKIP},+{H}]日, %）")
        print(pead_table(evo, H, by=by).to_string(index=False))
        print(f"### 年次減衰（{by}・PEAD Q5−Q1 %）")
        print(decay_table(evo, H, by=by).to_string())
    print("\n※ 期間2016+・本決算イベント・市場調整CAR・会社予想ベースSUE。論文の2002-2020前半は再現不可。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
