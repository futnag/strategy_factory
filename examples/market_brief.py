"""市場定点観測ブリーフ：ローカルミラーだけで市場状態の数値を1画面に集約する。

/market-brief スキルの決定論的コア。**数値はここで計算し、解釈は LLM/人間が行う**
（strategy_monitor と同じ分業）。ネットワーク不要・読み取り専用・常に exit 0
（データセット欠損は WARNINGS に列挙して劣化継続＝flag, don't fail）。

集約する内容:
  1) 指数: TOPIX ヘッドライン（リターン・実現ボラ・52週高値からの距離）＋サイズ指数
  2) セクター: 33業種指数の対TOPIX相対強度（1M/3M）上位・下位、60日線上の業種比率
  3) 需給: 投資部門別 週次ネットフロー（海外/個人/信託銀/投信）の直近値と52週z
  4) 信用・空売り: 週次信用残の売買比率と歴史分位、業種別空売り比率、空売り残高報告
  5) ボラ: N225 IV（ATM/スキュー/ターム）と歴史分位、TOPIX実現ボラ、IV−RV
  6) マクロ: USDJPY・日10年金利の水準と1ヶ月変化

出力:
  - stdout: テキストブリーフ
  - output/market_brief/brief_{YYYYMMDD}.json（機械可読。過去分は残る→前回比較に使う）

実行:
  $env:PYTHONUTF8="1"; .\\.venv\\Scripts\\python.exe examples\\market_brief.py
  オプション: --asof YYYY-MM-DD（その日時点に切り詰めて再現）
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from invest_system.data.external import load_external_prices, load_macro  # noqa: E402
from invest_system.equities import flows, margin  # noqa: E402

DATA = ROOT / "data"
OUT_DIR = ROOT / "output" / "market_brief"

# J-Quants 指数コード（既知分のみ命名。未知コードは生コード表示）
SIZE_INDEX_NAMES = {
    "0000": "TOPIX", "0028": "TOPIX Core30", "0029": "TOPIX Large70",
    "002A": "TOPIX 100", "002B": "TOPIX Mid400", "002C": "TOPIX 500",
    "002D": "TOPIX Small", "002E": "TOPIX 1000",
}
# 業種別指数 0040..0060（hex連番33本）は、上場マスタの S33 コード昇順（=JPX 33業種の
# 正準順）と同順で並ぶため、名称はマスタから導出する（ハードコードしない）。
SECTOR_INDEX_CODES = [f"{c:04X}" for c in range(0x40, 0x61)]

WARNINGS: list[str] = []


def _warn(msg: str) -> None:
    WARNINGS.append(msg)
    print(f"  [WARN] {msg}")


def _pct_rank(series: pd.Series, value: float) -> float | None:
    """value が series 内で下から何%か（0-100）。"""
    s = pd.Series(series).dropna()
    if len(s) < 20 or value is None or not np.isfinite(value):
        return None
    return round(float((s <= value).mean() * 100), 1)


def _ret(c: pd.Series, days: int) -> float | None:
    if len(c) <= days:
        return None
    return round(float(c.iloc[-1] / c.iloc[-1 - days] - 1) * 100, 2)


def _sector_names() -> dict[str, str]:
    """業種別指数コード → 業種名。上場マスタ S33/S33Nm の昇順から導出。"""
    fp = DATA / "jquants" / "equities_master.parquet"
    if not fp.exists():
        return {}
    m = pd.read_parquet(fp)
    if "S33" not in m.columns or "S33Nm" not in m.columns:
        return {}
    pairs = (m[["S33", "S33Nm"]].dropna().drop_duplicates()
             .query("S33 != '9999'").sort_values("S33"))
    names = list(pairs["S33Nm"])
    if len(names) != len(SECTOR_INDEX_CODES):
        _warn(f"S33 業種数 {len(names)} ≠ 33 — 業種指数名は生コード表示に切替")
        return {}
    return dict(zip(SECTOR_INDEX_CODES, names))


def _load_index(code: str, asof: pd.Timestamp) -> pd.Series:
    fp = DATA / "jquants" / "indices" / f"code_{code}.parquet"
    if not fp.exists():
        return pd.Series(dtype="float64")
    df = pd.read_parquet(fp)
    if df.empty or "C" not in df.columns:
        return pd.Series(dtype="float64")
    s = (df.assign(Date=pd.to_datetime(df["Date"])).set_index("Date")["C"]
         .astype(float).sort_index())
    return s.loc[:asof]


def sec_indices(asof: pd.Timestamp) -> dict:
    out: dict = {}
    topix = _load_index("0000", asof)
    if topix.empty:
        _warn("指数 0000（TOPIX）が読めない — 指数セクションをスキップ")
        return out
    ytd_start = topix.loc[:pd.Timestamp(asof.year, 1, 1)]
    rv20 = None
    if len(topix) > 21:
        rv20 = round(float(topix.pct_change().iloc[-20:].std() * np.sqrt(252) * 100), 2)
    hi52 = float(topix.iloc[-252:].max()) if len(topix) >= 60 else None
    out["topix"] = {
        "date": str(topix.index[-1].date()), "close": round(float(topix.iloc[-1]), 2),
        "ret_1w_pct": _ret(topix, 5), "ret_1m_pct": _ret(topix, 21),
        "ret_3m_pct": _ret(topix, 63),
        "ret_ytd_pct": (round(float(topix.iloc[-1] / ytd_start.iloc[-1] - 1) * 100, 2)
                        if len(ytd_start) else None),
        "realized_vol20_ann_pct": rv20,
        "dist_52w_high_pct": (round(float(topix.iloc[-1] / hi52 - 1) * 100, 2)
                              if hi52 else None),
    }
    out["size"] = {}
    for code, name in SIZE_INDEX_NAMES.items():
        if code == "0000":
            continue
        s = _load_index(code, asof)
        if len(s) > 63:
            out["size"][name] = {"ret_1m_pct": _ret(s, 21), "ret_3m_pct": _ret(s, 63)}

    names = _sector_names()
    rows, above_ma = [], []
    t1m, t3m = out["topix"]["ret_1m_pct"], out["topix"]["ret_3m_pct"]
    for code in SECTOR_INDEX_CODES:
        s = _load_index(code, asof)
        if len(s) < 70:
            continue
        r1, r3 = _ret(s, 21), _ret(s, 63)
        above_ma.append(float(s.iloc[-1] > s.iloc[-60:].mean()))
        rows.append({
            "code": code, "name": names.get(code, code),
            "ret_1m_pct": r1, "ret_3m_pct": r3,
            "rs_1m_pct": round(r1 - t1m, 2) if (r1 is not None and t1m is not None) else None,
            "rs_3m_pct": round(r3 - t3m, 2) if (r3 is not None and t3m is not None) else None,
        })
    rows.sort(key=lambda r: (r["rs_1m_pct"] is None, r["rs_1m_pct"]), reverse=True)
    out["sectors"] = {
        "n": len(rows),
        "pct_above_60d_ma": round(100 * np.mean(above_ma), 1) if above_ma else None,
        "top5_rs_1m": rows[:5], "bottom5_rs_1m": rows[-5:][::-1],
    }
    return out


def sec_flows(asof: pd.Timestamp) -> dict:
    df = flows.load_investor_types()
    if df.empty:
        _warn("investor_types が空 — 需給セクションをスキップ")
        return {}
    df = df[pd.to_datetime(df["EnDate"]) <= asof]
    out: dict = {}
    for inv in ("foreign", "individual", "trust_bank", "inv_trust"):
        try:
            bal = flows.section_net_flow(df, investor=inv)          # 単位=千円（J-Quants）
            inten = flows.net_flow_intensity(df, investor=inv)      # 無次元 −1..1
            if bal.empty:
                continue
            tail = bal.iloc[-52:]
            z = None
            if len(tail) >= 20 and float(tail.std()) > 0:
                z = round(float((bal.iloc[-1] - tail.mean()) / tail.std()), 2)
            out[inv] = {
                "week_end": str(pd.to_datetime(bal.index[-1]).date()),
                "net_oku_yen": round(float(bal.iloc[-1]) / 1e5, 0),   # 千円→億円
                "net_4w_oku_yen": round(float(bal.iloc[-4:].sum()) / 1e5, 0),
                "z_52w": z,
                "intensity": (round(float(inten.iloc[-1]), 3) if len(inten) else None),
            }
        except Exception as e:  # noqa: BLE001
            _warn(f"flows[{inv}]: {e}")
    return out


def sec_margin(asof: pd.Timestamp) -> dict:
    out: dict = {}
    try:
        wk = margin.load_weekly_margin()
        if not wk.empty and {"Date", "LongVol", "ShrtVol"} <= set(wk.columns):
            wk = wk[pd.to_datetime(wk["Date"]) <= asof]
            agg = (wk.groupby("Date")[["LongVol", "ShrtVol"]].sum().sort_index())
            agg = agg[(agg["LongVol"] > 0)]
            ratio = agg["ShrtVol"] / agg["LongVol"]
            last = agg.iloc[-1]
            out["weekly"] = {
                "date": str(pd.to_datetime(agg.index[-1]).date()),
                "long_vol": float(last["LongVol"]), "short_vol": float(last["ShrtVol"]),
                "short_to_long": round(float(ratio.iloc[-1]), 4),
                "short_to_long_pctile_hist": _pct_rank(ratio, float(ratio.iloc[-1])),
                "long_wow_pct": (round(float(agg["LongVol"].iloc[-1] /
                                             agg["LongVol"].iloc[-2] - 1) * 100, 2)
                                 if len(agg) > 1 else None),
            }
        else:
            _warn("margin_weekly が空/列不足")
    except Exception as e:  # noqa: BLE001
        _warn(f"margin_weekly: {e}")

    try:
        sr = margin.sector_short_ratio(margin.load_short_ratio())
        if not sr.empty:
            sr = sr[pd.to_datetime(sr["Date"]) <= asof]
            latest_date = sr["Date"].max()
            snap = sr[sr["Date"] == latest_date].copy()
            hist_1y = sr[pd.to_datetime(sr["Date"]) >= asof - pd.Timedelta(days=365)]
            mkt = sr.groupby("Date")["sector_short_ratio"].mean()
            snap["pctile_1y"] = snap.apply(
                lambda r: _pct_rank(
                    hist_1y[hist_1y["S33"] == r["S33"]]["sector_short_ratio"],
                    r["sector_short_ratio"]), axis=1)
            snap = snap.sort_values("sector_short_ratio", ascending=False)
            cols = ["S33", "sector_short_ratio", "pctile_1y"]
            out["sector_short"] = {
                "date": str(pd.to_datetime(latest_date).date()),
                "market_mean": round(float(mkt.iloc[-1]), 4),
                "market_mean_pctile_1y": _pct_rank(mkt.iloc[-252:], float(mkt.iloc[-1])),
                "top3": snap[cols].head(3).to_dict("records"),
                "bottom3": snap[cols].tail(3).to_dict("records"),
            }
    except Exception as e:  # noqa: BLE001
        _warn(f"short_ratio: {e}")

    try:  # 空売り残高報告（直近45日窓のスナップショット。単位はソース準拠=対発行済比）
        files = sorted((DATA / "jquants" / "short_positions").glob("calc_*.parquet"))
        cutoff = int((asof - pd.Timedelta(days=45)).strftime("%Y%m%d"))
        frames = [pd.read_parquet(p) for p in files
                  if cutoff <= int(p.stem.split("_")[1]) <= int(asof.strftime("%Y%m%d"))]
        frames = [f for f in frames if not f.empty and "_empty" not in f.columns]
        if frames:
            sp = pd.concat(frames, ignore_index=True)
            si = margin.short_interest(sp)
            si = si.sort_values("Date").groupby("Code").tail(1)
            top = si.sort_values("short_interest", ascending=False).head(5)
            out["short_positions_45d"] = {
                "n_reports": int(len(sp)), "n_names": int(si["Code"].nunique()),
                "median_si": round(float(si["short_interest"].median()), 4),
                "top5": [{"code": str(r.Code), "si": round(float(r.short_interest), 4)}
                         for r in top.itertuples()],
            }
    except Exception as e:  # noqa: BLE001
        _warn(f"short_positions: {e}")
    return out


def sec_vol(asof: pd.Timestamp, topix_rv20: float | None) -> dict:
    out: dict = {}
    try:
        iv = pd.read_parquet(DATA / "supplemental" / "n225_iv.parquet").sort_index()
        iv.index = pd.to_datetime(iv.index)
        iv = iv.loc[:asof]
        col = "n225_iv_atm" if "n225_iv_atm" in iv.columns else iv.columns[0]
        s = iv[col].dropna()
        if len(s):
            out["n225_iv_atm"] = round(float(s.iloc[-1]), 2)
            out["n225_iv_pctile_1y"] = _pct_rank(s.iloc[-252:], float(s.iloc[-1]))
            if topix_rv20 is not None:
                out["iv_minus_rv_pct"] = round(float(s.iloc[-1]) - topix_rv20, 2)
    except Exception as e:  # noqa: BLE001
        _warn(f"n225_iv: {e}")
    try:
        sig = pd.read_parquet(DATA / "processed" / "option_n225_signals.parquet").sort_index()
        sig.index = pd.to_datetime(sig.index)
        sig = sig.loc[:asof]
        for c in ("skew", "term"):
            if c in sig.columns and len(sig[c].dropna()):
                v = float(sig[c].dropna().iloc[-1])
                out[c] = round(v, 4)
                out[f"{c}_pctile_1y"] = _pct_rank(sig[c].dropna().iloc[-252:], v)
    except Exception as e:  # noqa: BLE001
        _warn(f"option_n225_signals: {e}")
    try:
        mac = load_macro(["vix"], base=str(DATA))
        if not mac.empty and "vix" in mac.columns:
            v = mac["vix"].dropna()
            v.index = pd.to_datetime(v.index)
            v = v.loc[:asof]
            if len(v):
                out["vix"] = round(float(v.iloc[-1]), 2)
                out["vix_pctile_1y"] = _pct_rank(v.iloc[-252:], float(v.iloc[-1]))
    except Exception as e:  # noqa: BLE001
        _warn(f"vix: {e}")
    return out


def sec_macro(asof: pd.Timestamp) -> dict:
    out: dict = {}
    try:
        px = load_external_prices(["usdjpy", "nk225"], base=str(DATA))
        px.index = pd.to_datetime(px.index)
        px = px.loc[:asof]
        if "usdjpy" in px.columns and len(px["usdjpy"].dropna()) > 21:
            s = px["usdjpy"].dropna()
            out["usdjpy"] = round(float(s.iloc[-1]), 2)
            out["usdjpy_chg_1m_pct"] = _ret(s, 21)
        if "nk225" in px.columns and len(px["nk225"].dropna()) > 21:
            s = px["nk225"].dropna()
            out["nk225"] = round(float(s.iloc[-1]), 0)
            out["nk225_ret_1m_pct"] = _ret(s, 21)
    except Exception as e:  # noqa: BLE001
        _warn(f"external_prices: {e}")
    try:
        mac = load_macro(["jp_10y"], base=str(DATA))
        if not mac.empty and "jp_10y" in mac.columns:
            s = mac["jp_10y"].dropna()
            s.index = pd.to_datetime(s.index)
            s = s.loc[:asof]
            if len(s) > 21:
                out["jp_10y_pct"] = round(float(s.iloc[-1]), 3)
                out["jp_10y_chg_1m_bp"] = round(float(s.iloc[-1] - s.iloc[-22]) * 100, 1)
    except Exception as e:  # noqa: BLE001
        _warn(f"macro jp_10y: {e}")
    return out


def _fmt_row(d: dict, keys: list[str]) -> str:
    return "  ".join(f"{k}={d.get(k)}" for k in keys)


def main() -> int:
    ap = argparse.ArgumentParser(description="市場定点観測ブリーフ（数値のみ）")
    ap.add_argument("--asof", default=None, help="YYYY-MM-DD（既定=今日）")
    args = ap.parse_args()
    asof = pd.Timestamp(args.asof) if args.asof else pd.Timestamp(datetime.now().date())

    print(f"=== MARKET BRIEF (asof {asof.date()}) — 数値のみ・解釈はしない ===")

    print("\n[1] 指数・セクター")
    idx = sec_indices(asof)
    if idx.get("topix"):
        print("  TOPIX: " + _fmt_row(idx["topix"], [
            "date", "close", "ret_1w_pct", "ret_1m_pct", "ret_3m_pct", "ret_ytd_pct",
            "realized_vol20_ann_pct", "dist_52w_high_pct"]))
        sec = idx.get("sectors", {})
        print(f"  業種指数 {sec.get('n')}本  60日線上 {sec.get('pct_above_60d_ma')}%")
        for r in sec.get("top5_rs_1m", []):
            print(f"    RS上位: {r['name']}  rs_1m={r['rs_1m_pct']}%  rs_3m={r['rs_3m_pct']}%")
        for r in sec.get("bottom5_rs_1m", []):
            print(f"    RS下位: {r['name']}  rs_1m={r['rs_1m_pct']}%  rs_3m={r['rs_3m_pct']}%")

    print("\n[2] 投資部門別フロー（週次・億円換算/千円単位ソース）")
    fl = sec_flows(asof)
    for inv, d in fl.items():
        print(f"  {inv:<11}: " + _fmt_row(d, ["week_end", "net_oku_yen",
                                              "net_4w_oku_yen", "z_52w", "intensity"]))

    print("\n[3] 信用・空売り")
    mg = sec_margin(asof)
    if "weekly" in mg:
        print("  信用週次: " + _fmt_row(mg["weekly"], [
            "date", "short_to_long", "short_to_long_pctile_hist", "long_wow_pct"]))
    if "sector_short" in mg:
        ss = mg["sector_short"]
        print(f"  業種空売り比率: date={ss['date']} 市場平均={ss['market_mean']}"
              f" (1y分位 {ss['market_mean_pctile_1y']}%)")
    if "short_positions_45d" in mg:
        print("  空売り残高(45d): " + _fmt_row(mg["short_positions_45d"],
                                          ["n_reports", "n_names", "median_si"]))

    print("\n[4] ボラティリティ")
    rv = idx.get("topix", {}).get("realized_vol20_ann_pct")
    vl = sec_vol(asof, rv)
    print("  " + _fmt_row(vl, ["n225_iv_atm", "n225_iv_pctile_1y", "iv_minus_rv_pct",
                               "skew", "skew_pctile_1y", "term", "term_pctile_1y",
                               "vix", "vix_pctile_1y"]))

    print("\n[5] マクロ")
    mc = sec_macro(asof)
    print("  " + _fmt_row(mc, ["usdjpy", "usdjpy_chg_1m_pct", "nk225",
                               "nk225_ret_1m_pct", "jp_10y_pct", "jp_10y_chg_1m_bp"]))

    payload = {
        "asof": str(asof.date()),
        "generated_by": "examples/market_brief.py",
        "warnings": WARNINGS,
        "indices": idx, "flows": fl, "margin_short": mg, "vol": vl, "macro": mc,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fp = OUT_DIR / f"brief_{asof.strftime('%Y%m%d')}.json"
    fp.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str),
                  encoding="utf-8")
    print(f"\nJSON -> {fp.relative_to(ROOT)}")
    if WARNINGS:
        print(f"WARNINGS: {len(WARNINGS)} 件（上記 [WARN]）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
