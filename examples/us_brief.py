"""米国市場ブリーフ：前夜の米国市況＋日本市場への橋渡し数値を1画面に集約する。

/us-brief スキルの決定論的コア。**数値はここで計算し、解釈は LLM/人間が行う**。
読み取り専用・ネットワーク不要（鮮度が必要なら先に examples/update_external.py を実行）。
欠損・停滞系列は WARNINGS に列挙して劣化継続（flag, don't fail・常に exit 0）。

集約する内容:
  1) 米国指数: S&P500 / NASDAQ / ダウ（1日・1週・1ヶ月・YTD・実現ボラ・52週高値距離）
  2) 金利・ボラ: 米10年金利（水準と変化）・FF・VIX（1年分位）
  3) FX・商品: USDJPY・WTI・金・銅（1日・1ヶ月変化）
  4) japan_handoff: NK225先物の直近リターン（夜間込み日足）→ 寄付ギャップの機械的推計、
     USDJPY 1日変化、米指数1日リターン。/jp-open がこの節を入力にする。

出力:
  - stdout: テキストブリーフ
  - output/us_brief/brief_{YYYYMMDD}.json

実行:
  $env:PYTHONUTF8="1"; .\\.venv\\Scripts\\python.exe examples\\us_brief.py
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

DATA = ROOT / "data"
OUT_DIR = ROOT / "output" / "us_brief"

# (キー, 表示名, 代替キー)。主系列が停滞していたら代替（CFD）で補う。
US_INDICES = [("sp500", "S&P500", "us500"),
              ("nasdaq_comp", "NASDAQ", "us_tech100"),
              ("dow", "Dow", "us30")]
COMMODS = [("wti", "WTI"), ("gold", "Gold"), ("copper", "Copper")]

WARNINGS: list[str] = []


def _warn(msg: str) -> None:
    WARNINGS.append(msg)
    print(f"  [WARN] {msg}")


def _pct_rank(series: pd.Series, value: float) -> float | None:
    s = pd.Series(series).dropna()
    if len(s) < 20 or value is None or not np.isfinite(value):
        return None
    return round(float((s <= value).mean() * 100), 1)


def _ret(c: pd.Series, days: int) -> float | None:
    if len(c) <= days:
        return None
    return round(float(c.iloc[-1] / c.iloc[-1 - days] - 1) * 100, 2)


def _series(px: pd.DataFrame, key: str, asof: pd.Timestamp) -> pd.Series:
    if key not in px.columns:
        return pd.Series(dtype="float64")
    s = px[key].dropna()
    s.index = pd.to_datetime(s.index)
    return s.loc[:asof]


def _stale(s: pd.Series, asof: pd.Timestamp, name: str, days: int = 5) -> bool:
    """系列末尾が asof から days 営業日相当より古ければ警告して True。"""
    if s.empty:
        _warn(f"{name}: データ無し")
        return True
    age = (asof - s.index[-1]).days
    if age > days + 2:  # 週末ぶんの猶予
        _warn(f"{name}: 最終 {s.index[-1].date()}（{age}日前）＝停滞。update_external を先に")
        return True
    return False


def _idx_stats(s: pd.Series) -> dict:
    ytd = s.loc[:pd.Timestamp(s.index[-1].year, 1, 1)]
    hi52 = float(s.iloc[-252:].max()) if len(s) >= 60 else None
    rv20 = (round(float(s.pct_change().iloc[-20:].std() * np.sqrt(252) * 100), 2)
            if len(s) > 21 else None)
    return {
        "date": str(s.index[-1].date()), "close": round(float(s.iloc[-1]), 2),
        "ret_1d_pct": _ret(s, 1), "ret_1w_pct": _ret(s, 5), "ret_1m_pct": _ret(s, 21),
        "ret_ytd_pct": (round(float(s.iloc[-1] / ytd.iloc[-1] - 1) * 100, 2)
                        if len(ytd) else None),
        "realized_vol20_ann_pct": rv20,
        "dist_52w_high_pct": (round(float(s.iloc[-1] / hi52 - 1) * 100, 2) if hi52 else None),
    }


def sec_us_indices(px: pd.DataFrame, asof: pd.Timestamp) -> dict:
    out: dict = {}
    for key, name, alt in US_INDICES:
        s = _series(px, key, asof)
        used = key
        if _stale(s, asof, name):
            s2 = _series(px, alt, asof)
            if not s2.empty and (s.empty or s2.index[-1] > s.index[-1]):
                s, used = s2, f"{alt}(代替CFD)"
        if len(s) > 21:
            out[name] = {**_idx_stats(s), "source_key": used}
    return out


def sec_rates_vol(asof: pd.Timestamp) -> dict:
    out: dict = {}
    try:
        mac = load_macro(["us_10y", "us_ff", "vix"], base=str(DATA))
        for k, label in (("us_10y", "us_10y_pct"), ("us_ff", "us_ff_pct")):
            if k in mac.columns:
                s = mac[k].dropna()
                s.index = pd.to_datetime(s.index)
                s = s.loc[:asof]
                if len(s) > 22:
                    out[label] = round(float(s.iloc[-1]), 3)
                    out[label.replace("_pct", "_chg_1w_bp")] = round(
                        float(s.iloc[-1] - s.iloc[-6]) * 100, 1)
                    out[label.replace("_pct", "_chg_1m_bp")] = round(
                        float(s.iloc[-1] - s.iloc[-22]) * 100, 1)
                    if k == "us_10y":
                        out["us_10y_date"] = str(s.index[-1].date())
                        _stale(s, asof, "us_10y(FRED)")
        if "vix" in mac.columns:
            v = mac["vix"].dropna()
            v.index = pd.to_datetime(v.index)
            v = v.loc[:asof]
            if len(v):
                out["vix"] = round(float(v.iloc[-1]), 2)
                out["vix_pctile_1y"] = _pct_rank(v.iloc[-252:], float(v.iloc[-1]))
                out["vix_date"] = str(v.index[-1].date())
    except Exception as e:  # noqa: BLE001
        _warn(f"rates/vix: {e}")
    return out


def sec_fx_commod(px: pd.DataFrame, asof: pd.Timestamp) -> dict:
    out: dict = {}
    s = _series(px, "usdjpy", asof)
    if len(s) > 21 and not _stale(s, asof, "USDJPY"):
        out["usdjpy"] = {"date": str(s.index[-1].date()), "close": round(float(s.iloc[-1]), 2),
                         "chg_1d_pct": _ret(s, 1), "chg_1m_pct": _ret(s, 21)}
    for key, name in COMMODS:
        c = _series(px, key, asof)
        if len(c) > 21:
            _stale(c, asof, name)
            out[key] = {"date": str(c.index[-1].date()), "close": round(float(c.iloc[-1]), 2),
                        "chg_1d_pct": _ret(c, 1), "chg_1m_pct": _ret(c, 21)}
    return out


def sec_japan_handoff(px: pd.DataFrame, asof: pd.Timestamp, us: dict) -> dict:
    """/jp-open への橋渡し。NK225先物（夜間込み日足）→寄付ギャップの機械的推計。"""
    out: dict = {}
    fut = _series(px, "nk225_fut", asof)
    if len(fut) > 1:
        stale = _stale(fut, asof, "NK225先物")
        out["nk225_fut"] = {
            "date": str(fut.index[-1].date()), "close": round(float(fut.iloc[-1]), 0),
            "ret_1d_pct": _ret(fut, 1), "stale": stale,
        }
        # 機械的な寄付ギャップ推計＝先物の直近日次リターン（夜間セッション込みの近似。
        # 日足のため大引け後〜朝の動きと日中の動きを完全には分離できない＝近似と明記）
        out["implied_open_gap_pct"] = _ret(fut, 1)
        out["note"] = ("implied_open_gap_pct は先物日足リターンによる近似。"
                       "寄付直前の板・CME終値の確認は Web で補完すること")
    topix = _series(px, "topix_fut", asof)
    if len(topix) > 1:
        out["topix_fut"] = {"date": str(topix.index[-1].date()),
                            "close": round(float(topix.iloc[-1]), 2),
                            "ret_1d_pct": _ret(topix, 1)}
    usd = _series(px, "usdjpy", asof)
    if len(usd) > 1:
        out["usdjpy_chg_1d_pct"] = _ret(usd, 1)
    out["us_ret_1d_pct"] = {name: d.get("ret_1d_pct") for name, d in us.items()}
    return out


def _fmt_row(d: dict, keys: list[str]) -> str:
    return "  ".join(f"{k}={d.get(k)}" for k in keys)


def main() -> int:
    ap = argparse.ArgumentParser(description="米国市場ブリーフ（数値のみ）")
    ap.add_argument("--asof", default=None, help="YYYY-MM-DD（既定=今日）")
    args = ap.parse_args()
    asof = pd.Timestamp(args.asof) if args.asof else pd.Timestamp(datetime.now().date())

    print(f"=== US BRIEF (asof {asof.date()}) — 数値のみ・解釈はしない ===")
    keys = [k for k, _, _ in US_INDICES] + [alt for _, _, alt in US_INDICES] + \
           [k for k, _ in COMMODS] + ["usdjpy", "nk225_fut", "topix_fut"]
    px = load_external_prices(keys, base=str(DATA))

    print("\n[1] 米国指数")
    us = sec_us_indices(px, asof)
    for name, d in us.items():
        print(f"  {name:<8}: " + _fmt_row(d, ["date", "close", "ret_1d_pct", "ret_1w_pct",
                                              "ret_1m_pct", "ret_ytd_pct",
                                              "realized_vol20_ann_pct", "dist_52w_high_pct"]))

    print("\n[2] 金利・ボラ")
    rv = sec_rates_vol(asof)
    print("  " + _fmt_row(rv, ["us_10y_pct", "us_10y_chg_1w_bp", "us_10y_chg_1m_bp",
                               "us_ff_pct", "vix", "vix_pctile_1y", "vix_date"]))

    print("\n[3] FX・商品")
    fx = sec_fx_commod(px, asof)
    for k, d in fx.items():
        print(f"  {k:<8}: " + _fmt_row(d, ["date", "close", "chg_1d_pct", "chg_1m_pct"]))

    print("\n[4] japan_handoff（/jp-open への入力）")
    jh = sec_japan_handoff(px, asof, us)
    if "nk225_fut" in jh:
        print("  NK225先物: " + _fmt_row(jh["nk225_fut"], ["date", "close", "ret_1d_pct",
                                                            "stale"]))
        print(f"  implied_open_gap_pct={jh.get('implied_open_gap_pct')}"
              f"  usdjpy_chg_1d_pct={jh.get('usdjpy_chg_1d_pct')}")

    payload = {
        "asof": str(asof.date()),
        "generated_by": "examples/us_brief.py",
        "warnings": WARNINGS,
        "us_indices": us, "rates_vol": rv, "fx_commodities": fx, "japan_handoff": jh,
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
