"""仮説検証：N225 オプションの体系的ボラ売り＝分散リスクプレミアム(VRP)。

代替戦略ドキュメント 戦略5（Systematic Volatility Premium）を、J-Quants N225 オプション
（data/jquants/options_225・2016-06〜2026-06・2607日・IV/Settle/Strike/SQ日/PC区分/原資産）で
裁く。ファクトリ唯一の「データはあるが未検証」の領域。

設計（事前登録・PIT・満期SQ整合）：
- ロール＝月次SQ（毎月第2金曜）。各サイクルで前限月SQ翌営業日に翌限月(≈1ヶ月)を売り、次SQまで保有。
- 変種（各点が scope vol_premium_n225 の独立試行）：
  ① short_straddle_atm（ATM コール+プット売り）
  ② short_strangle_5pct（±5% OTM ストラングル売り）
  ③ short_strangle_10pct（±10% OTM）
  ④ put_write_atm（ATM プットのみ売り＝PutWrite 型）
- 損益（指数ポイント）＝受取プレミアム(Settle) − 満期ペイオフ(max(SQ−Kc,0)+max(Kp−SQ,0))。
  リターン＝損益 / 原資産（フルキャッシュ担保・無レバの保守基準）。
- 現実性＝オプション往復スリッページをプレミアムの一定割合で控除（既定5%・別途感応）。
  テールリスク（負の歪度・2020-03/2024-08）は judge の skew/kurt/maxDD/PSR が補正。

実行: $env:PYTHONUTF8="1"; .venv\\Scripts\\python.exe examples\\research_vol_premium_n225.py
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from invest_system.config import get_env  # noqa: E402
from invest_system.research import AsOfView, Strategy, judge_grid, write_html  # noqa: E402
from invest_system.validation.dsr import sharpe_ratio  # noqa: E402
from invest_system.validation.registry import default_registry  # noqa: E402

SCOPE = "vol_premium_n225"
OOS = get_env("J_VRP_OOS", "2024-01") or "2024-01"
SLIP = float(get_env("J_VRP_SLIP", "0.05") or "0.05")    # プレミアムに対する往復スリッページ率
OPT_DIR = "data/jquants/options_225"


class _Replay(Strategy):
    def __init__(self, weights: dict, name: str, params: dict):
        self._w, self.name, self.params = weights, name, params

    def target_weights(self, asof):
        return self._w.get(asof.asof, pd.Series(dtype="float64"))


def _all_dates() -> list[tuple[pd.Timestamp, str]]:
    out = []
    for f in sorted(glob.glob(f"{OPT_DIR}/*.parquet")):
        d = pd.to_datetime(Path(f).stem, format="%Y%m%d")
        out.append((d, f))
    return out


def _load_after(files, date, max_ahead: int = 6):
    """date 以降で最初の非空チェーン（祝日スキップ）。"""
    for d, f in files:
        if d < date:
            continue
        if (d - date).days > max_ahead and date != d:
            pass
        df = pd.read_parquet(f)
        if len(df):
            return d, df
        if (d - date).days > 15:
            break
    return None, None


def _load_on_or_before(files, date, max_back: int = 6):
    """date 以前で最も近い非空チェーン（SQ 当日が休場のときの保険）。"""
    prev = None
    for d, f in files:
        if d > date:
            break
        prev = (d, f)
    if prev is None:
        return None, None
    # walk back until non-empty
    idx = [i for i, (d, _) in enumerate(files) if d == prev[0]][0]
    for j in range(idx, max(-1, idx - max_back - 1), -1):
        df = pd.read_parquet(files[j][1])
        if len(df):
            return files[j][0], df
    return None, None


def _chain_under(df: pd.DataFrame) -> float:
    up = df["UnderPx"].dropna()
    return float(up.median()) if len(up) else float("nan")


def _leg(sub: pd.DataFrame, pc: str, target_k: float, side: str) -> tuple[float, float]:
    """(strike, premium=Settle) を返す。side='atm'|'otm_call'|'otm_put' で選択。"""
    legs = sub[sub["PCDiv"].astype(str) == pc].dropna(subset=["Strike", "Settle"])
    legs = legs[legs["Settle"] > 0]
    if legs.empty:
        return float("nan"), float("nan")
    if side == "atm":
        row = legs.iloc[(legs["Strike"] - target_k).abs().argsort().iloc[0]]
    elif side == "otm_call":
        cand = legs[legs["Strike"] >= target_k]
        row = (cand.sort_values("Strike").iloc[0] if len(cand)
               else legs.sort_values("Strike").iloc[-1])
    else:  # otm_put
        cand = legs[legs["Strike"] <= target_k]
        row = (cand.sort_values("Strike").iloc[-1] if len(cand)
               else legs.sort_values("Strike").iloc[0])
    return float(row["Strike"]), float(row["Settle"])


def build_cycles() -> pd.DataFrame:
    files = _all_dates()
    # CM -> SQD マップ（密スキャンで全限月を捕捉）
    cmsqd: dict[str, pd.Timestamp] = {}
    for d, f in files[::5]:
        try:
            df = pd.read_parquet(f, columns=["CM", "SQD"])
        except Exception:           # 空マーカー(_empty スキーマ)はスキップ
            continue
        if not len(df):
            continue
        g = df.dropna(subset=["CM", "SQD"]).groupby(df["CM"].astype(str))["SQD"].first()
        for cm, sqd in g.items():
            cmsqd.setdefault(cm, pd.to_datetime(sqd))
    sq = pd.Series(cmsqd).sort_values()
    sq = sq[(sq >= "2016-07-01") & (sq <= "2026-06-30")]
    cms = list(sq.index)
    rows = []
    for i in range(len(cms) - 1):
        cm_next = cms[i + 1]                 # 売る限月（次の月次SQで満期）
        sq_prev, sq_next = sq.iloc[i], sq.iloc[i + 1]
        ed, chain = _load_after(files, sq_prev + pd.Timedelta(days=1))
        if chain is None:
            continue
        sub = chain[chain["CM"].astype(str) == cm_next]
        if not len(sub):
            continue
        und = _chain_under(chain)
        if not np.isfinite(und) or und <= 0:
            continue
        # 満期 SQ 値（原資産 at SQD）
        sd, sqchain = _load_on_or_before(files, sq_next)
        sqv = _chain_under(sqchain) if sqchain is not None else float("nan")
        if not np.isfinite(sqv):
            continue
        # 各レッグ（PCDiv: 1=プット, 2=コール ＝ストライク×価格の単調性で確定）
        kc_atm, pc_atm = _leg(sub, "2", und, "atm")
        kp_atm, pp_atm = _leg(sub, "1", und, "atm")
        kc5, pc5 = _leg(sub, "2", und * 1.05, "otm_call")
        kp5, pp5 = _leg(sub, "1", und * 0.95, "otm_put")
        kc10, pc10 = _leg(sub, "2", und * 1.10, "otm_call")
        kp10, pp10 = _leg(sub, "1", und * 0.90, "otm_put")
        rows.append(dict(entry=ed, sq=sq_next, und=und, sqv=sqv,
                         kc_atm=kc_atm, pc_atm=pc_atm, kp_atm=kp_atm, pp_atm=pp_atm,
                         kc5=kc5, pc5=pc5, kp5=kp5, pp5=pp5,
                         kc10=kc10, pc10=pc10, kp10=kp10, pp10=pp10))
    return pd.DataFrame(rows).set_index("entry").sort_index()


def _call_payoff(sqv, k):
    return max(float(sqv) - float(k), 0.0) if np.isfinite(k) else 0.0


def _put_payoff(sqv, k):
    return max(float(k) - float(sqv), 0.0) if np.isfinite(k) else 0.0


def variant_returns(cyc: pd.DataFrame, slip: float) -> dict[str, pd.Series]:
    """各変種のネット周期リターン（受取プレミアム−ペイオフ−スリッページ）/原資産。"""
    out: dict[str, list] = {k: [] for k in
                            ["short_straddle_atm", "short_strangle_5pct",
                             "short_strangle_10pct", "put_write_atm"]}
    idx = []
    for t, r in cyc.iterrows():
        idx.append(t)
        und, sqv = r["und"], r["sqv"]

        def net(prem, payoff):
            return (prem - payoff - slip * prem) / und

        # ① ATM ストラドル
        prem = r["pc_atm"] + r["pp_atm"]
        pay = _call_payoff(sqv, r["kc_atm"]) + _put_payoff(sqv, r["kp_atm"])
        out["short_straddle_atm"].append(net(prem, pay))
        # ② ±5% ストラングル
        prem = r["pc5"] + r["pp5"]
        pay = _call_payoff(sqv, r["kc5"]) + _put_payoff(sqv, r["kp5"])
        out["short_strangle_5pct"].append(net(prem, pay))
        # ③ ±10% ストラングル
        prem = r["pc10"] + r["pp10"]
        pay = _call_payoff(sqv, r["kc10"]) + _put_payoff(sqv, r["kp10"])
        out["short_strangle_10pct"].append(net(prem, pay))
        # ④ ATM プットライト
        prem = r["pp_atm"]
        pay = _put_payoff(sqv, r["kp_atm"])
        out["put_write_atm"].append(net(prem, pay))
    return {k: pd.Series(v, index=idx).dropna() for k, v in out.items()}


def _price_panel(rets: dict[str, pd.Series]) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """各変種ネット系列 → 合成 NAV 価格パネル（fwd pct_change が r を再現）。

    決定日 t_i で weight=1 を保持すると engine の fwd[t_i]=nav[t_{i+1}]/nav[t_i]−1=r_i。
    末尾に終端点を足し、engine の末尾 drop で全 r_i が判定対象になる。
    """
    idx = pd.DatetimeIndex(sorted(set().union(*[set(s.index) for s in rets.values()])))
    pidx = idx.append(pd.DatetimeIndex([idx[-1] + pd.Timedelta(days=28)]))
    cols = {}
    for name, s in rets.items():
        s = s.reindex(idx).fillna(0.0)
        nav = pd.Series(1.0, index=pidx)
        nav.iloc[1:] = (1.0 + s.values).cumprod()
        cols[name] = nav
    return pd.DataFrame(cols), idx


def main() -> int:
    print(f"=== N225 ボラ売り(VRP)検証  scope={SCOPE}  slip={SLIP:.0%} ===")
    cyc = build_cycles()
    print(f"サイクル数（月次SQ）= {len(cyc)}  期間 {cyc.index.min():%Y-%m}〜{cyc.index.max():%Y-%m}")
    # 診断：プレミアム水準・ブリーチ
    avgprem = ((cyc["pc_atm"] + cyc["pp_atm"]) / cyc["und"]).mean()
    print(f"ATMストラドル平均プレミアム ≈ {avgprem:.2%}（/原資産・月次）")

    rets = variant_returns(cyc, SLIP)
    panel, idx = _price_panel(rets)
    view = AsOfView({"close": panel})                   # close=NAV

    strategies = []
    for name in rets:
        w = {t: pd.Series({name: 1.0}) for t in idx}
        strategies.append(_Replay(w, name, {"slip": SLIP, "roll": "monthly_SQ"}))

    with default_registry() as reg:
        v = judge_grid(
            strategies, view, scope=SCOPE,
            hypothesis=("N225 オプションの体系的ボラ売り（ATM ストラドル/OTM ストラングル/"
                        "プットライト・月次SQロール）は分散リスクプレミアム(IV>実現ボラ)を"
                        "収穫し正の期待値を持つか。テールリスク控除後も DSR で生存するか"),
            economic_rationale=("インプライド・ボラはリスク回避により実現ボラを恒常的に上回る"
                                "（variance risk premium・文献頑健）。売り手は保険料を得るが、"
                                "急落時に非線形の大損（負の歪度）を負う。VRP の真価はテール込みで"
                                "DSR/PSR が正に残るか＝保険料がテール損を上回るか"),
            registry=reg, costs_bps=0.0)              # コストは系列内で控除済み
    print("\n" + v.report_md)
    write_html(v, f"data/reports/{SCOPE}.html")

    print(f"\n--- IS/OOS（保留 {OOS}〜・年率Sharpe）・歪度・テール ---")
    for r in v.results:
        s = rets[r.name].dropna()
        is_ = s[s.index < pd.Timestamp(OOS)]
        oos = s[s.index >= pd.Timestamp(OOS)]
        si = sharpe_ratio(is_) * np.sqrt(12) if is_.size >= 8 else np.nan
        so = sharpe_ratio(oos) * np.sqrt(12) if oos.size >= 8 else np.nan
        worst = s.nsmallest(3)
        worst_s = " ".join(f"{d:%Y-%m}:{x:+.1%}" for d, x in worst.items())
        print(f"  {r.name:<20} 全SR={r.sr_ann:+.2f} DSR={r.dsr:.2f} skew={s.skew():+.2f} "
              f"kurt={s.kurtosis():+.1f} | IS={si:+.2f} OOS={so:+.2f} | "
              f"年率平均={s.mean()*12:+.1%} 最悪3={worst_s}")

    print(f"\n--- スリッページ感応（short_strangle_5pct・往復%→net年率SR/平均）---")
    for sl in [0.0, 0.05, 0.10, 0.20]:
        rr = variant_returns(cyc, sl)["short_strangle_5pct"].dropna()
        print(f"  slip={sl:.0%}  net年率SR={sharpe_ratio(rr)*np.sqrt(12):+.2f}  "
              f"年率平均={rr.mean()*12:+.1%}  最悪={rr.min():+.1%}")
    print("\n  ※ VRP が文献通りなら net がテール込みでも正の DSR に残るか。"
          "残らねば『個人のフルキャッシュ担保ボラ売りは保険料<テール損』を実証。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
