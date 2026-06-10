"""仮説検証：日経225 入替イベントの強制フロー（ドリフト／実施後リバーサル）。

経済的根拠（Chan の問い）：日経225連動ファンド（ETF・インデックス投信・先物裁定）は
**実施日の前営業日の引け**で機械的に採用銘柄を買い・除外銘柄を売る義務を負う。
このフローは価格に依存しない（＝情報を持たない）ため、
  ① 発表→実施：裁定勢の先回りで採用銘柄が上昇・除外銘柄が下落（ドリフト）
  ② 実施後：需給の一巡で行き過ぎが戻る（リバーサル＝採用売り・除外買い）
という2つの歪みが残るはず。誰が反対側か＝「期日に必ず執行しなければならない
パッシブ」。なぜ裁定され尽くさないか＝イベントが年1〜2回と少なく、専業の資本が
張り付きにくい（低容量・個人向きの隙間）という仮説。

データ：`equities/index_events.py`（公式変更履歴から curate・発表日は報道で検証済み・
コミット可能な公開リファレンス）＋ J-Quants 日足ミラー。PIT：発表日アンカーの戦略は
発表日検証済みの定期見直しのみが構造的に対象になる（announce=None は脱落）。
継承上場（しずおかFG/ARCHION等）と TOB 消滅銘柄は取引レグから除外（frictions と同思想）。

規律：4 戦略（ドリフト1＋リバーサル3窓）を1格子として永続レジストリに事前登録・
K 計上（DP7/DP13）。execution_lag=1（同足先読み排除・DP17）・コスト15bps・容量込み。

実行: .venv\\Scripts\\python.exe examples\\research_index_events.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from invest_system.equities.index_events import (  # noqa: E402
    event_holding_windows, tradeable_event_legs, window_weights,
)
from invest_system.equities.panel import load_daily_panel  # noqa: E402
from invest_system.equities.stability import pre_post_sharpe  # noqa: E402
from invest_system.research import (  # noqa: E402
    AsOfView, Strategy, judge_grid, write_html,
)
from invest_system.validation.dsr import sharpe_ratio  # noqa: E402
from invest_system.validation.registry import default_registry  # noqa: E402

OOS = "2024-01"
SCOPE = "index_events_n225"


class _Replay(Strategy):
    """事前計算済み {決定日: ウェイト} を返す（PIT は index_events 側で担保）。"""

    def __init__(self, weights: dict, name: str, params: dict):
        self._w = weights
        self.name = name
        self.params = params

    def target_weights(self, asof):
        return self._w.get(asof.asof, pd.Series(dtype="float64"))


def _isoos(v) -> None:
    for r in v.results:
        s = v.series.get(r.name, pd.Series(dtype="float64")).dropna()
        is_ = s[s.index < pd.Timestamp(OOS)]
        oos = s[s.index >= pd.Timestamp(OOS)]
        ann = np.sqrt(252)
        si = sharpe_ratio(is_) * ann if is_.size >= 8 else np.nan
        so = sharpe_ratio(oos) * ann if oos.size >= 8 else np.nan
        (_, pre), (_, post) = pre_post_sharpe(s, "2020-01-01")
        act = int((s != 0).sum())
        print(f"  {r.name:<24} 全SR={r.sr_ann:+.2f} DSR={r.dsr:.2f} | "
              f"IS={si:+.2f} OOS={so:+.2f} | 前/後2020={pre:+.2f}/{post:+.2f} | "
              f"建玉日={act}")


def main() -> int:
    adds, dels = tradeable_event_legs()
    codes = sorted(set(adds["code"]) | set(dels["code"]))
    daily = load_daily_panel(field="AdjC", codes=codes)
    turn = load_daily_panel(field="Va", codes=codes)
    if daily.empty:
        print("ERROR: 日足ミラー（Silver/Raw）が見つかりません。")
        return 1
    dates = daily.index
    view = AsOfView({"close": daily})
    advp = turn.rolling(20, min_periods=5).mean()
    in_range = adds["effective"].between(dates[0], dates[-1])
    print(f"=== 日経225 入替イベント（{dates[0]:%Y-%m}〜{dates[-1]:%Y-%m}・"
          f"日次{len(dates)}本・scope={SCOPE}）===")
    print(f"取引可能レグ: 採用 {len(adds)}（うち期間内 {int(in_range.sum())}）"
          f" / 除外 {len(dels)}。発表日検証済み（定期）: "
          f"{int(adds['announce'].notna().sum())} 採用レグ")

    # --- ① ドリフト：発表翌営業日に建て、実施前営業日の引けまで保有 ---
    # 決定日 [announce, effective-3]・lag=1 ⇒ 実現 [announce+1, effective-1] の終値リターン
    # ＝発表当日の引け後発表を翌日終値で建てる保守側。リバランス引け（effective-1）を含む。
    dl = event_holding_windows(adds, dates, start_anchor="announce", start_offset=0,
                               end_anchor="effective", end_offset=-3)
    ds = event_holding_windows(dels, dates, start_anchor="announce", start_offset=0,
                               end_anchor="effective", end_offset=-3)
    drift = _Replay(window_weights(dl, ds, dates), "drift_ls(ann->eff-1)",
                    {"start": "announce+0", "end": "effective-3", "legs": "periodic"})

    # --- ② リバーサル：実施日の引けで逆向きに建て、N営業日保有 ---
    # 決定日 [effective-1, effective+N-2]・lag=1 ⇒ 実現 [effective, effective+N]
    strategies = [drift]
    for n in (5, 10, 20):
        rl = event_holding_windows(dels, dates, start_anchor="effective",
                                   start_offset=-1, end_anchor="effective",
                                   end_offset=n - 2)
        rs = event_holding_windows(adds, dates, start_anchor="effective",
                                   start_offset=-1, end_anchor="effective",
                                   end_offset=n - 2)
        strategies.append(_Replay(window_weights(rl, rs, dates),
                                  f"reversal_ls(eff,+{n}d)",
                                  {"start": "effective-1", "hold_days": n,
                                   "legs": "all_tradeable"}))

    with default_registry() as reg:
        v = judge_grid(
            strategies, view, scope=SCOPE,
            hypothesis=("日経225入替のパッシブ強制フローは、発表→実施のドリフトと"
                        "実施後のリバーサルという取引可能な歪みを残すか"),
            economic_rationale=("指数連動ファンドは実施前営業日の引けで価格非感応の売買を"
                                "強制される。年1-2回・少銘柄の低容量イベントのため専業資本が"
                                "張り付きにくく、個人規模に歪みが残るという仮説。発表日は"
                                "報道で個別検証済み・継承/TOB消滅レグは除外（PIT）。"),
            registry=reg, costs_bps=15.0, execution_lag=1,
            adv=advp, participation=0.1)
    print("\n" + v.report_md)
    print("HTML:", write_html(v, f"data/reports/{v.scope}.html"))

    print(f"\n--- IS/OOS（保留 {OOS}〜・年率Sharpe・建玉日数）---")
    _isoos(v)

    # --- 診断：定期見直しごとのドリフト窓 LS 累積リターン（どの回が効いたか）---
    print("\n--- 定期見直し別のドリフト窓（採用-除外 LS・累積リターン）---")
    per = adds.dropna(subset=["announce"]).groupby("effective")
    for eff, grp in per:
        d_grp = dels[dels["effective"] == eff]
        wl = event_holding_windows(grp, dates, start_anchor="announce",
                                   start_offset=0, end_anchor="effective",
                                   end_offset=-3)
        ws = event_holding_windows(d_grp, dates, start_anchor="announce",
                                   start_offset=0, end_anchor="effective",
                                   end_offset=-3)
        if wl.empty:
            continue
        s0, s1 = wl["start"].min(), wl["end"].max()
        i0, i1 = dates.get_loc(s0), dates.get_loc(s1)
        if i1 + 2 >= len(dates):
            i1 = len(dates) - 3
        seg = daily.iloc[i0:i1 + 3]
        ret = seg.pct_change()
        cum_l = float(np.nanmean((1 + ret[wl["code"]].iloc[1:]).prod() - 1))
        cum_s = (float(np.nanmean((1 + ret[ws["code"]].iloc[1:]).prod() - 1))
                 if len(ws) else np.nan)
        print(f"  {eff:%Y-%m-%d}  採用{len(wl)}銘柄 {cum_l:+7.2%} / "
              f"除外{len(ws)}銘柄 {cum_s:+7.2%} / LS {cum_l - cum_s:+7.2%}")

    print("\n※ ドリフトは定期見直し（発表日検証済み14回）のみ・リバーサルは臨時補充も含む"
          "全取引可能レグ。判定は scope 累計 K のデフレートDSR（多重検定補正）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
