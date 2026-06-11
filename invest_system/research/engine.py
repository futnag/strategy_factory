"""バックテスト・エンジン：戦略の目標ウェイト → コスト込みの損益系列。

各リバランス日 t で戦略に AsOf（t以前）を渡してウェイト w_t を得る。実現損益は
t→t+1 の各銘柄リターンと w_t の内積（＝将来価格で実現＝戦略は未来を見ないが
エンジンは実現値を計算）。取引コストは回転率 sum|Δw| に比例。空シグナル日は現金
（リターン0）。年率係数は日付間隔から自動推定（日次≈252, 月次≈12）。

執行現実性（日本市場固有・equities/frictions.py と対・DP15）：
- no_buy/no_sell: 値幅制限（ストップ高/安の引け張り付き）等で執行不能な銘柄は、
  目標でなく**前回ウェイトを保持**（買えない買い越し/売れない売り越しのキャリー）。
- short_borrow_bps: ショート想定元本に賦課する年率貸株コスト（制度貸株料＋逆日歩
  バッファの保守見積もり）。期間按分でネットから控除。
- costs_bps はスカラに加えて**日付×銘柄の bps パネル**も受ける（ボラ連動スリッページ
  等の状態依存コスト。`equities.frictions.vol_scaled_cost_bps` と対）。

執行タイミング（DP17）：既定 execution_lag=0 は「決定足の終値で執行」＝意思決定に
使った価格で約定する同足執行を含む。日次研究は execution_lag=1 を標準とし、月次研究は
`open_fill_backtest`（決定翌営業日の**始値**で約定・open→open 実現）で再評価する。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data_view import AsOfView
from .strategy import Strategy


@dataclass
class BacktestResult:
    returns: pd.Series          # ネット周期リターン（index=決定日 t）
    gross: pd.Series            # コスト前
    turnover: pd.Series         # sum|Δw|（両側）
    n_positions: pd.Series      # 建玉数
    ann_factor: float           # 年率換算の周期数/年
    name: str
    params: dict = field(default_factory=dict)
    capacity_jpy: float = float("nan")   # 容量(¥): participation%×ADV 制約の AUM 上限
    n_blocked: pd.Series | None = None   # 値幅制限等で執行不能だった注文数（診断）
    short_gross: pd.Series | None = None  # ショート想定元本 sum|w<0|（貸株コストの基礎）


def _ann_factor(idx: pd.DatetimeIndex) -> float:
    if len(idx) < 3:
        return 252.0
    years = (idx[-1] - idx[0]).days / 365.25
    return float(len(idx) / years) if years > 0 else 252.0


def _one_way_cost(delta: pd.Series, when, costs_bps, panel_default: float) -> float:
    """執行した売買 |Δw| への片道コスト（リターン比）を返す。

    costs_bps がスカラなら従来どおり bps×回転率。DataFrame（index=日付, col=銘柄,
    値=bps）なら銘柄別に課金（ボラ連動スリッページ等）。行内の NaN は行中央値、
    行が無い日はパネル全体の中央値（panel_default）で保守的に埋める。
    """
    if not isinstance(costs_bps, pd.DataFrame):
        return float(costs_bps) / 1e4 * float(delta.abs().sum())
    if delta.empty:
        return 0.0
    if when in costs_bps.index:
        row = costs_bps.loc[when].reindex(delta.index)
        row = row.fillna(float(row.median()) if row.notna().any() else panel_default)
    else:
        row = pd.Series(panel_default, index=delta.index)
    return float((delta.abs() * row).sum()) / 1e4


def apply_rebalance_band(weights_by_date: dict, band: float) -> dict:
    """リバランス・デッドバンド（cost-aware 実行フィルター・docs/04 P2-A・F1/D2）。

    {決定日: 目標ウェイト Series} の時系列に対し、**実行後（キャリー）ウェイト**を基準に
    |目標 − 保有| < band の銘柄は据え置く（取引しない）。閾値未満の微調整・ダスト清算に
    コストを払わない＝回転率の抑制。band は執行の現実的運用でありシグナルではない
    （意思決定は不変）。`open_fill_backtest` 等のリプレイ経路にはこの純関数を前段適用し、
    `backtest` には同セマンティクスの `rebalance_band` 引数がある。band=0 は恒等変換。
    """
    if band <= 0:
        return dict(weights_by_date)
    out: dict = {}
    prev = pd.Series(dtype="float64")
    for t in sorted(weights_by_date, key=pd.Timestamp):
        tgt = weights_by_date[t].astype(float)
        names = tgt.index.union(prev.index)
        cur = tgt.reindex(names).fillna(0.0)
        prv = prev.reindex(names).fillna(0.0)
        cur = cur.where((cur - prv).abs() >= band, prv)
        w = cur[cur != 0.0]
        out[t] = w
        prev = w
    return out


def backtest(strategy: Strategy, view: AsOfView, *, price_field: str = "close",
             costs_bps: float = 15.0, rebalance=None, execution_lag: int = 0,
             adv: pd.DataFrame | None = None, participation: float = 0.1,
             no_buy: pd.DataFrame | None = None,
             no_sell: pd.DataFrame | None = None,
             short_borrow_bps: float = 0.0,
             rebalance_band: float = 0.0) -> BacktestResult:
    """戦略を回してネット損益系列を返す。

    execution_lag: 決定から執行までの遅延（バー数）。0=決定足の終値で執行（既定・
      従来）、1=翌足で執行（観測した終値で建てない＝同足の先読みを排除する現実寄り）。
    rebalance: 決定日の列。**パネル index の連続部分列**（ウォームアップの切り落とし等）
      に限る。本エンジンは各決定日に「次の1バー分」しか実現しないため、パネル頻度より
      疎な日付（例：日次パネルに月末だけ）を渡すと間のリターンが静かに脱落する＝検出
      して ValueError。月次研究は月次パネルか `open_fill_backtest` を使うこと。
    adv: 各銘柄の平均売買代金(¥)パネル（index=リバランス日, col=銘柄）。与えると
      容量(capacity_jpy)＝「最も流動性の低い建玉が participation×ADV に達するAUM上限」を
      算出（実運用で約定可能な規模の上限）。
    no_buy / no_sell: 執行不能フラグの bool パネル（index=日付, col=銘柄。
      `equities.frictions.limit_lock_flags` 等）。**執行バー（t+execution_lag）**で
      買い越し/売り越しが不能な銘柄は前回ウェイトを保持（執行できない注文のキャリー）。
      回転率・コストも実際に執行できた分だけ計上する。
    short_borrow_bps: ショート想定元本に賦課する**年率**貸株コスト(bps)。日本の制度
      信用は貸株料約115bps＋逆日歩（不確定）が乗るため、保守見積もりで与える。
      期間按分（/年率係数）でネットから控除（gross には含めない）。
    rebalance_band: リバランス・デッドバンド（DP・docs/04 P2-A）。|目標 − 保有| < band の
      銘柄は据え置く＝閾値未満の微調整に取引コストを払わない。判定の基準は
      **no_buy/no_sell キャリー適用後**のウェイト。0（既定）で従来どおり。
      執行フィルタでありシグナルではない（意思決定は不変）。
    costs_bps: 片道コスト。スカラ bps か、**日付×銘柄の bps パネル**（執行バーの行を
      参照。ボラ連動スリッページ等の状態依存コスト＝`frictions.vol_scaled_cost_bps`）。
    """
    close = view.panels[price_field]
    ret = close.pct_change()
    fwd = ret.shift(-(1 + execution_lag))       # 決定t→(t+lag)建て→翌足で実現
    # 執行不能フラグ・コストパネルも執行バー（t+lag）の値を t に整列させる
    nb = no_buy.shift(-execution_lag) if no_buy is not None else None
    ns = no_sell.shift(-execution_lag) if no_sell is not None else None
    panel_costs = isinstance(costs_bps, pd.DataFrame)
    cp = costs_bps.shift(-execution_lag) if panel_costs else costs_bps
    cp_default = float(np.nanmedian(costs_bps.to_numpy())) if panel_costs else 0.0
    drop = 1 + execution_lag
    dates = pd.DatetimeIndex(rebalance) if rebalance is not None \
        else view.dates[:-drop]                 # 実現できない末尾は除外
    if rebalance is not None and len(dates) > 1:
        pos = view.dates.get_indexer(dates)
        if (pos < 0).any():
            raise ValueError("rebalance にパネル index に無い日付が含まれています。")
        if (np.diff(pos) != 1).any():
            raise ValueError(
                "rebalance がパネル頻度より疎です：本エンジンは各決定日に「次の1バー分」"
                "しか実現しないため、間のリターンが静かに脱落します。rebalance はパネル "
                "index の連続部分列に限り、月次研究は月次パネルか open_fill_backtest を"
                "使ってください。")
    borrow_per_period = short_borrow_bps / 1e4 / _ann_factor(dates)
    prev_w: pd.Series | None = None
    rows = []
    capacity = float("inf")
    for t in dates:
        w_tgt = strategy.target_weights(view.asof(t))
        pre0 = prev_w if prev_w is not None else pd.Series(dtype="float64")
        names = w_tgt.index.union(pre0.index)
        cur = w_tgt.reindex(names).fillna(0.0)
        prv = pre0.reindex(names).fillna(0.0)
        blocked_n = 0
        if (nb is not None or ns is not None) and len(names):
            delta0 = cur - prv                  # 執行不能な注文は前回ウェイトを保持
            block = pd.Series(False, index=names)
            if nb is not None and t in nb.index:
                f = nb.loc[t].reindex(names).fillna(False).astype(bool)
                block |= (delta0 > 0) & f
            if ns is not None and t in ns.index:
                f = ns.loc[t].reindex(names).fillna(False).astype(bool)
                block |= (delta0 < 0) & f
            if bool(block.any()):
                blocked_n = int(block.sum())
                cur = cur.where(~block, prv)
        if rebalance_band > 0 and len(names):
            # デッドバンド：キャリー後ウェイト基準で閾値未満の注文は出さない
            cur = cur.where((cur - prv).abs() >= rebalance_band, prv)
        w = cur[cur != 0.0]
        if len(w):
            r = float((w * fwd.loc[t].reindex(w.index)).sum())
            npos = int((w != 0).sum())
            if adv is not None and t in adv.index:
                a = adv.loc[t].reindex(w.index)
                wabs = w.abs()
                ok = (wabs > 0) & a.notna() & (a > 0)
                if bool(ok.any()):
                    capacity = min(capacity,
                                   float((participation * a[ok] / wabs[ok]).min()))
        else:
            w, r, npos = pd.Series(dtype="float64"), 0.0, 0
        delta = cur - prv
        turn = float(delta.abs().sum())
        cost = _one_way_cost(delta, t, cp, cp_default)
        short_gross = float(w[w < 0].abs().sum()) if len(w) else 0.0
        net = r - cost - borrow_per_period * short_gross
        rows.append((t, net, r, turn, npos, blocked_n, short_gross))
        prev_w = w
    df = pd.DataFrame(rows, columns=["date", "net", "gross", "turnover",
                                     "npos", "blocked", "short_gross"]
                      ).set_index("date")
    return BacktestResult(df["net"], df["gross"], df["turnover"], df["npos"],
                          _ann_factor(df.index), strategy.name, strategy.params,
                          capacity_jpy=(capacity if capacity < float("inf")
                                        else float("nan")),
                          n_blocked=df["blocked"], short_gross=df["short_gross"])


def open_fill_backtest(weights_by_date: dict, open_daily: pd.DataFrame, *,
                       costs_bps=15.0, short_borrow_bps: float = 0.0,
                       name: str = "open_fill", params: dict | None = None
                       ) -> BacktestResult:
    """低頻度（月次等）の意思決定を「決定日の翌営業日の**始値**」で執行する T+1 リプレイ（DP17）。

    月次ビューの `backtest`（既定 execution_lag=0）は「終値を見て同じ終値で約定」する
    同足執行を含む。本関数は同じ意思決定（PIT ウェイト）を翌営業日の寄付で約定し、次の
    約定日まで保有（open→open で実現）して再評価する＝タイムトラベルの排除と寄り
    ギャップ（窓開け）の負担を同時に課す、最も保守的な執行タイミング検証。

    weights_by_date: {決定日: 目標ウェイト Series}（決定日以前の情報のみで生成済みのこと）。
    open_daily: 日次**調整後始値**の wide（adj_open）。決定日 t の約定日は index 上で
      t より後の最初の営業日。最後の決定は次回約定日が無いため評価から落とす（engine の
      末尾 drop と同じ）。約定日に始値が無い銘柄はその期の損益に寄与しない（fwd NaN と
      同じ扱い）。決定日は月次など十分な間隔を想定（同一約定日への重複は想定外）。
    costs_bps: スカラ bps か「日次日付×銘柄」の bps パネル（**約定日**の行を参照）。
    short_borrow_bps: 年率の貸株コスト（短グロスへ期間按分・backtest と同じ）。
    返り値の index は決定日（backtest と比較しやすい）。
    """
    didx = open_daily.index
    fills = []                                  # (決定日, 約定日, ウェイト)
    for key, wt in sorted(weights_by_date.items(),
                          key=lambda kv: pd.Timestamp(kv[0])):
        t = pd.Timestamp(key)
        pos = didx.searchsorted(t, side="right")
        if pos < len(didx):
            fills.append((t, didx[pos], wt))
    if len(fills) < 2:
        empty = pd.Series(dtype="float64")
        return BacktestResult(empty, empty, empty, empty, float("nan"), name,
                              params or {})
    dates = pd.DatetimeIndex([t for t, _, _ in fills[:-1]])
    borrow_per_period = short_borrow_bps / 1e4 / _ann_factor(dates)
    panel_costs = isinstance(costs_bps, pd.DataFrame)
    cp_default = float(np.nanmedian(costs_bps.to_numpy())) if panel_costs else 0.0
    prev = pd.Series(dtype="float64")
    rows = []
    for k in range(len(fills) - 1):
        t, f0, wt = fills[k]
        f1 = fills[k + 1][1]
        w = wt[wt != 0.0].astype(float)
        if len(w):
            px0 = open_daily.loc[f0].reindex(w.index)
            px1 = open_daily.loc[f1].reindex(w.index)
            r = float((w * (px1 / px0 - 1.0)).sum())
            npos = int(len(w))
        else:
            r, npos = 0.0, 0
        names = w.index.union(prev.index)
        delta = (w.reindex(names).fillna(0.0) - prev.reindex(names).fillna(0.0)) \
            if len(names) else pd.Series(dtype="float64")
        turn = float(delta.abs().sum())
        cost = _one_way_cost(delta, f0, costs_bps, cp_default)
        short_gross = float(w[w < 0].abs().sum()) if len(w) else 0.0
        rows.append((t, r - cost - borrow_per_period * short_gross, r, turn,
                     npos, 0, short_gross))
        prev = w
    df = pd.DataFrame(rows, columns=["date", "net", "gross", "turnover",
                                     "npos", "blocked", "short_gross"]
                      ).set_index("date")
    return BacktestResult(df["net"], df["gross"], df["turnover"], df["npos"],
                          _ann_factor(df.index), name, params or {},
                          n_blocked=df["blocked"], short_gross=df["short_gross"])
