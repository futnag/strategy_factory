"""月次ライブ監視：因果レジーム・adverse フラグ（docs/39 凍結条件）。

監視専用（LdP: 監視≠毎月売買）。ハイブリッドゲート判定は FAIL だが、
因果エッジ分離は有効（docs/40）→ 月次スナップショットで downside リスクを追跡する。

出力: data/reports/causal_sector/monitor_latest.{json,md}
       data/reports/causal_sector/monitor_YYYY-MM.{json,md}（アーカイブ）
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

from .config import ALL_SECTORS, SECTOR_PROFILES
from .hybrid_gate import (
    DEFAULT_SHRINK_MULT, DEFAULT_UNSTABLE_THRESH, adverse_causal_regime,
)
from .meta import (
    AGG_FEATURES, CACHE_PATH, build_all_sector_monthly,
    build_causal_meta_features_all33,
)

REPORT_DIR = Path(__file__).resolve().parents[3] / "data" / "reports" / "causal_sector"

MonitorStatus = Literal["NORMAL", "CAUTION", "ADVERSE"]


@dataclass(frozen=True)
class AdverseStatus:
    """docs/39 §2.2 の3条件チェック（凍結）。"""
    causal_edge_neg: bool
    edge_financial_neg: bool
    n_unstable_high: bool
    all_met: bool
    missing: bool
    shrink_mult: float = DEFAULT_SHRINK_MULT
    unstable_thresh: int = DEFAULT_UNSTABLE_THRESH

    @property
    def n_conditions_met(self) -> int:
        return int(self.causal_edge_neg) + int(self.edge_financial_neg) + int(
            self.n_unstable_high,
        )


@dataclass
class CausalMonitorSnapshot:
    """1時点の因果監視スナップショット。"""
    asof: pd.Timestamp
    generated_at: datetime
    aggregate: dict[str, Optional[float]]
    adverse: AdverseStatus
    status: MonitorStatus
    sector_edges: pd.DataFrame
    recent_history: pd.DataFrame
    n_sectors: int
    cache_path: Optional[str] = None
    notes: list[str] = field(default_factory=list)


def _status_from_adverse(adv: AdverseStatus) -> MonitorStatus:
    if adv.missing:
        return "CAUTION"
    if adv.all_met:
        return "ADVERSE"
    if adv.n_conditions_met >= 2 or adv.causal_edge_neg:
        return "CAUTION"
    return "NORMAL"


def compute_adverse_status(row: pd.Series, *,
                           unstable_thresh: int = DEFAULT_UNSTABLE_THRESH,
                           ) -> AdverseStatus:
    """1行（月次集約特徴）から adverse 判定。"""
    required = ["causal_edge", "edge_financial", "n_sectors_unstable"]
    if not all(c in row.index for c in required) or row[required].isna().any():
        return AdverseStatus(False, False, False, False, True, unstable_thresh=unstable_thresh)
    ce = bool(row["causal_edge"] < 0)
    ef = bool(row["edge_financial"] < 0)
    nu = bool(row["n_sectors_unstable"] >= unstable_thresh)
    return AdverseStatus(ce, ef, nu, ce and ef and nu, False, unstable_thresh=unstable_thresh)


def _sector_table_at(long_m: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """指定月の業種別エッジ一覧（unstable フラグ付き）。"""
    if long_m.empty or asof not in long_m.index:
        return pd.DataFrame()
    g = long_m.loc[long_m.index == asof].copy()
    if g.empty:
        return pd.DataFrame()
    med_vol = g["edge_volatility"].median()
    g["unstable"] = g["edge_volatility"] > med_vol
    g["sector_name"] = g["s33"].map(
        lambda s: SECTOR_PROFILES[s].name if s in SECTOR_PROFILES else s,
    )
    cols = ["s33", "sector_name", "kind", "causal_edge", "edge_volatility",
            "edge_chg_3m", "months_since_break", "unstable"]
    return g[cols].sort_values("causal_edge")


def _resolve_asof(agg: pd.DataFrame, asof: Optional[pd.Timestamp]) -> pd.Timestamp:
    valid = agg.dropna(how="all")
    if valid.empty:
        raise ValueError("因果集約特徴量が空です")
    if asof is None:
        return valid.index[-1]
    ts = pd.Timestamp(asof)
    if ts not in valid.index:
        prior = valid.index[valid.index <= ts]
        if prior.empty:
            raise ValueError(f"asof={ts:%Y-%m} 以前のデータがありません")
        return prior[-1]
    return ts


def build_monitor_snapshot(*, asof: Optional[pd.Timestamp] = None,
                           refresh_cache: bool = False,
                           verbose: bool = False) -> CausalMonitorSnapshot:
    """最新（または指定）月の監視スナップショットを構築。"""
    notes: list[str] = []
    rebal = pd.date_range("2016-07-31", pd.Timestamp.now(tz=None), freq="ME")
    agg = build_causal_meta_features_all33(
        rebal, verbose=verbose, use_cache=not refresh_cache,
    )
    if refresh_cache:
        notes.append("キャッシュ再構築済み")

    asof_ts = _resolve_asof(agg, asof)
    row = agg.loc[asof_ts]
    adv = compute_adverse_status(row)
    status = _status_from_adverse(adv)

    aggregate = {
        k: (float(row[k]) if k in row.index and pd.notna(row[k]) else None)
        for k in AGG_FEATURES + ("edge_heavy_industry", "edge_it_comm", "edge_financial")
    }

    if verbose:
        print(f"  業種別エッジ（{asof_ts:%Y-%m}）…")
    long_m = build_all_sector_monthly(sectors=list(ALL_SECTORS), verbose=verbose)
    sector_edges = _sector_table_at(long_m, asof_ts)

    hist_cols = list(AGG_FEATURES) + ["edge_financial"]
    recent = agg[hist_cols].dropna(how="all").tail(6)
    if asof_ts not in recent.index:
        recent = pd.concat([recent, agg.loc[[asof_ts], hist_cols]]).tail(6)

    return CausalMonitorSnapshot(
        asof=asof_ts,
        generated_at=datetime.now(timezone.utc),
        aggregate=aggregate,
        adverse=adv,
        status=status,
        sector_edges=sector_edges,
        recent_history=recent,
        n_sectors=len(sector_edges),
        cache_path=str(CACHE_PATH) if CACHE_PATH.exists() else None,
        notes=notes,
    )


def snapshot_to_dict(snap: CausalMonitorSnapshot) -> dict[str, Any]:
    """JSON 直列化用 dict。"""
    adv = asdict(snap.adverse)
    sectors = []
    if not snap.sector_edges.empty:
        for _, r in snap.sector_edges.iterrows():
            sectors.append({
                "s33": r["s33"],
                "name": r["sector_name"],
                "kind": r["kind"],
                "causal_edge": _json_num(r["causal_edge"]),
                "edge_volatility": _json_num(r["edge_volatility"]),
                "unstable": bool(r["unstable"]),
            })
    hist = []
    for dt, r in snap.recent_history.iterrows():
        hist.append({"month": f"{dt:%Y-%m}", **{c: _json_num(r[c]) for c in r.index}})

    return {
        "asof": f"{snap.asof:%Y-%m}",
        "generated_at": snap.generated_at.isoformat(),
        "status": snap.status,
        "adverse": adv,
        "aggregate": snap.aggregate,
        "n_sectors": snap.n_sectors,
        "sectors_negative_edge": sum(1 for s in sectors if s["causal_edge"] is not None and s["causal_edge"] < 0),
        "sectors_unstable": sum(1 for s in sectors if s["unstable"]),
        "sectors": sectors,
        "recent_history": hist,
        "hybrid_gate_reference": {
            "shrink_mult": DEFAULT_SHRINK_MULT,
            "unstable_thresh": DEFAULT_UNSTABLE_THRESH,
            "would_shrink": snap.adverse.all_met,
        },
        "cache_path": snap.cache_path,
        "notes": snap.notes,
    }


def _json_num(v) -> Optional[float]:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    return float(v)


def format_monitor_md(snap: CausalMonitorSnapshot) -> str:
    """人間可読の Markdown レポート。"""
    a = snap.aggregate
    adv = snap.adverse
    lines = [
        "# 因果レジーム月次監視",
        "",
        f"> asof: **{snap.asof:%Y-%m}**  |  status: **{snap.status}**  |  "
        f"generated: {snap.generated_at:%Y-%m-%d %H:%M UTC}",
        "",
        "## 横断集約（全33業種）",
        "",
        "| 指標 | 値 | 閾値/解釈 |",
        "|---|--:|---|",
        f"| causal_edge | {a.get('causal_edge', '—')} | <0 = バリュー逆風 |",
        f"| edge_financial | {a.get('edge_financial', '—')} | <0 = 金融セクター逆風 |",
        f"| n_sectors_unstable | {a.get('n_sectors_unstable', '—')} | ≥{adv.unstable_thresh} = 不安定多数 |",
        f"| edge_volatility | {a.get('edge_volatility', '—')} | エッジの横断ボラ |",
        f"| edge_chg_3m | {a.get('edge_chg_3m', '—')} | 3ヶ月変化 |",
        f"| months_since_break | {a.get('months_since_break', '—')} | 直近構造ブレイクからの月数 |",
        f"| edge_heavy_industry | {a.get('edge_heavy_industry', '—')} | 重工業 kind 平均 |",
        f"| edge_it_comm | {a.get('edge_it_comm', '—')} | IT・通信 kind 平均 |",
        "",
        "## adverse 判定（docs/39 凍結・監視専用）",
        "",
        f"- causal_edge < 0: **{'✓' if adv.causal_edge_neg else '✗'}**",
        f"- edge_financial < 0: **{'✓' if adv.edge_financial_neg else '✗'}**",
        f"- n_sectors_unstable ≥ {adv.unstable_thresh}: **{'✓' if adv.n_unstable_high else '✗'}**",
        f"- **adverse（3条件AND）**: {'**はい**' if adv.all_met else 'いいえ'}",
        f"- ハイブリッド縮小参考（0.5x）: {'適用候補' if adv.all_met else '該当なし'}",
        "",
        "> 注: docs/40 でハイブリッドゲートは FAIL。本レポートは**監視のみ**（自動売買なし）。",
        "",
    ]

    if not snap.sector_edges.empty:
        neg = snap.sector_edges[snap.sector_edges["causal_edge"] < 0]
        unst = snap.sector_edges[snap.sector_edges["unstable"]]
        lines += [
            "## 業種別（エッジ昇順）",
            "",
            f"観測 {snap.n_sectors} 業種 / エッジ負 {len(neg)} / 不安定 {len(unst)}",
            "",
            "| S33 | 業種 | kind | edge | vol | unstable |",
            "|---|---|---|--:|--:|:---:|",
        ]
        for _, r in snap.sector_edges.iterrows():
            flag = "⚠" if r["unstable"] else ""
            lines.append(
                f"| {r['s33']} | {r['sector_name']} | {r['kind']} | "
                f"{r['causal_edge']:.4f} | {r['edge_volatility']:.4f} | {flag} |",
            )
        lines.append("")

    if not snap.recent_history.empty:
        lines += ["## 直近6ヶ月", ""]
        lines.append("| 月 | causal_edge | edge_financial | n_unstable |")
        lines.append("|---|---:|---:|---:|")
        for dt, r in snap.recent_history.iterrows():
            ce = r.get("causal_edge", np.nan)
            ef = r.get("edge_financial", np.nan)
            nu = r.get("n_sectors_unstable", np.nan)
            lines.append(
                f"| {dt:%Y-%m} | {ce:.4f} | {ef:.4f} | {int(nu) if pd.notna(nu) else '—'} |",
            )
        lines.append("")

    return "\n".join(lines)


def write_monitor_report(snap: CausalMonitorSnapshot, *,
                         report_dir: Optional[Path] = None,
                         archive: bool = True) -> tuple[Path, Path]:
    """latest + 月次アーカイブを書き出す。戻り値 (json_path, md_path)。"""
    report_dir = report_dir or REPORT_DIR
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = snapshot_to_dict(snap)
    tag = f"{snap.asof:%Y-%m}"

    json_latest = report_dir / "monitor_latest.json"
    md_latest = report_dir / "monitor_latest.md"
    json_latest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_latest.write_text(format_monitor_md(snap), encoding="utf-8")

    if archive:
        (report_dir / f"monitor_{tag}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        (report_dir / f"monitor_{tag}.md").write_text(
            format_monitor_md(snap), encoding="utf-8",
        )
    return json_latest, md_latest