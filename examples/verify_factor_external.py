"""P1-A 補強: 自前ファクターの外部データ突合（Kenneth French 日本ファクター）。

docs/04 P1-A の任意補強・docs/03 §6.20 の残課題：実装リスク（A2）を「独立エンジン」
（§6.20 で機械精度一致を確認済み）に加えて「**独立データ源**」でも裏取りする。
自前 J-Quants ファクター（value=B/M zN・momentum=12-1 zN・月次 L/S・グロス）と
Kenneth French Data Library の日本ファクター（HML・WML, 無料 CSV・WRDS 不要）の
月次リターンを突合する。

判定基準（docs/04 P1-A の「符号が逆・相関がほぼゼロならバグを疑う」を実装）：
**完全一致は期待しない**（ユニバース=全市場VW 2x3 vs PIT上位300 EW分位・セクター
中立化の有無等の構築差）。WARN（exit 2）＝ (a) 相関 < 0.15（系列が無関係＝構築バグの
主シグナル）、または (b) 年率平均の符号が逆かつ**両者とも経済的に有意（|平均|≥2%/年）**。
(b) の下限が無いと「プレミアム≈0 のファクター（日本のモメンタム等）でほぼゼロ同士の
符号違い」が偽警報になる（バグなら系列相関ごと壊れるので (a) が捕まえる）。

タイミング整合（重要）：自前系列の「決定月 t」のリターンは t→t+1 で実現＝**French の
暦月 t+1** に対応する。1ヶ月ずらして結合しないと相関が大きく減衰する。

throwaway：建玉なし・K 消費ゼロ・レジストリ不使用。data/external_factors/ にキャッシュ。
JKP（jkpfactors.com）は JS 駆動で CLI 直取得不可のため、手動 DL 時の比較は将来課題。

実行: .venv\\Scripts\\python.exe examples\\verify_factor_external.py
"""
from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from invest_system.config import get_env  # noqa: E402
from invest_system.data.sources import jquants as jq  # noqa: E402
from invest_system.equities.universe import (  # noqa: E402
    apply_universe_mask, filter_common_stocks, point_in_time_universe,
    universe_members,
)
from invest_system.equities.panel import (  # noqa: E402
    assemble_panel, fetch_month_end_snapshots, trailing_momentum,
)
from invest_system.equities.fundamentals import load_fundamentals, point_in_time  # noqa: E402
from invest_system.equities.factors import (  # noqa: E402
    cross_sectional_zscore, sector_neutralize, value_quality_size_factors,
)
from invest_system.research import AsOfView, CrossSectionalStrategy, backtest  # noqa: E402

START, END = "2016-07", "2026-05"
CACHE = Path("data/external_factors")
_FF = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
FF_FILES = {"3factors": f"{_FF}/Japan_3_Factors_CSV.zip",
            "momentum": f"{_FF}/Japan_Mom_Factor_CSV.zip"}
CORR_WARN = 0.15      # 相関がこれ未満 → 構築バグを疑う（主シグナル）
SIGN_MIN = 0.02       # 符号逆の判定は両者 |年率平均| がこれ以上の時のみ（偽警報防止）


def _fetch_french(key: str) -> pd.DataFrame:
    """French CSV(zip) を取得・キャッシュし、月次 % リターンの DataFrame を返す。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    fp = CACHE / f"ff_japan_{key}.parquet"
    if fp.exists():
        return pd.read_parquet(fp)
    with urllib.request.urlopen(FF_FILES[key], timeout=60) as resp:
        zf = zipfile.ZipFile(io.BytesIO(resp.read()))
    text = zf.read(zf.namelist()[0]).decode("utf-8", "ignore")
    rows, header = [], None
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if parts[0][:6].isdigit() and len(parts[0]) == 6:
            if header is None:
                continue
            rows.append(parts)
        elif len(parts) > 1 and parts[0] == "" and header is None:
            header = [p for p in parts[1:]]
        elif rows:
            break                                  # 月次ブロック終端（年次表の前）
    df = pd.DataFrame(rows, columns=["yyyymm"] + header)
    df["yyyymm"] = pd.PeriodIndex(pd.to_datetime(df["yyyymm"], format="%Y%m"),
                                  freq="M")
    df = df.set_index("yyyymm").apply(pd.to_numeric, errors="coerce") / 100.0
    df.to_parquet(fp)
    return df


def _own_factor_series() -> dict[str, pd.Series]:
    """自前 value/momentum の月次 L/S（§6.10 構成・グロス＝構築の突合用にコスト0）。"""
    listed = jq.fetch_listed_info()
    snaps = fetch_month_end_snapshots(START, END)
    adj, raw, turn = (assemble_panel(snaps, c) for c in ("AdjC", "C", "Va"))
    common = set(filter_common_stocks(listed)["Code"].astype(str))
    turn_c = turn[[c for c in turn.columns if str(c) in common]]
    umask = point_in_time_universe(turn_c, top_n=300, lookback=12, min_obs=6)
    superset = universe_members(umask)
    adj, raw = adj.reindex(columns=superset), raw.reindex(columns=superset)
    umask = umask.reindex(columns=superset).fillna(False)
    sector = listed.assign(Code=listed["Code"].astype(str)).set_index("Code")["S33"]
    rebal = adj.index
    view = AsOfView({"close": adj})
    fund = load_fundamentals(superset)

    def zN(f):
        return cross_sectional_zscore(sector_neutralize(apply_universe_mask(f, umask),
                                                        sector))
    pit = point_in_time(fund, rebal, ["ShOutFY", "TrShFY", "Eq"], lag_days=1)
    value = zN(value_quality_size_factors(pit, raw, adj)["book_to_market"])
    mom = zN(trailing_momentum(adj, lookback=12, skip=1))
    out = {}
    for nm, fac in (("value", value), ("momentum", mom)):
        s = CrossSectionalStrategy(fac, 0.2, name=nm)
        out[nm] = backtest(s, view, costs_bps=0.0).returns.dropna()
    return out


def _crosscheck(own: pd.Series, ext: pd.Series, own_name: str,
                ext_name: str) -> bool:
    """符号・相関の突合。決定月 t の自前リターン ↔ French 暦月 t+1。"""
    own_p = pd.Series(own.values,
                      index=pd.PeriodIndex(own.index, freq="M") + 1)
    both = pd.concat({"own": own_p, "ext": ext}, axis=1).dropna()
    n = len(both)
    corr = float(both["own"].corr(both["ext"]))
    mu_o = float(both["own"].mean()) * 12
    mu_e = float(both["ext"].mean()) * 12
    sign_flip = (np.sign(mu_o) != np.sign(mu_e)
                 and min(abs(mu_o), abs(mu_e)) >= SIGN_MIN)
    ok = corr >= CORR_WARN and not sign_flip
    note = "" if min(abs(mu_o), abs(mu_e)) >= SIGN_MIN or \
        np.sign(mu_o) == np.sign(mu_e) else "（平均≈0同士の符号差＝許容）"
    print(f"  {own_name:<10} vs {ext_name:<4}  n={n:>3}  corr={corr:+.2f}  "
          f"年率平均 自前{mu_o:+.1%} / French{mu_e:+.1%}  "
          f"{'OK' if ok else 'WARN（構築バグを疑え）'}{note}")
    return ok


def main() -> int:
    if not get_env("J_QUANTS_API_KEY"):
        print("ERROR: .env に J_QUANTS_API_KEY が必要です。")
        return 1
    print("=== P1-A 補強：外部データ突合（Kenneth French 日本ファクター・"
          f"WARN基準 corr<{CORR_WARN:g} or 符号逆）===")
    ff3 = _fetch_french("3factors")
    ffm = _fetch_french("momentum")
    hml = ff3["HML"].dropna()
    wml = ffm.iloc[:, -1].dropna()                 # 列名は WML/Mom 系で末尾列
    print(f"French: HML {ff3.index.min()}〜{ff3.index.max()} / "
          f"momentum 列={ffm.columns[-1]}")

    print("自前ファクター組立中（§6.10 構成・グロス）...")
    own = _own_factor_series()
    ok_v = _crosscheck(own["value"], hml, "value(B/M)", "HML")
    ok_m = _crosscheck(own["momentum"], wml, "mom(12-1)", "WML")

    print("\n※ 読み方：構築差（全市場VW 2x3 vs PIT上位300 EW分位・セクター中立化）が"
          "あるため中程度の相関で正常。corr がほぼゼロ・符号が逆なら J-Quants 由来の"
          "自前構築（財務の PIT 整合・調整係数・ユニバース）のバグを疑う（docs/04 P1-A）。")
    return 0 if (ok_v and ok_m) else 2


if __name__ == "__main__":
    raise SystemExit(main())
