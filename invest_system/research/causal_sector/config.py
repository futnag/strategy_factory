"""セクター別変数選択と collider 候補の事前定義。

López de Prado (2023, 2025) の specification error 回避：
- 交絡因子（cause）は調整に含める
- 合流点（collider）は調整から除外（符号反転・mirage の原因）
- セクター特性に応じた外生ドライバを明示的に選ぶ（重工業 vs IT 等）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SectorKind = Literal["heavy_industry", "it_comm", "financial", "energy_util", "default"]


@dataclass(frozen=True)
class SectorProfile:
    """1セクター（S33）の因果分析プロファイル。"""
    s33: str
    name: str
    kind: SectorKind
    drivers: tuple[str, ...]          # macro_extended 列名（外生）
    collider_watch: tuple[str, ...]   # 合流点になりやすい内生変数
    notes: str = ""


# 代表的セクター（可視化・検証用）。全33業種は S33_TO_PROFILE で網羅。
REPRESENTATIVE_SECTORS = ("3300", "3450", "3600", "3650", "5250", "7050")

# 業種種別ごとの外生ドライバテンプレート（日次・macro_extended.parquet）
_DRIVER_TEMPLATES: dict[SectorKind, tuple[str, ...]] = {
    # 重工業：コモディティ・金利・為替
    "heavy_industry": (
        "wti_crude", "copper", "usd_jpy", "japan_10y_yield", "us_10y_yield", "vix",
    ),
    # 情報通信：テックサイクル・金利・ボラ
    "it_comm": (
        "nasdaq", "sp500", "japan_10y_yield", "vix", "usd_jpy",
    ),
    # 金融：金利カーブ・政策金利・ボラ
    "financial": (
        "japan_10y_yield", "japan_policy_rate", "us_10y_yield",
        "us_federal_funds_rate", "vix", "usd_jpy",
    ),
    # エネルギー・公益
    "energy_util": (
        "wti_crude", "japan_10y_yield", "usd_jpy", "vix",
    ),
    "default": (
        "japan_10y_yield", "usd_jpy", "vix", "sp500",
    ),
}

# 合流点になりやすい内生変数（リターンとファクターの両方の「結果」になりうる）
_COLLIDER_WATCH: dict[SectorKind, tuple[str, ...]] = {
    "heavy_industry": ("VOL", "SHORT_RATIO", "MOM"),
    "it_comm": ("VOL", "SHORT_RATIO", "MOM"),
    "financial": ("VOL", "MOM"),
    "energy_util": ("VOL", "SHORT_RATIO"),
    "default": ("VOL", "SHORT_RATIO", "MOM"),
}

# S33 → 業種種別マッピング（日本株33業種＋9999）
_S33_KIND: dict[str, SectorKind] = {
    "3300": "energy_util",       # 石油・石炭
    "3400": "heavy_industry",    # ガラス・土石
    "3450": "heavy_industry",    # 鉄鋼
    "3500": "heavy_industry",    # 非鉄金属
    "3550": "heavy_industry",    # 金属製品
    "3600": "heavy_industry",    # 機械
    "3700": "heavy_industry",    # 輸送用機器
    "3650": "it_comm",           # 電気機器（テックサイクル敏感）
    "3750": "it_comm",           # 精密機器
    "5250": "it_comm",           # 情報・通信
    "7050": "financial",
    "7100": "financial",
    "7150": "financial",
    "7200": "financial",
    "4050": "energy_util",       # 電気・ガス
}


def _profile_for(s33: str, name: str) -> SectorProfile:
    kind = _S33_KIND.get(s33, "default")
    return SectorProfile(
        s33=s33,
        name=name,
        kind=kind,
        drivers=_DRIVER_TEMPLATES[kind],
        collider_watch=_COLLIDER_WATCH[kind],
        notes=f"kind={kind}",
    )


# 全業種名（jquants master 準拠）
_S33_NAMES: dict[str, str] = {
    "0050": "水産・農林業", "1050": "鉱業", "2050": "建設業", "3050": "食料品",
    "3100": "繊維製品", "3150": "パルプ・紙", "3200": "化学", "3250": "医薬品",
    "3300": "石油･石炭製品", "3350": "ゴム製品", "3400": "ガラス･土石製品",
    "3450": "鉄鋼", "3500": "非鉄金属", "3550": "金属製品", "3600": "機械",
    "3650": "電気機器", "3700": "輸送用機器", "3750": "精密機器", "3800": "その他製品",
    "4050": "電気･ガス業", "5050": "陸運業", "5100": "海運業", "5150": "空運業",
    "5200": "倉庫･運輸関連業", "5250": "情報･通信業", "6050": "卸売業",
    "6100": "小売業", "7050": "銀行業", "7100": "証券･商品先物取引業",
    "7150": "保険業", "7200": "その他金融業", "8050": "不動産業",
    "9050": "サービス業", "9999": "その他",
}

SECTOR_PROFILES: dict[str, SectorProfile] = {
    s33: _profile_for(s33, nm) for s33, nm in _S33_NAMES.items()
}

# メタ特徴量用：33業種（9999=観測不足のため除外）
ALL_SECTORS: tuple[str, ...] = tuple(s for s in _S33_NAMES if s != "9999")

# 業種種別（メタ特徴量の kind 集約用・凍結3種）
KIND_EDGE_FEATURES: tuple[str, ...] = ("heavy_industry", "it_comm", "financial")

# 日本株特有イベント（因果構造変化の参照アンカー）
JP_STRUCTURAL_EVENTS: tuple[tuple[str, str], ...] = (
    ("2020-03", "COVIDショック"),
    ("2020-12", "バリュー回帰・グロース崩壊"),
    ("2022-09", "円安・金利上昇局面"),
    ("2023-03", "BOJ YCC柔化"),
    ("2024-03", "企業統治改革・PBR是正加速"),
)


def collider_candidates(var_names: list[str], profile: SectorProfile) -> list[str]:
    """事前定義＋変数名マッチで collider 監視リストを返す。"""
    watch = set(profile.collider_watch)
    out = [v for v in var_names if v in watch]
    # リターンの結果になりうる流動性・空売り系
    for v in var_names:
        if any(k in v.upper() for k in ("SHORT", "MARGIN", "VOL")):
            out.append(v)
    return sorted(set(out))