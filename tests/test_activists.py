"""アクティビスト名簿（CSV ローダー・名寄せ・purpose フィルタ）の検証。"""
from invest_system.equities import activists as av


def test_registry_loads_and_styles():
    df = av.registry()
    assert len(df) >= 30                                  # 34 グループ
    assert set(df["style"]).issubset(set(av.STYLES))     # hard/soft のみ
    assert df["group_slug"].is_unique


def test_resolve_by_edinet_code_and_name():
    assert av.resolve(edinet_code="E11852")["group_slug"] == "effissimo"
    m = av.resolve("エフィッシモ・キャピタル・マネージメント株式会社")
    assert m is not None and m["group_slug"] == "effissimo"


def test_canonical_group_dedups_families():
    # ダルトン LLC/Inc は別 EDINET コードでも同一グループに dedup
    assert av.canonical_group_of(edinet_code="E08827") == "dalton"
    assert av.canonical_group_of(edinet_code="E39237") == "dalton"
    # 村上系（city11/reno/南青山）も murakami に集約
    assert av.canonical_group_of(edinet_code="E24231") == "murakami"   # reno コード
    assert av.canonical_group_of("南青山不動産") == "murakami"
    assert av.canonical_group_of("みさき投資") == "misaki"


def test_style_hard_soft():
    assert av.style_of(edinet_code="E11852") == "engagement_hard"
    assert av.style_of("みさき投資") == "engagement_soft"


def test_is_important_proposal_filter():
    # docs/10 §7：保有目的テキストで機械判定（registry 掲載に依らない）
    yes = "投資及び状況に応じて経営陣への助言、重要提案行為等を行うこと"
    assert av.is_important_proposal(yes)
    assert not av.is_important_proposal("純投資")
    assert not av.is_important_proposal(None)


def test_unlisted_holder_returns_none():
    assert av.resolve("全く無関係の投資ファンド合同会社") is None
    assert av.style_of("無関係ファンド") is None


def test_short_alias_no_false_positive():
    # "MI2"/"UGS" 等の短い別名は完全一致のみ＝無関係名に誤マッチしない
    assert av.resolve("MI2 something unrelated corp") is None or \
        av.resolve("MI2 something unrelated corp")["group_slug"] == "mi2"
    assert av.resolve("全く別のUGS的な何か") is None
