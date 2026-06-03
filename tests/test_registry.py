"""試行レジストリの事前登録ゲート・不変性・DSR自動算出を検証。"""
import numpy as np
import pytest

from invest_system.validation.registry import TrialRegistry
from invest_system.validation import dsr as dsrmod


def _reg():
    return TrialRegistry(":memory:")


def test_preregistration_requires_hypothesis_and_rationale():
    reg = _reg()
    with pytest.raises(ValueError):
        reg.preregister(scope="s", hypothesis="", economic_rationale="valid rationale text")
    with pytest.raises(ValueError):
        reg.preregister(scope="s", hypothesis="valid hypothesis text", economic_rationale="x")


def test_record_requires_preregistration():
    reg = _reg()
    with pytest.raises(KeyError):
        reg.record_result("does-not-exist", sharpe=0.1, n_obs=100, skew=0.0, kurt=3.0)


def test_results_are_immutable():
    reg = _reg()
    tid = reg.preregister(scope="cryptoMR",
                          hypothesis="mean reversion in BTC dollar bars",
                          economic_rationale="liquidity rebate microstructure")
    reg.record_result(tid, sharpe=0.1, n_obs=250, skew=0.0, kurt=3.0)
    with pytest.raises(ValueError):
        reg.record_result(tid, sharpe=0.2, n_obs=250, skew=0.0, kurt=3.0)


def test_trial_count_is_scoped():
    reg = _reg()
    for scope in ("famA", "famA", "famB"):
        t = reg.preregister(scope=scope, hypothesis="hypothesis text here",
                            economic_rationale="rationale text here")
        reg.record_result(t, sharpe=0.1, n_obs=250, skew=0.0, kurt=3.0)
    assert reg.trial_count("famA") == 2
    assert reg.trial_count("famB") == 1
    assert reg.trial_count("unknown") == 0


def test_preregistered_not_counted_until_completed():
    reg = _reg()
    reg.preregister(scope="fam", hypothesis="hypothesis text here",
                    economic_rationale="rationale text here")
    assert reg.trial_count("fam") == 0  # 未完了はKに数えない


def test_sharpe_variance_and_deflated_sharpe():
    reg = _reg()
    sharpes = [0.05, 0.10, 0.15, 0.20]
    tids = []
    for s in sharpes:
        t = reg.preregister(scope="fam", hypothesis="hypothesis text here",
                            economic_rationale="rationale text here")
        reg.record_result(t, sharpe=s, n_obs=250, skew=0.0, kurt=3.0)
        tids.append(t)
    assert reg.trial_count("fam") == 4
    assert reg.sharpe_variance("fam") == pytest.approx(np.var(sharpes, ddof=1))

    dsr_best = reg.deflated_sharpe(tids[-1])
    assert 0.0 <= dsr_best <= 1.0
    direct = dsrmod.deflated_sharpe_ratio(
        sr=0.20, sr_variance=np.var(sharpes, ddof=1), n_trials=4,
        n_obs=250, skew=0.0, kurt=3.0,
    )
    assert dsr_best == pytest.approx(direct)


def test_deflated_sharpe_requires_completed_trial():
    reg = _reg()
    tid = reg.preregister(scope="fam", hypothesis="hypothesis text here",
                          economic_rationale="rationale text here")
    with pytest.raises(ValueError):
        reg.deflated_sharpe(tid)
