"""Compatibility tests for strategy config fields used at runtime."""

from core.config import get_config


def test_factor_model_config_has_rebalance_days():
    cfg = get_config(reload=True)
    assert hasattr(cfg.strategy.factor_model, "rebalance_days")
    assert cfg.strategy.factor_model.rebalance_days >= 1


def test_stat_arb_config_has_runtime_fields():
    cfg = get_config(reload=True)
    sa = cfg.strategy.stat_arb
    assert hasattr(sa, "coint_pvalue")
    assert hasattr(sa, "half_life_min")
    assert hasattr(sa, "half_life_max")
    assert hasattr(sa, "max_pairs")
    assert 0.0 < sa.coint_pvalue <= 1.0
    assert sa.half_life_min >= 1
    assert sa.half_life_max >= sa.half_life_min
    assert sa.max_pairs >= 1
