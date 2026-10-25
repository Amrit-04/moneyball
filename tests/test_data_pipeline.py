"""Unit tests for deterministic multi-asset market & factor pipeline."""

import numpy as np
import pandas as pd
import pytest

from quant_engine.data import (
    DEFAULT_ASSETS,
    generate_deterministic_market_panel,
    compute_factor_panel,
    MarketDataLoader,
)


def test_deterministic_reproducibility():
    """Verify that identical seed produces byte-for-byte identical returns and prices."""
    panel1 = generate_deterministic_market_panel(assets=DEFAULT_ASSETS[:5], n_days=100, seed=123)
    panel2 = generate_deterministic_market_panel(assets=DEFAULT_ASSETS[:5], n_days=100, seed=123)
    
    pd.testing.assert_frame_equal(panel1["close"], panel2["close"])
    pd.testing.assert_frame_equal(panel1["returns"], panel2["returns"])
    pd.testing.assert_frame_equal(panel1["volume"], panel2["volume"])


def test_ohlcv_mathematical_bounds():
    """Verify that High is always >= max(Open, Close) and Low <= min(Open, Close)."""
    panel = generate_deterministic_market_panel(assets=DEFAULT_ASSETS[:8], n_days=150, seed=42)
    
    close = panel["close"]
    open_p = panel["open"]
    high = panel["high"]
    low = panel["low"]
    
    max_oc = np.maximum(close, open_p)
    min_oc = np.minimum(close, open_p)
    
    assert (high >= max_oc - 1e-6).all().all(), "High must be greater than or equal to max(Open, Close)"
    assert (low <= min_oc + 1e-6).all().all(), "Low must be less than or equal to min(Open, Close)"
    assert (low > 0.0).all().all(), "All prices must be strictly positive"


def test_factor_engineering_panel():
    """Verify factor engineering computes all required signals without invalid values."""
    panel = generate_deterministic_market_panel(assets=DEFAULT_ASSETS[:6], n_days=250, seed=42)
    factors = compute_factor_panel(panel, seed=42)
    
    expected_cols = [
        "close", "volume", "daily_return",
        "mom_5d", "mom_20d", "mom_60d",
        "realized_vol_20d", "rsi_14d",
        "bollinger_z_20d", "volume_shock", "sentiment_proxy",
        "target_fwd_1d", "target_fwd_5d"
    ]
    for col in expected_cols:
        assert col in factors.columns, f"Missing feature: {col}"
        
    # Check RSI bounds in [0, 100]
    valid_rsi = factors["rsi_14d"].dropna()
    assert (valid_rsi >= 0.0).all() and (valid_rsi <= 100.0).all()
    
    # Check realized volatility is positive
    valid_vol = factors["realized_vol_20d"].dropna()
    assert (valid_vol >= 0.0).all()


def test_loader_cache_integration(tmp_path):
    """Verify MarketDataLoader correctly caches to and loads from disk."""
    loader = MarketDataLoader(cache_dir=tmp_path)
    p1 = loader.load_panel(assets=DEFAULT_ASSETS[:4], n_days=80, seed=99, use_cache=True)
    assert len(p1["close"]) == 80
    
    # Second load should hit cache
    p2 = loader.load_panel(assets=DEFAULT_ASSETS[:4], n_days=80, seed=99, use_cache=True)
    pd.testing.assert_frame_equal(p1["close"], p2["close"])
