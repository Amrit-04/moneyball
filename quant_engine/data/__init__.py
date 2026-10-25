"""Data ingestion, synthetic multi-asset panel generation, and factor feature engineering."""

from quant_engine.data.synthetic import (
    DEFAULT_ASSETS,
    generate_deterministic_market_panel,
    build_calibrated_covariance,
)
from quant_engine.data.features import compute_factor_panel, compute_rsi
from quant_engine.data.loader import MarketDataLoader

__all__ = [
    "DEFAULT_ASSETS",
    "generate_deterministic_market_panel",
    "build_calibrated_covariance",
    "compute_factor_panel",
    "compute_rsi",
    "MarketDataLoader",
]
