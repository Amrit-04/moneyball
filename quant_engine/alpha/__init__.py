"""Alpha generation, regime detection, and statistical validation pipeline."""

from quant_engine.alpha.hmm_regime import MarketRegimeHMM, REGIME_NAMES
from quant_engine.alpha.walk_forward import PurgedWalkForwardCV
from quant_engine.alpha.models import AlphaModelSuite, FEATURE_COLUMNS
from quant_engine.alpha.metrics import (
    compute_daily_rank_ic,
    compute_newey_west_stats,
    compute_alpha_performance_metrics,
)

__all__ = [
    "MarketRegimeHMM",
    "REGIME_NAMES",
    "PurgedWalkForwardCV",
    "AlphaModelSuite",
    "FEATURE_COLUMNS",
    "compute_daily_rank_ic",
    "compute_newey_west_stats",
    "compute_alpha_performance_metrics",
]
