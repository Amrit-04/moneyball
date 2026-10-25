"""Unit tests for Purged & Embargoed Walk-Forward Cross-Validation."""

import numpy as np
import pandas as pd
import pytest

from quant_engine.alpha import (
    PurgedWalkForwardCV,
    AlphaModelSuite,
    compute_alpha_performance_metrics,
)
from quant_engine.data import (
    DEFAULT_ASSETS,
    generate_deterministic_market_panel,
    compute_factor_panel,
)


@pytest.fixture
def sample_factor_panel():
    panel = generate_deterministic_market_panel(assets=DEFAULT_ASSETS[:8], n_days=700, seed=42)
    factors = compute_factor_panel(panel, seed=42)
    return factors


def test_purged_walk_forward_leakage_prevention(sample_factor_panel):
    """Verify that training set and test set are strictly disjoint in time with purge separation."""
    train_window = 252
    test_window = 63
    purge_window = 5
    embargo_window = 5
    
    cv = PurgedWalkForwardCV(
        n_splits=3,
        train_window=train_window,
        test_window=test_window,
        purge_window=purge_window,
        embargo_window=embargo_window,
    )
    
    splits = list(cv.split(sample_factor_panel))
    assert len(splits) >= 2
    
    for fold, (train_slice, test_slice, train_dates, test_dates) in enumerate(splits):
        # 1. Zero date overlap between train and test
        overlap = set(train_dates).intersection(set(test_dates))
        assert len(overlap) == 0, f"Fold {fold}: Train and test dates overlap!"
        
        # 2. Strict chronological ordering: all train dates before test dates
        assert max(train_dates) < min(test_dates), f"Fold {fold}: Lookahead bias detected!"
        
        # 3. Purge buffer gap: min(test) - max(train) >= purge_window trading days
        gap_days = (min(test_dates) - max(train_dates)).days
        assert gap_days >= purge_window, f"Fold {fold}: Purge gap violated ({gap_days} < {purge_window})!"


def test_alpha_model_suite_fitting_and_metrics(sample_factor_panel):
    """Verify Ridge, Random Forest, and Gradient Boosting fit and generate valid metrics."""
    train_slice = sample_factor_panel.iloc[:1000]
    test_slice = sample_factor_panel.iloc[1000:1500]
    
    X_train = train_slice
    y_train = train_slice["target_fwd_1d"]
    X_test = test_slice
    y_test = test_slice["target_fwd_1d"]
    
    suite = AlphaModelSuite(random_state=42)
    suite.fit(X_train, y_train)
    
    for name in ["Ridge", "RandomForest", "GradientBoosting"]:
        preds = suite.predict(X_test, model_name=name)
        assert len(preds) == len(X_test)
        assert not np.isnan(preds).any()
        
        importances = suite.get_feature_importances(model_name=name)
        assert len(importances) == 8
        assert np.isclose(sum(importances.values()), 1.0, atol=1e-3)
