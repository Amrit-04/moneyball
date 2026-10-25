"""Unit tests for 3-State Gaussian Hidden Markov Model (HMM) regime detection."""

import numpy as np
import pandas as pd
import pytest

from quant_engine.alpha import MarketRegimeHMM
from quant_engine.data import generate_deterministic_market_panel


@pytest.fixture
def market_returns():
    panel = generate_deterministic_market_panel(assets=["SPY"], n_days=500, seed=42)
    return panel["returns"]["SPY"]


def test_hmm_fit_and_states(market_returns):
    """Verify HMM converges with 3 distinct semantic states."""
    hmm = MarketRegimeHMM(n_states=3, random_state=42)
    hmm.fit(market_returns)
    
    assert hmm.is_fitted
    assert len(hmm.state_map) == 3
    
    states = hmm.predict(market_returns)
    assert set(np.unique(states)).issubset({0, 1, 2})
    assert len(states) == len(market_returns)


def test_transition_matrix_stochastic_property(market_returns):
    """Verify transition probability matrix rows sum to 1.0 (Markov property)."""
    hmm = MarketRegimeHMM(n_states=3, random_state=42)
    hmm.fit(market_returns)
    
    trans_mat = hmm.transition_matrix
    assert trans_mat.shape == (3, 3)
    row_sums = np.sum(trans_mat, axis=1)
    np.testing.assert_allclose(row_sums, np.ones(3), atol=1e-5)
    assert (trans_mat >= 0.0).all() and (trans_mat <= 1.0).all()


def test_stationary_distribution_property(market_returns):
    """Verify stationary distribution pi is valid probability vector."""
    hmm = MarketRegimeHMM(n_states=3, random_state=42)
    hmm.fit(market_returns)
    
    pi = hmm.stationary_distribution
    assert len(pi) == 3
    assert np.isclose(np.sum(pi), 1.0, atol=1e-5)
    assert (pi >= 0.0).all()


def test_posterior_probabilities_sum_to_one(market_returns):
    """Verify posterior smoothed probabilities sum to 1 at every time step."""
    hmm = MarketRegimeHMM(n_states=3, random_state=42)
    hmm.fit(market_returns)
    
    proba = hmm.predict_proba(market_returns)
    assert proba.shape == (len(market_returns), 3)
    row_sums = np.sum(proba, axis=1)
    np.testing.assert_allclose(row_sums, np.ones(len(market_returns)), atol=1e-5)
