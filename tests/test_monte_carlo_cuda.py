"""Unit tests for CUDA and CPU Monte Carlo VaR & CVaR Risk Engine."""

import numpy as np
import pytest
import torch

from quant_engine.risk_cuda import (
    compute_var_cvar,
    CudaMonteCarloEngine,
    CpuVectorizedMonteCarloEngine,
    MonteCarloRiskEngine,
)


@pytest.fixture
def risk_inputs():
    n_assets = 10
    rng = np.random.default_rng(42)
    mu = np.full(n_assets, 0.10)
    
    # Generate positive definite covariance matrix
    A = rng.standard_normal((n_assets, n_assets))
    cov = 0.04 * (A @ A.T) / n_assets + 0.02 * np.eye(n_assets)
    weights = np.ones(n_assets) / n_assets
    return mu, cov, weights


def test_var_cvar_mathematical_invariance():
    """Verify fundamental risk bound: CVaR_alpha >= VaR_alpha for all alpha in (0, 1)."""
    rng = np.random.default_rng(42)
    # Generate skewed heavy-tailed returns
    synthetic_rets = rng.standard_t(df=4, size=50_000) * 0.02 - 0.001
    
    metrics = compute_var_cvar(synthetic_rets, confidence_levels=(0.95, 0.99))
    
    assert metrics["CVaR_95"] >= metrics["VaR_95"] - 1e-7, "CVaR_95 must be >= VaR_95"
    assert metrics["CVaR_99"] >= metrics["VaR_99"] - 1e-7, "CVaR_99 must be >= VaR_99"
    assert metrics["VaR_99"] >= metrics["VaR_95"] - 1e-7, "VaR_99 must be >= VaR_95"
    assert metrics["CVaR_99"] >= metrics["CVaR_95"] - 1e-7, "CVaR_99 must be >= CVaR_95"


def test_cuda_and_cpu_vectorized_consistency(risk_inputs):
    """Verify that CUDA and CPU vectorized backends produce consistent estimates within statistical error."""
    mu, cov, weights = risk_inputs
    
    cuda_eng = CudaMonteCarloEngine()
    cpu_eng = CpuVectorizedMonteCarloEngine()
    
    n_paths = 50_000
    res_cuda = cuda_eng.compute_risk(mu, cov, weights, n_paths=n_paths, seed=42)
    res_cpu = cpu_eng.compute_risk(mu, cov, weights, n_paths=n_paths, seed=42)
    
    # Check VaR and CVaR are within 5% relative tolerance
    assert np.isclose(res_cuda["VaR_95"], res_cpu["VaR_95"], rtol=0.08)
    assert np.isclose(res_cuda["CVaR_95"], res_cpu["CVaR_95"], rtol=0.08)
    assert np.isclose(res_cuda["VaR_99"], res_cpu["VaR_99"], rtol=0.08)
    assert np.isclose(res_cuda["CVaR_99"], res_cpu["CVaR_99"], rtol=0.08)


def test_monte_carlo_engine_speedup_profiler(risk_inputs):
    """Verify benchmark_speedup execution and speedup metrics."""
    mu, cov, weights = risk_inputs
    engine = MonteCarloRiskEngine()
    
    profile = engine.benchmark_speedup(mu, cov, weights, n_paths=20_000)
    assert "cuda_latency_ms" in profile
    assert "speedup_vs_cpu_vectorized" in profile
    assert "speedup_vs_cpu_unvectorized_loop" in profile
    assert profile["CVaR_95"] >= profile["VaR_95"]
