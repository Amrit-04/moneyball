"""CPU Baseline Monte Carlo Engines.

Provides:
1. CpuVectorizedMonteCarloEngine: High-performance vectorized NumPy baseline.
2. CpuLoopMonteCarloEngine: Unvectorized single-thread reference loop for benchmarking speedup multipliers.
"""

import time
from typing import Dict, Optional, Tuple
import numpy as np

from quant_engine.risk_cuda.risk_metrics import compute_var_cvar


class CpuVectorizedMonteCarloEngine:
    """Vectorized NumPy Monte Carlo baseline on CPU."""
    
    def simulate_portfolio_returns(
        self,
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray,
        weights: np.ndarray,
        n_paths: int = 1_000_000,
        horizon_days: int = 10,
        lambda_jump: float = 3.0,
        jump_mean: float = -0.02,
        jump_std: float = 0.04,
        seed: Optional[int] = 42,
    ) -> Tuple[np.ndarray, float]:
        """Simulate portfolio returns using vectorized NumPy operations."""
        rng = np.random.default_rng(seed)
        n_assets = len(mean_returns)
        dt = horizon_days / 252.0
        sqrt_dt = np.sqrt(dt)
        
        jitter = 1e-7 * np.eye(n_assets)
        cov_reg = cov_matrix + jitter
        L = np.linalg.cholesky(cov_reg)
        vols = np.sqrt(np.diag(cov_matrix))
        
        kappa = np.exp(jump_mean + 0.5 * jump_std**2) - 1.0
        drift = (mean_returns - lambda_jump * kappa - 0.5 * (vols ** 2)) * dt
        
        start_time = time.perf_counter()
        
        # 1. Correlated Brownian increments
        z_uncorr = rng.standard_normal((n_paths, n_assets))
        diff_term = (z_uncorr @ L.T) * sqrt_dt
        
        # 2. Poisson Jumps
        poisson_rate = lambda_jump * dt
        jump_counts = rng.poisson(poisson_rate, size=(n_paths, n_assets))
        jump_noise = rng.standard_normal((n_paths, n_assets))
        jump_term = jump_counts * jump_mean + np.sqrt(jump_counts) * jump_std * jump_noise
        
        # Total log-return increment and terminal portfolio return
        delta_ln_S = drift + diff_term + jump_term
        asset_returns = np.expm1(delta_ln_S)
        port_returns = asset_returns @ weights
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return port_returns, float(elapsed_ms)

    def compute_risk(
        self,
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray,
        weights: np.ndarray,
        n_paths: int = 1_000_000,
        horizon_days: int = 10,
        seed: Optional[int] = 42,
    ) -> Dict[str, float]:
        returns, elapsed_ms = self.simulate_portfolio_returns(
            mean_returns=mean_returns,
            cov_matrix=cov_matrix,
            weights=weights,
            n_paths=n_paths,
            horizon_days=horizon_days,
            seed=seed,
        )
        metrics = compute_var_cvar(returns)
        metrics["elapsed_ms"] = elapsed_ms
        metrics["throughput_paths_sec"] = (n_paths / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0.0
        metrics["device"] = "cpu_numpy_vectorized"
        return metrics


class CpuLoopMonteCarloEngine:
    """Unvectorized single-thread path-by-path loop baseline."""
    
    def simulate_portfolio_returns(
        self,
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray,
        weights: np.ndarray,
        n_paths: int = 10_000,
        horizon_days: int = 10,
        lambda_jump: float = 3.0,
        jump_mean: float = -0.02,
        jump_std: float = 0.04,
        seed: Optional[int] = 42,
    ) -> Tuple[np.ndarray, float]:
        """Simulate portfolio returns sequentially path-by-path."""
        rng = np.random.default_rng(seed)
        n_assets = len(mean_returns)
        dt = horizon_days / 252.0
        sqrt_dt = np.sqrt(dt)
        
        L = np.linalg.cholesky(cov_matrix + 1e-7 * np.eye(n_assets))
        vols = np.sqrt(np.diag(cov_matrix))
        kappa = np.exp(jump_mean + 0.5 * jump_std**2) - 1.0
        drift = (mean_returns - lambda_jump * kappa - 0.5 * (vols ** 2)) * dt
        poisson_rate = lambda_jump * dt
        
        port_returns = np.zeros(n_paths)
        start_time = time.perf_counter()
        
        for m in range(n_paths):
            z = rng.standard_normal(n_assets)
            diff = (L @ z) * sqrt_dt
            j_cnt = rng.poisson(poisson_rate, size=n_assets)
            j_term = np.zeros(n_assets)
            for i in range(n_assets):
                if j_cnt[i] > 0:
                    j_term[i] = rng.normal(j_cnt[i] * jump_mean, np.sqrt(j_cnt[i]) * jump_std)
            d_ln_s = drift + diff + j_term
            r_assets = np.exp(d_ln_s) - 1.0
            port_returns[m] = np.dot(r_assets, weights)
            
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return port_returns, float(elapsed_ms)

    def compute_risk(
        self,
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray,
        weights: np.ndarray,
        n_paths: int = 10_000,
        horizon_days: int = 10,
        seed: Optional[int] = 42,
    ) -> Dict[str, float]:
        returns, elapsed_ms = self.simulate_portfolio_returns(
            mean_returns=mean_returns,
            cov_matrix=cov_matrix,
            weights=weights,
            n_paths=n_paths,
            horizon_days=horizon_days,
            seed=seed,
        )
        metrics = compute_var_cvar(returns)
        metrics["elapsed_ms"] = elapsed_ms
        metrics["throughput_paths_sec"] = (n_paths / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0.0
        metrics["device"] = "cpu_unvectorized_loop"
        return metrics
