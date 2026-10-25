"""GPU / CUDA-Accelerated Correlated Monte Carlo Engine.

Uses PyTorch CUDA tensors, cuBLAS matrix multiplication, and GPU random number
generators to simulate millions of correlated asset price paths under Merton
Jump-Diffusion in milliseconds.
"""

import time
from typing import Dict, Optional, Tuple
import numpy as np
import torch

from quant_engine.risk_cuda.risk_metrics import compute_var_cvar


class CudaMonteCarloEngine:
    """PyTorch CUDA Monte Carlo simulation engine for multi-asset risk."""
    
    def __init__(self, device: Optional[str] = None):
        """Initialize CUDA engine.
        
        Args:
            device: 'cuda', 'cuda:0', or 'cpu'. Defaults to 'cuda' if available.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.is_cuda = self.device.type == "cuda"
        
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
        """Simulate portfolio terminal returns under correlated Merton Jump-Diffusion.
        
        Args:
            mean_returns: Annualized expected returns vector (N,).
            cov_matrix: Annualized covariance matrix (N, N).
            weights: Portfolio asset weights vector (N,).
            n_paths: Number of Monte Carlo paths (e.g. 100,000 to 1,000,000).
            horizon_days: Risk horizon in trading days (default 10 days).
            lambda_jump: Expected jump frequency per asset per year.
            jump_mean: Mean of log-jump size.
            jump_std: Standard deviation of log-jump size.
            seed: RNG seed.
            
        Returns:
            Tuple of (portfolio_returns_array, elapsed_time_ms)
        """
        if seed is not None:
            torch.manual_seed(seed)
            if self.is_cuda:
                torch.cuda.manual_seed_all(seed)
                
        n_assets = len(mean_returns)
        dt = horizon_days / 252.0
        sqrt_dt = np.sqrt(dt)
        
        # Cholesky factor of covariance matrix on CPU/GPU
        # Add small ridge if near singular
        jitter = 1e-7 * np.eye(n_assets)
        cov_reg = cov_matrix + jitter
        L_np = np.linalg.cholesky(cov_reg)
        
        # Warmup and synchronization
        if self.is_cuda:
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            start_time = time.perf_counter()
        else:
            start_time = time.perf_counter()
            
        # Move tensors to device
        t_mu = torch.as_tensor(mean_returns, dtype=torch.float32, device=self.device)
        t_weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        t_L = torch.as_tensor(L_np, dtype=torch.float32, device=self.device)
        t_vols = torch.sqrt(torch.diag(torch.as_tensor(cov_matrix, dtype=torch.float32, device=self.device)))
        
        # Jump compensator: kappa = exp(mu_J + 0.5 * sigma_J^2) - 1
        kappa = np.exp(jump_mean + 0.5 * jump_std**2) - 1.0
        drift = (t_mu - lambda_jump * kappa - 0.5 * (t_vols ** 2)) * dt
        
        # 1. Correlated Brownian shocks: Z ~ N(0, I) -> Z_corr = Z @ L.T * sqrt(dt)
        # Note: L is lower triangular of cov_matrix, so Z @ L.T gives covariance cov_matrix * dt
        z_uncorr = torch.randn((n_paths, n_assets), dtype=torch.float32, device=self.device)
        # Diffusion term has covariance cov_matrix * dt
        diff_term = torch.matmul(z_uncorr, t_L.t()) * sqrt_dt
        
        # 2. Merton Jump shocks
        # Poisson jump counts per path: Poisson(lambda * dt)
        poisson_rate = lambda_jump * dt
        jump_counts = torch.poisson(torch.full((n_paths, n_assets), poisson_rate, dtype=torch.float32, device=self.device))
        
        # Total jump magnitude
        # Approximate composite jump for Poisson counts: N(k * jump_mean, k * jump_std^2)
        jump_noise = torch.randn((n_paths, n_assets), dtype=torch.float32, device=self.device)
        jump_term = jump_counts * jump_mean + torch.sqrt(jump_counts) * jump_std * jump_noise
        
        # Total log-return increment: Delta ln S = drift + diff_term + jump_term
        delta_ln_S = drift + diff_term + jump_term
        
        # Terminal asset return: exp(Delta ln S) - 1
        asset_returns = torch.expm1(delta_ln_S)
        
        # Portfolio return: R_p = asset_returns @ weights
        port_returns_t = torch.matmul(asset_returns, t_weights)
        
        if self.is_cuda:
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
        else:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            
        port_returns_np = port_returns_t.detach().cpu().numpy()
        return port_returns_np, float(elapsed_ms)

    def compute_risk(
        self,
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray,
        weights: np.ndarray,
        n_paths: int = 1_000_000,
        horizon_days: int = 10,
        seed: Optional[int] = 42,
    ) -> Dict[str, float]:
        """Simulate paths and return full VaR and CVaR risk report with timing."""
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
        metrics["device"] = str(self.device)
        return metrics
