"""Unified Multi-Asset Monte Carlo Risk Engine.

Orchestrates CUDA GPU, CPU Vectorized, and CPU Unvectorized backends.
Computes 95% and 99% VaR and CVaR (Expected Shortfall) across 100,000 to 1,000,000+
portfolio paths with comprehensive benchmark profiling.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import torch

from quant_engine.risk_cuda.cuda_backend import CudaMonteCarloEngine
from quant_engine.risk_cuda.cpu_backend import (
    CpuVectorizedMonteCarloEngine,
    CpuLoopMonteCarloEngine,
)


class MonteCarloRiskEngine:
    """Unified Monte Carlo risk simulation manager."""
    
    def __init__(self, preferred_device: Optional[str] = None):
        self.cuda_engine = CudaMonteCarloEngine(device=preferred_device)
        self.cpu_vectorized_engine = CpuVectorizedMonteCarloEngine()
        self.cpu_loop_engine = CpuLoopMonteCarloEngine()
        
    def compute_risk(
        self,
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray,
        weights: np.ndarray,
        n_paths: int = 1_000_000,
        horizon_days: int = 10,
        backend: str = "cuda",
        seed: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """Compute portfolio VaR and CVaR using specified backend."""
        if backend == "cuda":
            return self.cuda_engine.compute_risk(
                mean_returns=mean_returns,
                cov_matrix=cov_matrix,
                weights=weights,
                n_paths=n_paths,
                horizon_days=horizon_days,
                seed=seed,
            )
        elif backend == "cpu_vectorized":
            return self.cpu_vectorized_engine.compute_risk(
                mean_returns=mean_returns,
                cov_matrix=cov_matrix,
                weights=weights,
                n_paths=n_paths,
                horizon_days=horizon_days,
                seed=seed,
            )
        elif backend == "cpu_loop":
            return self.cpu_loop_engine.compute_risk(
                mean_returns=mean_returns,
                cov_matrix=cov_matrix,
                weights=weights,
                n_paths=min(n_paths, 20_000),  # Limit unvectorized loop to 20k to avoid freezing
                horizon_days=horizon_days,
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def benchmark_speedup(
        self,
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray,
        weights: np.ndarray,
        n_paths: int = 1_000_000,
        horizon_days: int = 10,
    ) -> Dict[str, Any]:
        """Profile and compare execution latency across CUDA GPU, CPU Vectorized, and Loop backends."""
        # 1. Warmup CUDA
        _ = self.cuda_engine.compute_risk(
            mean_returns, cov_matrix, weights, n_paths=10_000, horizon_days=horizon_days
        )
        
        # 2. CUDA Run
        cuda_res = self.cuda_engine.compute_risk(
            mean_returns, cov_matrix, weights, n_paths=n_paths, horizon_days=horizon_days
        )
        
        # 3. CPU Vectorized Run
        cpu_vec_res = self.cpu_vectorized_engine.compute_risk(
            mean_returns, cov_matrix, weights, n_paths=n_paths, horizon_days=horizon_days
        )
        
        # 4. CPU Unvectorized Loop Run (scaled from 10k paths)
        loop_sample_paths = 5_000
        cpu_loop_res = self.cpu_loop_engine.compute_risk(
            mean_returns, cov_matrix, weights, n_paths=loop_sample_paths, horizon_days=horizon_days
        )
        extrapolated_loop_ms = cpu_loop_res["elapsed_ms"] * (n_paths / loop_sample_paths)
        
        cuda_ms = cuda_res["elapsed_ms"]
        cpu_vec_ms = cpu_vec_res["elapsed_ms"]
        
        speedup_vs_loop = extrapolated_loop_ms / max(cuda_ms, 1e-4)
        speedup_vs_cpu_vec = cpu_vec_ms / max(cuda_ms, 1e-4)
        
        return {
            "n_paths": n_paths,
            "horizon_days": horizon_days,
            "cuda_device": str(self.cuda_engine.device),
            "cuda_latency_ms": round(cuda_ms, 2),
            "cuda_throughput_paths_sec": round(cuda_res["throughput_paths_sec"], 0),
            "cpu_vectorized_latency_ms": round(cpu_vec_ms, 2),
            "cpu_loop_extrapolated_latency_ms": round(extrapolated_loop_ms, 2),
            "speedup_vs_cpu_vectorized": round(speedup_vs_cpu_vec, 2),
            "speedup_vs_cpu_unvectorized_loop": round(speedup_vs_loop, 1),
            "VaR_95": cuda_res["VaR_95"],
            "CVaR_95": cuda_res["CVaR_95"],
            "VaR_99": cuda_res["VaR_99"],
            "CVaR_99": cuda_res["CVaR_99"],
        }
