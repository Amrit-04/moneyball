"""Quantum Portfolio Optimization Benchmark Suite.

Compares:
1. Parameterized QAOA Statevector Circuit (depths p=1, 2, 3)
2. Quantum-Inspired Simulated Annealing
3. Classical Markowitz Mean-Variance (Continuous SLSQP + Cardinality Selection)
4. Equal-Weight (1/N) Baseline
5. Exact Classical Brute-Force Ground State (all 2^N states)
"""

import time
from typing import Any, Dict, List, Optional
import numpy as np
from scipy.optimize import minimize

from quant_engine.quantum.qubo import PortfolioQUBO
from quant_engine.quantum.qaoa_simulator import QAOAPortfolioSimulator
from quant_engine.quantum.annealing import SimulatedAnnealingPortfolioSolver


def run_classical_markowitz(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    k_assets: int,
    risk_aversion: float = 1.0,
    asset_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Continuous Markowitz Mean-Variance optimization solved via SLSQP, selecting top K assets."""
    start_time = time.perf_counter()
    N = len(expected_returns)
    names = asset_names or [f"Asset_{i}" for i in range(N)]
    
    # Continuous objective: min -mu^T w + 0.5 * gamma * w^T Sigma w
    def obj(w: np.ndarray) -> float:
        return -float(np.dot(expected_returns, w)) + 0.5 * risk_aversion * float(w.T @ cov_matrix @ w)
        
    w0 = np.full(N, 1.0 / N)
    bounds = [(0.0, 1.0) for _ in range(N)]
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    
    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    
    continuous_weights = res.x
    # Cardinality projection: select top K largest weights
    top_k_indices = np.argsort(continuous_weights)[-k_assets:]
    discrete_weights = np.zeros(N)
    discrete_weights[top_k_indices] = 1.0 / k_assets
    
    # Evaluate discrete portfolio metrics
    port_ret = float(np.dot(expected_returns, discrete_weights))
    port_vol = float(np.sqrt(discrete_weights.T @ cov_matrix @ discrete_weights))
    sharpe = float(port_ret / max(port_vol, 1e-6))
    
    selected_assets = [names[i] for i in sorted(top_k_indices)]
    binary_bitstring = np.zeros(N, dtype=int)
    binary_bitstring[top_k_indices] = 1
    
    return {
        "method": "Classical_Markowitz_SLSQP",
        "solve_latency_ms": round(elapsed_ms, 2),
        "selected_assets": selected_assets,
        "bitstring": "".join(map(str, binary_bitstring)),
        "expected_return": round(port_ret, 4),
        "expected_volatility": round(port_vol, 4),
        "sharpe_ratio": round(sharpe, 3),
    }


def run_equal_weight_baseline(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    asset_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Equal-Weight 1/N baseline across all candidates."""
    N = len(expected_returns)
    w = np.full(N, 1.0 / N)
    port_ret = float(np.dot(expected_returns, w))
    port_vol = float(np.sqrt(w.T @ cov_matrix @ w))
    sharpe = float(port_ret / max(port_vol, 1e-6))
    
    return {
        "method": "Equal_Weight_1_over_N",
        "solve_latency_ms": 0.01,
        "selected_assets": asset_names or [f"Asset_{i}" for i in range(N)],
        "expected_return": round(port_ret, 4),
        "expected_volatility": round(port_vol, 4),
        "sharpe_ratio": round(sharpe, 3),
    }


def run_quantum_portfolio_benchmark(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    k_assets: int = 4,
    risk_aversion: float = 1.0,
    depths: List[int] = [1, 2, 3],
    asset_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Execute end-to-end quantum benchmark comparing QAOA (p=1,2,3), Annealing, Markowitz, and Exact."""
    qubo = PortfolioQUBO(
        expected_returns=expected_returns,
        cov_matrix=cov_matrix,
        k_assets=k_assets,
        risk_aversion=risk_aversion,
        asset_names=asset_names,
    )
    
    # 1. Classical Brute Force Exact Ground State
    start_exact = time.perf_counter()
    all_energies = qubo.get_all_state_energies()
    min_idx = int(np.argmin(all_energies))
    max_idx = int(np.argmax(all_energies))
    exact_ms = (time.perf_counter() - start_exact) * 1000.0
    
    exact_ground_energy = float(all_energies[min_idx])
    worst_energy = float(all_energies[max_idx])
    exact_bitstring = "".join([(min_idx >> (qubo.N - 1 - b)) & 1 for b in range(qubo.N)].__iter__().__str__())
    exact_bits = np.array([(min_idx >> (qubo.N - 1 - b)) & 1 for b in range(qubo.N)], dtype=int)
    exact_selected = [qubo.asset_names[i] for i in range(qubo.N) if exact_bits[i] == 1]
    
    exact_results = {
        "method": "Exact_Ground_State_BruteForce",
        "solve_latency_ms": round(exact_ms, 2),
        "ground_state_energy": round(exact_ground_energy, 5),
        "worst_energy": round(worst_energy, 5),
        "bitstring": "".join(map(str, exact_bits)),
        "selected_assets": exact_selected,
        "approximation_ratio": 1.0,
    }
    
    # 2. QAOA across depths p in depths
    qaoa_results = {}
    for p in depths:
        sim = QAOAPortfolioSimulator(qubo, depth_p=p)
        res_p = sim.optimize(max_iter=60)
        qaoa_results[f"p_{p}"] = res_p
        
    # 3. Quantum-Inspired Simulated Annealing
    annealer = SimulatedAnnealingPortfolioSolver(qubo)
    sa_res = annealer.solve()
    
    # 4. Classical Markowitz Continuous SLSQP
    markowitz_res = run_classical_markowitz(
        expected_returns, cov_matrix, k_assets, risk_aversion, asset_names
    )
    
    # 5. Equal Weight Baseline
    eq_res = run_equal_weight_baseline(expected_returns, cov_matrix, asset_names)
    
    return {
        "N_qubits": qubo.N,
        "K_cardinality": k_assets,
        "exact_ground_state": exact_results,
        "qaoa_circuits": qaoa_results,
        "simulated_annealing": sa_res,
        "classical_markowitz": markowitz_res,
        "equal_weight": eq_res,
    }
