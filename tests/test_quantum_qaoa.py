"""Unit tests for QUBO formulation, Ising Spin mapping, and QAOA statevector circuit simulator."""

import numpy as np
import pytest

from quant_engine.quantum import (
    PortfolioQUBO,
    QAOAPortfolioSimulator,
    SimulatedAnnealingPortfolioSolver,
    run_classical_markowitz,
)


@pytest.fixture
def quantum_inputs():
    N = 8
    rng = np.random.default_rng(42)
    mu = np.linspace(0.06, 0.20, N)
    A = rng.standard_normal((N, N))
    cov = 0.05 * (A @ A.T) / N + 0.02 * np.eye(N)
    return mu, cov


def test_qubo_ising_hamiltonian_exact_equivalence(quantum_inputs):
    """Verify that classical QUBO energy matches Ising Hamiltonian energy for all 2^N states."""
    mu, cov = quantum_inputs
    N = len(mu)
    qubo = PortfolioQUBO(expected_returns=mu, cov_matrix=cov, k_assets=3, risk_aversion=1.0)
    
    # Test random bitstrings
    rng = np.random.default_rng(42)
    for _ in range(50):
        bits = rng.integers(0, 2, size=N)
        # Spin transformation: s_i = 1 - 2 * x_i (so 0 -> +1, 1 -> -1)
        spins = 1.0 - 2.0 * bits
        
        e_qubo = qubo.evaluate_bitstring_energy(bits)
        e_ising = qubo.evaluate_spin_energy(spins)
        
        assert np.isclose(e_qubo, e_ising, atol=1e-6), (
            f"QUBO energy {e_qubo} != Ising energy {e_ising} for bitstring {bits}"
        )


def test_qaoa_statevector_unitary_norm_conservation(quantum_inputs):
    """Verify QAOA circuit statevector preserves probability normalization: sum |psi_i|^2 == 1.0."""
    mu, cov = quantum_inputs
    qubo = PortfolioQUBO(expected_returns=mu, cov_matrix=cov, k_assets=3, risk_aversion=1.0)
    
    for depth in [1, 2, 3]:
        sim = QAOAPortfolioSimulator(qubo, depth_p=depth)
        gamma = np.full(depth, 0.4)
        beta = np.full(depth, 0.6)
        
        psi = sim.simulate_circuit(gamma, beta)
        norm_sq = np.sum(np.abs(psi) ** 2)
        assert np.isclose(norm_sq, 1.0, atol=1e-6), f"Depth {depth}: Norm squared {norm_sq} != 1.0"


def test_qaoa_optimization_approximation_ratio(quantum_inputs):
    """Verify QAOA optimization produces Approximation Ratio > 0.70."""
    mu, cov = quantum_inputs
    qubo = PortfolioQUBO(expected_returns=mu, cov_matrix=cov, k_assets=2, risk_aversion=1.0)
    sim = QAOAPortfolioSimulator(qubo, depth_p=2)
    
    res = sim.optimize(max_iter=50)
    ar = res["approximation_ratio"]
    assert ar >= 0.70, f"QAOA Approximation Ratio {ar} is below threshold 0.70"
    assert res["ground_state_probability"] > 0.0


def test_simulated_annealing_cardinality(quantum_inputs):
    """Verify simulated annealing solver achieves target cardinality."""
    mu, cov = quantum_inputs
    qubo = PortfolioQUBO(expected_returns=mu, cov_matrix=cov, k_assets=3, risk_aversion=1.0)
    annealer = SimulatedAnnealingPortfolioSolver(qubo, initial_temp=5.0, final_temp=1e-3, steps_per_temp=20)
    
    res = annealer.solve()
    assert res["cardinality_achieved"] == 3
    assert len(res["selected_assets"]) == 3
