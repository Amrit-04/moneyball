"""Quadratic Unconstrained Binary Optimization (QUBO) & Ising Hamiltonian Mapping.

Formulates discrete cardinality-constrained portfolio optimization:
  min_{x in {0,1}^N} -mu^T x + (gamma / 2) x^T Sigma x + lambda_penalty (sum_i x_i - K)^2

Maps binary variables x_i in {0, 1} to Ising spin variables s_i in {+1, -1} via:
  x_i = (I - Z_i) / 2
yielding the Ising Spin Hamiltonian:
  H_C = sum_i h_i Z_i + sum_{i < j} J_{ij} Z_i Z_j + C_offset I
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


class PortfolioQUBO:
    """Constructs Markowitz cardinality-constrained QUBO and maps to Ising Hamiltonian."""
    
    def __init__(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        k_assets: int,
        risk_aversion: float = 1.0,
        penalty_factor: Optional[float] = None,
        asset_names: Optional[List[str]] = None,
    ):
        """Initialize Portfolio QUBO.
        
        Args:
            expected_returns: Vector of expected asset returns (N,).
            cov_matrix: Covariance matrix of asset returns (N, N).
            k_assets: Cardinality constraint (exact number of assets to select, K).
            risk_aversion: Risk tolerance parameter gamma (default 1.0).
            penalty_factor: Multiplier for (sum x_i - K)^2 penalty.
            asset_names: Optional list of asset ticker strings.
        """
        self.mu = np.asarray(expected_returns, dtype=np.float64)
        self.cov = np.asarray(cov_matrix, dtype=np.float64)
        self.N = len(self.mu)
        self.K = k_assets
        self.gamma = risk_aversion
        self.asset_names = asset_names or [f"Asset_{i}" for i in range(self.N)]
        
        # Auto-scale penalty factor if not provided:
        # Penalty must exceed the maximum marginal return/risk trade-off
        if penalty_factor is None:
            max_risk = np.max(np.abs(self.cov)) * self.gamma
            max_ret = np.max(np.abs(self.mu))
            self.penalty = max(1.0, 5.0 * (max_ret + max_risk))
        else:
            self.penalty = penalty_factor
            
        self.Q, self.const_offset = self._build_qubo_matrix()
        self.h, self.J, self.ising_offset = self._qubo_to_ising()
        
    def _build_qubo_matrix(self) -> Tuple[np.ndarray, float]:
        """Build upper-triangular / symmetric QUBO matrix Q such that f(x) = x^T Q x + const."""
        N = self.N
        K = self.K
        lam = self.penalty
        
        # Base Markowitz objective: -mu^T x + 0.5 * gamma * x^T Sigma x
        # Penalty term: lambda * (sum x_i - K)^2 = lambda * [ sum x_i + 2 sum_{i<j} x_i x_j - 2K sum x_i + K^2 ]
        #               = lambda * (1 - 2K) sum x_i + 2 lambda sum_{i<j} x_i x_j + lambda * K^2
        
        Q = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            Q[i, i] = -self.mu[i] + 0.5 * self.gamma * self.cov[i, i] + lam * (1.0 - 2.0 * K)
            for j in range(i + 1, N):
                cov_ij = 0.5 * self.gamma * (self.cov[i, j] + self.cov[j, i])
                term_ij = cov_ij + 2.0 * lam
                Q[i, j] = term_ij
                Q[j, i] = 0.0  # Keep upper-triangular format
                
        const_offset = lam * (K ** 2)
        return Q, float(const_offset)

    def _qubo_to_ising(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Map binary QUBO to Ising Spin Hamiltonian: H = sum h_i Z_i + sum_{i<j} J_ij Z_i Z_j + offset.
        
        Using transformation: x_i = (1 - Z_i) / 2
        """
        N = self.N
        h = np.zeros(N, dtype=np.float64)
        J = np.zeros((N, N), dtype=np.float64)
        
        # Symmetrize off-diagonal terms: W_ij = Q_ij for i < j
        W = self.Q + self.Q.T
        np.fill_diagonal(W, 0.0)
        
        # J_ij = 1/4 * W_ij for i < j
        for i in range(N):
            for j in range(i + 1, N):
                J[i, j] = 0.25 * W[i, j]
                J[j, i] = J[i, j]
                
        # h_i = -1/2 * Q_ii - 1/4 * sum_{j != i} W_ij
        for i in range(N):
            h[i] = -0.5 * self.Q[i, i] - 0.25 * np.sum(W[i, :])
            
        # Offset = const_offset + 1/2 * sum Q_ii + 1/8 * sum_{i,j != i} W_ij
        ising_offset = self.const_offset + 0.5 * np.trace(self.Q) + 0.125 * np.sum(W)
        return h, J, float(ising_offset)

    def evaluate_bitstring_energy(self, bitstring: np.ndarray) -> float:
        """Compute exact classical QUBO objective energy for binary vector x in {0, 1}^N."""
        x = np.asarray(bitstring, dtype=np.float64)
        energy = float(x.T @ self.Q @ x + self.const_offset)
        return energy

    def evaluate_spin_energy(self, spins: np.ndarray) -> float:
        """Compute exact Ising Hamiltonian energy for spin vector s in {-1, +1}^N."""
        s = np.asarray(spins, dtype=np.float64)
        e_h = np.sum(self.h * s)
        # sum_{i<j} J_ij s_i s_j
        e_J = 0.5 * float(s.T @ self.J @ s)
        return float(e_h + e_J + self.ising_offset)

    def get_all_state_energies(self) -> np.ndarray:
        """Compute classical energies for all 2^N computational basis states."""
        dim = 1 << self.N
        energies = np.zeros(dim, dtype=np.float64)
        
        # Generate binary matrix of all bitstrings (dim, N)
        for idx in range(dim):
            # Extract bits
            bits = np.array([(idx >> (self.N - 1 - b)) & 1 for b in range(self.N)], dtype=np.float64)
            energies[idx] = self.evaluate_bitstring_energy(bits)
            
        return energies
