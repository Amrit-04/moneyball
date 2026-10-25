"""Parameterized Quantum Circuit (QAOA) Statevector Simulator.

Simulates Quantum Approximate Optimization Algorithm (QAOA) statevector dynamics
over N qubits (up to 16 qubits) across circuit depths p in {1, 2, 3}:
1. Superposition initialization: |+>^{otimes N}
2. Cost Hamiltonian unitary phase application: U(C, gamma) = exp(-i gamma H_C)
3. Transverse-field mixer rotation: U(B, beta) = prod_j exp(-i beta X_j)
4. Expectation evaluation: <H_C> = <psi|H_C|psi>
5. Classical parameter optimization via COBYLA
"""

import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import minimize

from quant_engine.quantum.qubo import PortfolioQUBO


class QAOAPortfolioSimulator:
    """Statevector QAOA simulator for cardinality-constrained portfolio optimization."""
    
    def __init__(
        self,
        qubo: PortfolioQUBO,
        depth_p: int = 2,
    ):
        """Initialize QAOA statevector simulator.
        
        Args:
            qubo: Instance of PortfolioQUBO.
            depth_p: QAOA circuit depth / number of layers (1, 2, or 3).
        """
        self.qubo = qubo
        self.N = qubo.N
        self.p = depth_p
        self.dim = 1 << self.N
        
        # Precompute diagonal energies E(z) across all 2^N basis states
        self.energies = qubo.get_all_state_energies()
        self.min_energy = float(np.min(self.energies))
        self.max_energy = float(np.max(self.energies))
        self.best_state_idx = int(np.argmin(self.energies))
        
        # Binary representation of optimal ground state
        self.optimal_bitstring = np.array(
            [(self.best_state_idx >> (self.N - 1 - b)) & 1 for b in range(self.N)],
            dtype=int
        )
        
    def _apply_mixer_qubit(
        self,
        psi: np.ndarray,
        qubit: int,
        beta: float
    ) -> np.ndarray:
        """Apply single-qubit X-rotation R_X(2*beta) = cos(beta)*I - i*sin(beta)*X on qubit axis."""
        cos_b = np.cos(beta)
        sin_b = -1j * np.sin(beta)
        
        # Reshape to (2^qubit, 2, 2^(N - 1 - qubit))
        stride_left = 1 << qubit
        stride_right = 1 << (self.N - 1 - qubit)
        
        psi_reshaped = psi.reshape((stride_left, 2, stride_right))
        
        # Matrix multiply along middle dimension:
        # out[0] = cos_b * in[0] + sin_b * in[1]
        # out[1] = sin_b * in[0] + cos_b * in[1]
        in_0 = psi_reshaped[:, 0, :]
        in_1 = psi_reshaped[:, 1, :]
        
        out_0 = cos_b * in_0 + sin_b * in_1
        out_1 = sin_b * in_0 + cos_b * in_1
        
        out = np.stack([out_0, out_1], axis=1)
        return out.reshape(-1)

    def simulate_circuit(self, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Execute statevector circuit given parameter vectors gamma and beta.
        
        Args:
            gamma: Array of length p containing cost layer angles.
            beta: Array of length p containing mixer layer angles.
            
        Returns:
            Normalized complex statevector psi of dimension 2^N.
        """
        # 1. Initialize uniform superposition |+>^{otimes N} = 1 / sqrt(2^N)
        psi = np.full(self.dim, 1.0 / np.sqrt(self.dim), dtype=np.complex128)
        
        # 2. Iterate over p layers
        for layer in range(self.p):
            # Cost Unitary: exp(-i * gamma_l * E_z)
            phase = np.exp(-1j * gamma[layer] * self.energies)
            psi = psi * phase
            
            # Mixer Unitary: prod_j exp(-i * beta_l * X_j)
            for j in range(self.N):
                psi = self._apply_mixer_qubit(psi, j, beta[layer])
                
        return psi

    def expectation_value(self, params: np.ndarray) -> float:
        """Compute expected cost Hamiltonian energy <psi(gamma, beta)| H_C |psi(gamma, beta)>."""
        gamma = params[:self.p]
        beta = params[self.p:]
        
        psi = self.simulate_circuit(gamma, beta)
        probs = np.abs(psi) ** 2
        exp_energy = float(np.sum(probs * self.energies))
        return exp_energy

    def optimize(
        self,
        max_iter: int = 100,
        initial_params: Optional[np.ndarray] = None,
    ) -> Dict[str, Union[float, np.ndarray, int, str]]:
        """Optimize QAOA parameters (gamma, beta) using classical COBYLA optimizer."""
        start_time = time.perf_counter()
        
        if initial_params is None:
            # Heuristic standard initialization: gamma in [0, pi], beta in [0, pi/2]
            rng = np.random.default_rng(42)
            init_gamma = rng.uniform(0.1, 0.8, size=self.p)
            init_beta = rng.uniform(0.2, 0.7, size=self.p)
            initial_params = np.concatenate([init_gamma, init_beta])
            
        res = minimize(
            self.expectation_value,
            initial_params,
            method="COBYLA",
            options={"maxiter": max_iter, "rhobeg": 0.5, "tol": 1e-4}
        )
        
        elapsed_sec = time.perf_counter() - start_time
        opt_params = res.x
        opt_gamma = opt_params[:self.p]
        opt_beta = opt_params[self.p:]
        
        final_psi = self.simulate_circuit(opt_gamma, opt_beta)
        probs = np.abs(final_psi) ** 2
        expected_energy = float(np.sum(probs * self.energies))
        
        # Ground state probability
        p_ground_state = float(probs[self.best_state_idx])
        
        # Most likely bitstring
        sampled_best_idx = int(np.argmax(probs))
        sampled_bitstring = np.array(
            [(sampled_best_idx >> (self.N - 1 - b)) & 1 for b in range(self.N)],
            dtype=int
        )
        sampled_energy = float(self.energies[sampled_best_idx])
        
        # Exact Approximation Ratio: (E_max - <H>) / (E_max - E_min)
        energy_range = max(self.max_energy - self.min_energy, 1e-8)
        approx_ratio = float((self.max_energy - expected_energy) / energy_range)
        sampled_approx_ratio = float((self.max_energy - sampled_energy) / energy_range)
        
        # Selected asset names
        selected_assets = [
            self.qubo.asset_names[i] for i in range(self.N) if sampled_bitstring[i] == 1
        ]
        
        return {
            "qubits_N": self.N,
            "depth_p": self.p,
            "solve_latency_ms": round(elapsed_sec * 1000.0, 2),
            "expected_energy": round(expected_energy, 5),
            "optimal_ground_energy": round(self.min_energy, 5),
            "worst_energy": round(self.max_energy, 5),
            "approximation_ratio": round(approx_ratio, 4),
            "sampled_approximation_ratio": round(sampled_approx_ratio, 4),
            "ground_state_probability": round(p_ground_state, 4),
            "optimal_bitstring": "".join(map(str, self.optimal_bitstring)),
            "sampled_bitstring": "".join(map(str, sampled_bitstring)),
            "selected_assets": selected_assets,
            "cardinality_target": self.qubo.K,
            "cardinality_achieved": int(np.sum(sampled_bitstring)),
            "iterations": int(res.nfev),
        }
