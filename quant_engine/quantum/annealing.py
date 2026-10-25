"""Quantum-Inspired Simulated Annealing Solver for QUBO Portfolio Optimization.

Implements thermal Metropolis-Hastings annealing over binary spin configurations
with single and pair bit-flip neighborhood mutations to find the minimum-energy
discrete portfolio state.
"""

import time
from typing import Dict, List, Optional, Union
import numpy as np

from quant_engine.quantum.qubo import PortfolioQUBO


class SimulatedAnnealingPortfolioSolver:
    """Quantum-inspired simulated annealing solver for QUBO optimization."""
    
    def __init__(
        self,
        qubo: PortfolioQUBO,
        initial_temp: float = 10.0,
        final_temp: float = 1e-4,
        cooling_rate: float = 0.98,
        steps_per_temp: int = 50,
        seed: int = 42,
    ):
        self.qubo = qubo
        self.N = qubo.N
        self.K = qubo.K
        self.t_init = initial_temp
        self.t_final = final_temp
        self.cooling_rate = cooling_rate
        self.steps_per_temp = steps_per_temp
        self.seed = seed
        
    def solve(self) -> Dict[str, Union[float, int, List[str], str]]:
        """Run simulated annealing on QUBO objective."""
        start_time = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        
        # Initialize with random state satisfying cardinality constraint K
        state = np.zeros(self.N, dtype=int)
        init_indices = rng.choice(self.N, size=self.K, replace=False)
        state[init_indices] = 1
        
        current_energy = self.qubo.evaluate_bitstring_energy(state)
        best_state = state.copy()
        best_energy = current_energy
        
        T = self.t_init
        total_steps = 0
        
        while T > self.t_final:
            for _ in range(self.steps_per_temp):
                total_steps += 1
                neighbor = state.copy()
                
                # Propose either a swap mutation (preserves cardinality) or a single bit-flip
                if rng.uniform() < 0.70:
                    # Pair swap: pick one active and one inactive
                    ones = np.where(neighbor == 1)[0]
                    zeros = np.where(neighbor == 0)[0]
                    if len(ones) > 0 and len(zeros) > 0:
                        i = rng.choice(ones)
                        j = rng.choice(zeros)
                        neighbor[i] = 0
                        neighbor[j] = 1
                else:
                    # Single bit flip
                    k = rng.integers(0, self.N)
                    neighbor[k] = 1 - neighbor[k]
                    
                candidate_energy = self.qubo.evaluate_bitstring_energy(neighbor)
                delta_e = candidate_energy - current_energy
                
                # Metropolis criterion
                if delta_e < 0.0 or rng.uniform() < np.exp(-delta_e / max(T, 1e-8)):
                    state = neighbor
                    current_energy = candidate_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_state = state.copy()
                        
            T *= self.cooling_rate
            
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        selected_assets = [self.qubo.asset_names[i] for i in range(self.N) if best_state[i] == 1]
        
        return {
            "solver": "SimulatedAnnealing",
            "solve_latency_ms": round(elapsed_ms, 2),
            "best_energy": round(best_energy, 5),
            "bitstring": "".join(map(str, best_state)),
            "selected_assets": selected_assets,
            "cardinality_achieved": int(np.sum(best_state)),
            "total_steps": total_steps,
        }
