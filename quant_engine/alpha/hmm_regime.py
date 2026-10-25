"""3-State Gaussian Hidden Markov Model for Financial Market Regime Detection.

Classifies market dynamics into three canonical economic states:
- State 0: Low-Vol Bull (positive drift, lowest realized volatility)
- State 1: Transitional / Sideways (neutral drift, medium volatility)
- State 2: High-Vol Bear (negative drift / tail risk, highest realized volatility)

Includes automatic state sorting/relabeling to guarantee deterministic semantic
interpretation across fitting runs, stationary distribution computation, and
filtered/smoothed regime probability estimation.
"""

from typing import Dict, List, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd
from hmmlearn import hmm

logger = logging.getLogger(__name__)

REGIME_NAMES: Dict[int, str] = {
    0: "Low-Vol Bull",
    1: "Transitional / Sideways",
    2: "High-Vol Bear",
}


class MarketRegimeHMM:
    """3-State Gaussian HMM fitted on market log-returns and realized volatility."""
    
    def __init__(
        self,
        n_states: int = 3,
        covariance_type: str = "full",
        n_iter: int = 200,
        random_state: int = 42,
    ):
        """Initialize the HMM regime model.
        
        Args:
            n_states: Number of hidden states (default: 3).
            covariance_type: Covariance specification ('full', 'diag', 'spherical').
            n_iter: Maximum EM iterations.
            random_state: Random seed for initialization.
        """
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
            tol=1e-2,
            init_params="stmc",
            params="stmc",
        )
        self.is_fitted: bool = False
        self.state_map: Dict[int, int] = {}  # internal raw state -> sorted semantic state
        self.reverse_state_map: Dict[int, int] = {}
        
    def _prepare_observations(
        self,
        market_returns: Union[pd.Series, np.ndarray],
        realized_vol: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> np.ndarray:
        """Prepare bivariate observation matrix [log_returns, realized_volatility]."""
        if isinstance(market_returns, pd.Series):
            ret_vals = market_returns.values
        else:
            ret_vals = np.asarray(market_returns)
            
        if realized_vol is None:
            # Compute rolling 20-day realized volatility from returns
            s = pd.Series(ret_vals)
            vol_vals = (s.rolling(window=20, min_periods=10).std() * np.sqrt(252)).bfill().values
        elif isinstance(realized_vol, pd.Series):
            vol_vals = realized_vol.values
        else:
            vol_vals = np.asarray(realized_vol)
            
        X = np.column_stack([ret_vals, vol_vals])
        valid_idx = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)
        if not np.all(valid_idx):
            X = X[valid_idx]
        return X

    def fit(
        self,
        market_returns: Union[pd.Series, np.ndarray],
        realized_vol: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> "MarketRegimeHMM":
        """Fit 3-state HMM on market observations and sort states by Sharpe-like ratio / volatility.
        
        Sorting rule:
          State 0 (Low-Vol Bull): High return, lowest volatility
          State 1 (Transitional): Medium return, medium volatility
          State 2 (High-Vol Bear): Lowest return, highest volatility
        """
        X = self._prepare_observations(market_returns, realized_vol)
        if len(X) < 100:
            raise ValueError(f"Insufficient observations for HMM training: {len(X)} < 100 required.")
            
        self.model.fit(X)
        self.is_fitted = True
        
        # Analyze state means to determine deterministic sorting
        # X column 0: returns, column 1: volatility
        means = self.model.means_
        state_scores = []
        for s in range(self.n_states):
            mu_s = means[s, 0]
            sigma_s = means[s, 1]
            # Higher score = more bullish / lower risk
            # score = mu / (sigma + 1e-4) - sigma
            score = mu_s / (sigma_s + 1e-4)
            state_scores.append((s, score, sigma_s, mu_s))
            
        # Sort states in descending order of bullishness: 0=Bull, 1=Transitional, 2=Bear
        state_scores.sort(key=lambda item: item[1], reverse=True)
        self.state_map = {orig[0]: sorted_idx for sorted_idx, orig in enumerate(state_scores)}
        self.reverse_state_map = {sorted_idx: orig[0] for sorted_idx, orig in enumerate(state_scores)}
        
        logger.info(f"HMM fitted successfully. State mapping (raw -> semantic): {self.state_map}")
        return self

    def predict(
        self,
        market_returns: Union[pd.Series, np.ndarray],
        realized_vol: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> np.ndarray:
        """Predict semantic regime sequence (0=Bull, 1=Transitional, 2=Bear)."""
        if not self.is_fitted:
            raise RuntimeError("HMM model must be fitted before calling predict.")
            
        X = self._prepare_observations(market_returns, realized_vol)
        raw_states = self.model.predict(X)
        semantic_states = np.array([self.state_map[s] for s in raw_states])
        return semantic_states

    def predict_proba(
        self,
        market_returns: Union[pd.Series, np.ndarray],
        realized_vol: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> np.ndarray:
        """Compute posterior smoothed regime probabilities (T x n_states).
        
        Returns:
            np.ndarray of shape (T, 3) where columns correspond to [Bull, Transitional, Bear].
        """
        if not self.is_fitted:
            raise RuntimeError("HMM model must be fitted before calling predict_proba.")
            
        X = self._prepare_observations(market_returns, realized_vol)
        raw_posteriors = self.model.predict_proba(X)
        
        # Reorder columns to align with semantic states 0, 1, 2
        sorted_posteriors = np.zeros_like(raw_posteriors)
        for raw_s, sem_s in self.state_map.items():
            sorted_posteriors[:, sem_s] = raw_posteriors[:, raw_s]
            
        return sorted_posteriors

    @property
    def transition_matrix(self) -> np.ndarray:
        """Return the transition probability matrix aligned to semantic states (3x3)."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
            
        raw_trans = self.model.transmat_
        sorted_trans = np.zeros((self.n_states, self.n_states))
        for raw_i, sem_i in self.state_map.items():
            for raw_j, sem_j in self.state_map.items():
                sorted_trans[sem_i, sem_j] = raw_trans[raw_i, raw_j]
        return sorted_trans

    @property
    def stationary_distribution(self) -> np.ndarray:
        """Compute stationary distribution pi = pi * P, sum(pi) = 1."""
        P = self.transition_matrix
        # Solve (P^T - I) pi = 0 with constraint sum(pi) = 1
        A = P.T - np.eye(self.n_states)
        A[-1, :] = 1.0
        b = np.zeros(self.n_states)
        b[-1] = 1.0
        try:
            pi = np.linalg.solve(A, b)
            pi = np.maximum(pi, 0.0)
            return pi / np.sum(pi)
        except np.linalg.LinAlgError:
            return np.ones(self.n_states) / self.n_states
