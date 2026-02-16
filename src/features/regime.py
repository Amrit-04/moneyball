"""Market regime detection using Hidden Markov Models.

This module identifies Bull/Bear/Sideways market regimes and creates
regime-aware feature interactions.
"""

import logging
from typing import Optional, Tuple
import pandas as pd
import numpy as np
from hmmlearn import hmm

logger = logging.getLogger(__name__)


class RegimeDetector:
    """Hidden Markov Model for market regime detection."""
    
    def __init__(self, n_states: int = 3):
        """Initialize HMM with Bull/Bear/Sideways states.
        
        Args:
            n_states: Number of hidden states (default 3 for Bull/Bear/Sideways)
        """
        self.n_states = n_states
        self.hmm = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        self.is_fitted = False
        
        # State labels (assigned after fitting based on mean returns)
        self.state_labels = {}
        
    def fit(self, btc_returns: pd.Series, vix: pd.Series) -> None:
        """Train HMM on weekly BTC returns and VIX.
        
        Args:
            btc_returns: Weekly Bitcoin returns
            vix: Weekly VIX values
        """
        logger.info(f"Training HMM with {self.n_states} states")
        
        # Prepare features
        features = np.column_stack([btc_returns.values, vix.values])
        
        # Remove NaN values
        mask = ~np.isnan(features).any(axis=1)
        features = features[mask]
        
        if len(features) < 52:
            logger.warning(f"Insufficient data for HMM training: {len(features)} weeks")
            logger.warning("Minimum 52 weeks recommended")
        
        # Fit HMM
        self.hmm.fit(features)
        self.is_fitted = True
        
        # Assign state labels based on mean returns
        # State with highest mean return = Bull
        # State with lowest mean return = Bear
        # Middle state = Sideways
        state_means = []
        for state in range(self.n_states):
            state_mask = self.hmm.predict(features) == state
            if state_mask.sum() > 0:
                mean_return = features[state_mask, 0].mean()  # BTC return column
                state_means.append((state, mean_return))
        
        # Sort by mean return
        state_means.sort(key=lambda x: x[1])
        
        if len(state_means) == 3:
            self.state_labels = {
                state_means[0][0]: 'Bear',
                state_means[1][0]: 'Sideways',
                state_means[2][0]: 'Bull'
            }
        else:
            # Fallback labels
            self.state_labels = {i: f'State_{i}' for i in range(self.n_states)}
        
        logger.info(f"HMM trained successfully. State labels: {self.state_labels}")
        logger.info(f"Transition matrix:\n{self.hmm.transmat_}")
        
    def predict_regime(self, btc_returns: pd.Series, vix: pd.Series) -> pd.Series:
        """Predict most likely regime state (0=Bear, 1=Sideways, 2=Bull).
        
        Args:
            btc_returns: Bitcoin returns
            vix: VIX values
            
        Returns:
            Series with regime labels indexed by original index
        """
        if not self.is_fitted:
            raise ValueError("HMM not fitted. Call fit() first.")
        
        logger.info("Predicting market regimes")
        
        # Prepare features
        features = np.column_stack([btc_returns.values, vix.values])
        
        # Handle NaN values
        mask = ~np.isnan(features).any(axis=1)
        
        # Predict states
        states = np.full(len(features), -1)
        if mask.sum() > 0:
            states[mask] = self.hmm.predict(features[mask])
        
        # Convert to labeled regimes
        regime_labels = pd.Series(states, index=btc_returns.index)
        regime_labels = regime_labels.map(lambda x: self.state_labels.get(x, 'Unknown') if x >= 0 else 'Unknown')
        
        logger.info(f"Predicted regimes: {regime_labels.value_counts().to_dict()}")
        return regime_labels
        
    def get_transition_probs(self) -> np.ndarray:
        """Return regime transition probability matrix.
        
        Returns:
            Transition probability matrix
        """
        if not self.is_fitted:
            raise ValueError("HMM not fitted. Call fit() first.")
        
        return self.hmm.transmat_
    
    def get_state_probabilities(self, btc_returns: pd.Series, vix: pd.Series) -> pd.DataFrame:
        """Get probability of each state at each time point.
        
        Args:
            btc_returns: Bitcoin returns
            vix: VIX values
            
        Returns:
            DataFrame with state probabilities
        """
        if not self.is_fitted:
            raise ValueError("HMM not fitted. Call fit() first.")
        
        features = np.column_stack([btc_returns.values, vix.values])
        mask = ~np.isnan(features).any(axis=1)
        
        # Get posterior probabilities
        probs = np.zeros((len(features), self.n_states))
        if mask.sum() > 0:
            probs[mask] = self.hmm.predict_proba(features[mask])
        
        # Create DataFrame with labeled columns
        prob_df = pd.DataFrame(
            probs,
            index=btc_returns.index,
            columns=[self.state_labels.get(i, f'State_{i}') for i in range(self.n_states)]
        )
        
        return prob_df


def create_regime_interactions(
    features: pd.DataFrame,
    regime: pd.Series
) -> pd.DataFrame:
    """Create interaction terms between regime and all features.
    
    For each feature F and regime R, creates F * I(R=Bull), F * I(R=Bear), F * I(R=Sideways).
    
    Args:
        features: DataFrame with feature columns
        regime: Series with regime labels ('Bull', 'Bear', 'Sideways')
        
    Returns:
        DataFrame with original features plus regime interactions
    """
    logger.info("Creating regime interaction features")
    
    df = features.copy()
    
    # Add regime column if not present
    if 'regime' not in df.columns:
        df = df.join(regime.rename('regime'), how='left')
    
    # Create regime indicator variables
    df['regime_bull'] = (df['regime'] == 'Bull').astype(int)
    df['regime_bear'] = (df['regime'] == 'Bear').astype(int)
    df['regime_sideways'] = (df['regime'] == 'Sideways').astype(int)
    
    # Get feature columns (exclude identifiers and regime columns)
    exclude_cols = ['timestamp', 'asset', 'ticker', 'regime', 'regime_bull', 'regime_bear', 'regime_sideways']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Create interactions
    interaction_count = 0
    for col in feature_cols:
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            # Create interaction with each regime
            df[f'{col}_x_bull'] = df[col] * df['regime_bull']
            df[f'{col}_x_bear'] = df[col] * df['regime_bear']
            df[f'{col}_x_sideways'] = df[col] * df['regime_sideways']
            interaction_count += 3
    
    logger.info(f"Created {interaction_count} regime interaction features")
    return df


def prepare_regime_data(
    btc_prices: pd.DataFrame,
    vix_data: Optional[pd.DataFrame] = None,
    resample_freq: str = 'W'
) -> Tuple[pd.Series, pd.Series]:
    """Prepare BTC returns and VIX data for regime detection.
    
    Args:
        btc_prices: DataFrame with BTC prices [timestamp, close]
        vix_data: DataFrame with VIX values [timestamp, close] (optional)
        resample_freq: Resampling frequency (default 'W' for weekly)
        
    Returns:
        Tuple of (btc_returns, vix_series)
    """
    logger.info(f"Preparing regime data (resampling to {resample_freq})")
    
    # Resample BTC prices to weekly
    btc_weekly = btc_prices.set_index('timestamp')['close'].resample(resample_freq).last()
    btc_returns = btc_weekly.pct_change().dropna()
    
    # Handle VIX data
    if vix_data is not None and not vix_data.empty:
        vix_weekly = vix_data.set_index('timestamp')['close'].resample(resample_freq).last()
        vix_series = vix_weekly.reindex(btc_returns.index)
    else:
        # If VIX not available, use BTC volatility as proxy
        logger.warning("VIX data not available, using BTC volatility as proxy")
        vix_series = btc_returns.rolling(window=4).std() * np.sqrt(52) * 100  # Annualized vol
    
    # Align indices
    common_index = btc_returns.index.intersection(vix_series.index)
    btc_returns = btc_returns.loc[common_index]
    vix_series = vix_series.loc[common_index]
    
    logger.info(f"Prepared {len(btc_returns)} periods for regime detection")
    return btc_returns, vix_series


def main():
    """Main function for testing regime detection."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create sample BTC price data
    dates = pd.date_range('2022-01-01', '2024-12-31', freq='D')
    
    np.random.seed(42)
    btc_prices = pd.DataFrame({
        'timestamp': dates,
        'close': 100 * np.exp(np.random.randn(len(dates)).cumsum() * 0.02)
    })
    
    # Prepare data
    btc_returns, vix_proxy = prepare_regime_data(btc_prices)
    
    # Train regime detector
    detector = RegimeDetector(n_states=3)
    detector.fit(btc_returns, vix_proxy)
    
    # Predict regimes
    regimes = detector.predict_regime(btc_returns, vix_proxy)
    
    logger.info(f"\nRegime distribution:\n{regimes.value_counts()}")
    logger.info(f"\nTransition probabilities:\n{detector.get_transition_probs()}")
    
    # Get state probabilities
    probs = detector.get_state_probabilities(btc_returns, vix_proxy)
    logger.info(f"\nState probabilities (first 10):\n{probs.head(10)}")
    
    # Create sample features for interaction testing
    features = pd.DataFrame({
        'timestamp': btc_returns.index,
        'asset': 'BTC',
        'feature1': np.random.randn(len(btc_returns)),
        'feature2': np.random.randn(len(btc_returns))
    })
    
    # Create interactions
    features_with_interactions = create_regime_interactions(features, regimes)
    logger.info(f"\nFeatures with interactions: {features_with_interactions.columns.tolist()}")


if __name__ == "__main__":
    main()
