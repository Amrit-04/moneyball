"""Cross-Sectional & Time-Series Factor Feature Engineering.

Computes:
1. Multi-horizon momentum (5-day, 20-day, 60-day log returns)
2. Realized volatility (rolling 20-day annualized std)
3. 14-day Relative Strength Index (RSI)
4. Bollinger Z-score (price vs. 20-day moving average normalized by std)
5. Volume shock (5-day vs. 20-day rolling volume ratio)
6. Alternative sentiment & activity proxy
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def compute_rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Compute Relative Strength Index (RSI) across asset price columns.
    
    Args:
        prices: DataFrame of asset prices (index=dates, columns=assets).
        window: Lookback period (default 14).
        
    Returns:
        DataFrame of RSI values bounded in [0, 100].
    """
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_factor_panel(
    panel: Dict[str, pd.DataFrame],
    seed: int = 42
) -> pd.DataFrame:
    """Generate stacked cross-sectional & time-series factor feature panel.
    
    Args:
        panel: Dict containing 'close', 'open', 'high', 'low', 'volume', 'returns'.
        seed: Random seed for deterministic sentiment/activity proxy.
        
    Returns:
        pd.DataFrame with MultiIndex (date, asset) containing all engineered features
        and forward returns for supervised alpha modeling.
    """
    close = panel["close"]
    volume = panel["volume"]
    returns = panel["returns"]
    assets = list(close.columns)
    dates = close.index
    
    log_prices = np.log(close)
    daily_log_ret = log_prices.diff()
    
    # 1. Multi-horizon Momentum
    mom_5d = log_prices.diff(5)
    mom_20d = log_prices.diff(20)
    mom_60d = log_prices.diff(60)
    
    # 2. Realized Volatility (annualized, 20-day rolling std)
    realized_vol_20d = daily_log_ret.rolling(window=20, min_periods=15).std() * np.sqrt(252.0)
    
    # 3. RSI 14-day
    rsi_14d = compute_rsi(close, window=14)
    
    # 4. Bollinger Z-Score (20-day)
    rolling_mean_20 = close.rolling(window=20, min_periods=15).mean()
    rolling_std_20 = close.rolling(window=20, min_periods=15).std()
    bollinger_z_20d = (close - rolling_mean_20) / (rolling_std_20 + 1e-8)
    
    # 5. Volume Shock (5d volume SMA / 20d volume SMA - 1.0)
    vol_sma_5 = volume.rolling(window=5, min_periods=3).mean()
    vol_sma_20 = volume.rolling(window=20, min_periods=10).mean()
    volume_shock = (vol_sma_5 / (vol_sma_20 + 1e-8)) - 1.0
    
    # 6. Alternative Sentiment / Activity Proxy
    # Synthesized from short-term momentum, volume surprise, with mean-reverting AR(1) noise
    rng = np.random.default_rng(seed)
    n_dates, n_assets = len(dates), len(assets)
    ar_noise = np.zeros((n_dates, n_assets))
    eps = rng.standard_normal((n_dates, n_assets)) * 0.15
    for t in range(1, n_dates):
        ar_noise[t] = 0.70 * ar_noise[t - 1] + eps[t]
        
    df_noise = pd.DataFrame(ar_noise, index=dates, columns=assets)
    # Combine standardized 5d momentum and volume shock with noise, clipped to [-3, 3]
    mom_z = (mom_5d - mom_5d.rolling(60).mean()) / (mom_5d.rolling(60).std() + 1e-8)
    vol_z = (volume_shock - volume_shock.rolling(60).mean()) / (volume_shock.rolling(60).std() + 1e-8)
    sentiment_proxy = (0.4 * mom_z + 0.3 * vol_z + df_noise).clip(-3.0, 3.0)
    
    # 7. Forward Target: Next 1-day return and 5-day return
    fwd_ret_1d = returns.shift(-1)
    fwd_ret_5d = (close.shift(-5) / close) - 1.0
    
    # Unpivot to long format
    feature_dict = {
        "close": close,
        "volume": volume,
        "daily_return": returns,
        "mom_5d": mom_5d,
        "mom_20d": mom_20d,
        "mom_60d": mom_60d,
        "realized_vol_20d": realized_vol_20d,
        "rsi_14d": rsi_14d,
        "bollinger_z_20d": bollinger_z_20d,
        "volume_shock": volume_shock,
        "sentiment_proxy": sentiment_proxy,
        "target_fwd_1d": fwd_ret_1d,
        "target_fwd_5d": fwd_ret_5d,
    }
    
    stacked_series = {}
    for feat_name, df_feat in feature_dict.items():
        s = df_feat.stack(future_stack=True) if hasattr(df_feat, "stack") else df_feat.stack()
        stacked_series[feat_name] = s
        
    panel_df = pd.DataFrame(stacked_series)
    panel_df.index.names = ["date", "asset"]
    
    # Drop rows where initial rolling lookbacks are NaN
    # Lookback 60 days needed for mom_60d and rolling standardization
    valid_panel = panel_df.dropna(subset=["mom_60d", "realized_vol_20d", "rsi_14d", "bollinger_z_20d"])
    return valid_panel
