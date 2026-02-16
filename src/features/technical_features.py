"""Technical indicator feature engineering.

This module computes standard technical indicators for all assets:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger %B
- ATR (Average True Range)
- Volume change
- Price momentum
"""

import logging
from typing import Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index.
    
    Args:
        prices: Price series
        period: RSI period (default 14)
        
    Returns:
        RSI values (0-100)
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def compute_macd(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> pd.DataFrame:
    """Compute MACD (Moving Average Convergence Divergence).
    
    Args:
        prices: Price series
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line period (default 9)
        
    Returns:
        DataFrame with macd, signal, and histogram columns
    """
    ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
    ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return pd.DataFrame({
        'macd': macd_line,
        'macd_signal': signal_line,
        'macd_histogram': histogram
    })


def compute_bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    num_std: float = 2.0
) -> pd.DataFrame:
    """Compute Bollinger Bands and %B.
    
    Args:
        prices: Price series
        period: Moving average period (default 20)
        num_std: Number of standard deviations (default 2)
        
    Returns:
        DataFrame with upper, middle, lower bands and %B
    """
    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    
    # %B: position within bands (0 = lower band, 1 = upper band)
    pct_b = (prices - lower) / (upper - lower)
    
    return pd.DataFrame({
        'bb_upper': upper,
        'bb_middle': middle,
        'bb_lower': lower,
        'bollinger_pct_b': pct_b
    })


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """Compute Average True Range.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period (default 14)
        
    Returns:
        ATR values
    """
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr


def compute_technical_indicators(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute RSI, MACD, Bollinger %B, ATR, volume change, momentum.
    
    Args:
        ohlcv: DataFrame with columns [timestamp, asset, open, high, low, close, volume]
        
    Returns:
        DataFrame with additional columns for each technical indicator
    """
    logger.info("Computing technical indicators")
    
    if ohlcv.empty:
        logger.warning("Empty OHLCV DataFrame")
        return pd.DataFrame()
    
    df = ohlcv.sort_values(['asset', 'timestamp']).copy()
    
    # Initialize result columns
    df['rsi_14'] = np.nan
    df['macd'] = np.nan
    df['macd_signal'] = np.nan
    df['macd_histogram'] = np.nan
    df['bollinger_pct_b'] = np.nan
    df['atr_14'] = np.nan
    df['volume_change_7d'] = np.nan
    df['momentum_1m'] = np.nan
    df['momentum_3m'] = np.nan
    
    # Compute indicators for each asset
    for asset in df['asset'].unique():
        mask = df['asset'] == asset
        asset_data = df[mask].copy()
        
        if len(asset_data) < 30:  # Need minimum data for indicators
            logger.debug(f"Insufficient data for {asset} ({len(asset_data)} rows)")
            continue
        
        try:
            # RSI
            df.loc[mask, 'rsi_14'] = compute_rsi(asset_data['close'], period=14).values
            
            # MACD
            macd_data = compute_macd(asset_data['close'])
            df.loc[mask, 'macd'] = macd_data['macd'].values
            df.loc[mask, 'macd_signal'] = macd_data['macd_signal'].values
            df.loc[mask, 'macd_histogram'] = macd_data['macd_histogram'].values
            
            # Bollinger Bands
            bb_data = compute_bollinger_bands(asset_data['close'])
            df.loc[mask, 'bollinger_pct_b'] = bb_data['bollinger_pct_b'].values
            
            # ATR
            df.loc[mask, 'atr_14'] = compute_atr(
                asset_data['high'],
                asset_data['low'],
                asset_data['close'],
                period=14
            ).values
            
            # Volume change (7-day)
            df.loc[mask, 'volume_change_7d'] = asset_data['volume'].pct_change(periods=7).values * 100
            
            # Price momentum
            df.loc[mask, 'momentum_1m'] = asset_data['close'].pct_change(periods=30).values * 100
            df.loc[mask, 'momentum_3m'] = asset_data['close'].pct_change(periods=90).values * 100
            
        except Exception as e:
            logger.error(f"Failed to compute indicators for {asset}: {e}")
            continue
    
    logger.info(f"Computed technical indicators for {df['asset'].nunique()} assets")
    return df


def handle_insufficient_history(
    df: pd.DataFrame,
    min_periods: int = 90
) -> pd.DataFrame:
    """Handle assets with insufficient price history.
    
    Marks indicators as NaN for assets with less than minimum periods.
    
    Args:
        df: DataFrame with technical indicators
        min_periods: Minimum number of periods required
        
    Returns:
        DataFrame with insufficient history handled
    """
    logger.info(f"Handling insufficient history (min {min_periods} periods)")
    
    # Count periods per asset
    period_counts = df.groupby('asset').size()
    insufficient_assets = period_counts[period_counts < min_periods].index
    
    if len(insufficient_assets) > 0:
        logger.warning(f"Found {len(insufficient_assets)} assets with insufficient history")
        
        # Mark indicators as NaN for these assets
        indicator_cols = [
            'rsi_14', 'macd', 'macd_signal', 'macd_histogram',
            'bollinger_pct_b', 'atr_14', 'volume_change_7d',
            'momentum_1m', 'momentum_3m'
        ]
        
        for col in indicator_cols:
            if col in df.columns:
                df.loc[df['asset'].isin(insufficient_assets), col] = np.nan
    
    return df


def compute_all_technical_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical features with error handling.
    
    Args:
        ohlcv: DataFrame with OHLCV data
        
    Returns:
        DataFrame with all technical features
    """
    logger.info("Computing all technical features")
    
    # Compute indicators
    features = compute_technical_indicators(ohlcv)
    
    # Handle insufficient history
    features = handle_insufficient_history(features, min_periods=90)
    
    # Select relevant columns
    feature_cols = [
        'timestamp', 'asset', 'close',
        'rsi_14', 'macd', 'bollinger_pct_b', 'atr_14',
        'volume_change_7d', 'momentum_1m', 'momentum_3m'
    ]
    
    features = features[feature_cols]
    
    logger.info(f"Technical features shape: {features.shape}")
    return features


def main():
    """Main function for testing technical feature engineering."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create sample OHLCV data
    dates = pd.date_range('2024-01-01', periods=200, freq='D')
    
    np.random.seed(42)
    base_price = 100
    
    ohlcv = pd.DataFrame({
        'timestamp': dates.repeat(3),
        'asset': ['BTC', 'ETH', 'SOL'] * 200,
        'open': np.random.randn(600).cumsum() + base_price,
        'high': np.random.randn(600).cumsum() + base_price + 2,
        'low': np.random.randn(600).cumsum() + base_price - 2,
        'close': np.random.randn(600).cumsum() + base_price,
        'volume': np.random.randint(1000000, 10000000, 600)
    })
    
    # Compute features
    features = compute_all_technical_features(ohlcv)
    
    logger.info(f"Features shape: {features.shape}")
    logger.info(f"Feature columns: {features.columns.tolist()}")
    logger.info(f"\nSample features:\n{features.head(50)}")
    
    # Check for missing values
    missing = features.isna().sum()
    logger.info(f"\nMissing values:\n{missing[missing > 0]}")


if __name__ == "__main__":
    main()
