"""Cryptocurrency-specific feature engineering.

This module computes hidden signals for crypto assets:
- Whale accumulation patterns
- Exchange netflow divergence
- Stablecoin supply ratio
- DeFi TVL momentum
- Developer activity
"""

import logging
from typing import Optional
import pandas as pd
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def compute_whale_accumulation_7d(whale_balances: pd.DataFrame) -> pd.Series:
    """Compute 7-day percentage change in top 1% holder balances.
    
    This signal captures whale accumulation before price moves.
    Positive values indicate whales are accumulating.
    
    Args:
        whale_balances: DataFrame with columns [timestamp, asset, top1pct_balance]
        
    Returns:
        Series indexed by (timestamp, asset) with accumulation percentage
        
    Example:
        >>> df = pd.DataFrame({
        ...     'timestamp': pd.date_range('2024-01-01', periods=10),
        ...     'asset': 'BTC',
        ...     'top1pct_balance': [100, 102, 105, 103, 108, 110, 112, 115, 118, 120]
        ... })
        >>> result = compute_whale_accumulation_7d(df)
    """
    logger.info("Computing whale accumulation (7-day)")
    
    if whale_balances.empty:
        logger.warning("Empty whale_balances DataFrame")
        return pd.Series(dtype=float)
    
    # Sort by asset and timestamp
    df = whale_balances.sort_values(['asset', 'timestamp']).copy()
    
    # Compute 7-day percentage change
    df['whale_accumulation_7d'] = df.groupby('asset')['top1pct_balance'].pct_change(periods=7) * 100
    
    # Set multi-index
    result = df.set_index(['timestamp', 'asset'])['whale_accumulation_7d']
    
    logger.info(f"Computed whale accumulation for {df['asset'].nunique()} assets")
    return result


def compute_exchange_netflow_divergence(
    prices: pd.DataFrame,
    netflows: pd.DataFrame
) -> pd.Series:
    """Compute divergence between price change and exchange net flows.
    
    Measures correlation between 7-day price return and 7-day net flow z-score.
    High divergence (price up, flows negative) indicates accumulation not yet priced in.
    
    Args:
        prices: DataFrame with columns [timestamp, asset, close]
        netflows: DataFrame with columns [timestamp, asset, netflow_usd]
        
    Returns:
        Series indexed by (timestamp, asset) with divergence score
    """
    logger.info("Computing exchange netflow divergence")
    
    if prices.empty or netflows.empty:
        logger.warning("Empty prices or netflows DataFrame")
        return pd.Series(dtype=float)
    
    # Merge prices and netflows
    df = pd.merge(
        prices[['timestamp', 'asset', 'close']],
        netflows[['timestamp', 'asset', 'netflow_usd']],
        on=['timestamp', 'asset'],
        how='inner'
    )
    
    if df.empty:
        logger.warning("No matching data after merge")
        return pd.Series(dtype=float)
    
    df = df.sort_values(['asset', 'timestamp']).copy()
    
    # Compute 7-day price return
    df['price_return_7d'] = df.groupby('asset')['close'].pct_change(periods=7)
    
    # Compute 7-day rolling z-score of netflows
    df['netflow_zscore'] = df.groupby('asset')['netflow_usd'].transform(
        lambda x: (x - x.rolling(90, min_periods=30).mean()) / x.rolling(90, min_periods=30).std()
    )
    
    # Divergence = price return - netflow z-score
    # Positive divergence: price up but netflows negative (accumulation)
    df['exchange_netflow_divergence'] = df['price_return_7d'] - df['netflow_zscore']
    
    result = df.set_index(['timestamp', 'asset'])['exchange_netflow_divergence']
    
    logger.info(f"Computed netflow divergence for {df['asset'].nunique()} assets")
    return result


def compute_stablecoin_supply_ratio(
    stablecoin_supply: pd.DataFrame,
    total_market_cap: pd.DataFrame
) -> pd.Series:
    """Compute ratio of stablecoin supply to total market cap.
    
    High SSR indicates "dry powder" ready to flow into crypto.
    
    Args:
        stablecoin_supply: DataFrame with columns [timestamp, total_supply_usd]
        total_market_cap: DataFrame with columns [timestamp, total_mcap_usd]
        
    Returns:
        Series indexed by timestamp with SSR ratio
    """
    logger.info("Computing stablecoin supply ratio")
    
    if stablecoin_supply.empty or total_market_cap.empty:
        logger.warning("Empty stablecoin_supply or total_market_cap DataFrame")
        return pd.Series(dtype=float)
    
    # Merge on timestamp
    df = pd.merge(
        stablecoin_supply[['timestamp', 'total_supply_usd']],
        total_market_cap[['timestamp', 'total_mcap_usd']],
        on='timestamp',
        how='inner'
    )
    
    if df.empty:
        logger.warning("No matching data after merge")
        return pd.Series(dtype=float)
    
    # Compute ratio
    df['stablecoin_supply_ratio'] = df['total_supply_usd'] / df['total_mcap_usd']
    
    result = df.set_index('timestamp')['stablecoin_supply_ratio']
    
    logger.info(f"Computed SSR for {len(result)} timestamps")
    return result


def compute_defi_tvl_momentum_30d(tvl_data: pd.DataFrame) -> pd.Series:
    """Compute 30-day rate of change in DeFi total value locked.
    
    Captures capital rotation into/out of DeFi protocols.
    
    Args:
        tvl_data: DataFrame with columns [timestamp, protocol, tvl_usd]
        
    Returns:
        Series indexed by timestamp with TVL momentum
    """
    logger.info("Computing DeFi TVL momentum (30-day)")
    
    if tvl_data.empty:
        logger.warning("Empty tvl_data DataFrame")
        return pd.Series(dtype=float)
    
    # Aggregate TVL across all protocols
    df = tvl_data.groupby('timestamp')['tvl_usd'].sum().reset_index()
    df = df.sort_values('timestamp').copy()
    
    # Compute 30-day percentage change
    df['defi_tvl_momentum_30d'] = df['tvl_usd'].pct_change(periods=30) * 100
    
    result = df.set_index('timestamp')['defi_tvl_momentum_30d']
    
    logger.info(f"Computed TVL momentum for {len(result)} timestamps")
    return result


def compute_dev_activity_30d(github_data: pd.DataFrame) -> pd.Series:
    """Compute developer activity as commit count in past 30 days.
    
    Measures development velocity and project health.
    
    Args:
        github_data: DataFrame with columns [timestamp, repo_name, commit_count]
        
    Returns:
        Series indexed by (timestamp, repo_name) with commit counts
    """
    logger.info("Computing developer activity (30-day)")
    
    if github_data.empty:
        logger.warning("Empty github_data DataFrame")
        return pd.Series(dtype=float)
    
    df = github_data.copy()
    
    # Rename commit_count to dev_activity_30d for consistency
    df['dev_activity_30d'] = df['commit_count']
    
    result = df.set_index(['timestamp', 'repo_name'])['dev_activity_30d']
    
    logger.info(f"Computed dev activity for {df['repo_name'].nunique()} repos")
    return result


def compute_all_crypto_features(
    prices: pd.DataFrame,
    whale_balances: Optional[pd.DataFrame] = None,
    netflows: Optional[pd.DataFrame] = None,
    stablecoin_supply: Optional[pd.DataFrame] = None,
    total_market_cap: Optional[pd.DataFrame] = None,
    tvl_data: Optional[pd.DataFrame] = None,
    github_data: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Compute all cryptocurrency features.
    
    Args:
        prices: Price data with columns [timestamp, asset, close]
        whale_balances: Whale balance data (optional)
        netflows: Exchange netflow data (optional)
        stablecoin_supply: Stablecoin supply data (optional)
        total_market_cap: Total market cap data (optional)
        tvl_data: DeFi TVL data (optional)
        github_data: GitHub commit data (optional)
        
    Returns:
        DataFrame with all crypto features
    """
    logger.info("Computing all crypto features")
    
    # Start with prices
    features = prices[['timestamp', 'asset', 'close']].copy()
    features = features.set_index(['timestamp', 'asset'])
    
    # Compute whale accumulation
    if whale_balances is not None and not whale_balances.empty:
        try:
            whale_acc = compute_whale_accumulation_7d(whale_balances)
            features = features.join(whale_acc, how='left')
        except Exception as e:
            logger.error(f"Failed to compute whale accumulation: {e}")
            features['whale_accumulation_7d'] = np.nan
    else:
        features['whale_accumulation_7d'] = np.nan
        
    # Compute exchange netflow divergence
    if netflows is not None and not netflows.empty:
        try:
            netflow_div = compute_exchange_netflow_divergence(prices, netflows)
            features = features.join(netflow_div, how='left')
        except Exception as e:
            logger.error(f"Failed to compute netflow divergence: {e}")
            features['exchange_netflow_divergence'] = np.nan
    else:
        features['exchange_netflow_divergence'] = np.nan
        
    # Compute stablecoin supply ratio (broadcast to all assets)
    if stablecoin_supply is not None and total_market_cap is not None:
        try:
            ssr = compute_stablecoin_supply_ratio(stablecoin_supply, total_market_cap)
            # Broadcast to all assets
            features = features.reset_index()
            features = pd.merge(
                features,
                ssr.reset_index().rename(columns={'stablecoin_supply_ratio': 'stablecoin_supply_ratio'}),
                on='timestamp',
                how='left'
            )
            features = features.set_index(['timestamp', 'asset'])
        except Exception as e:
            logger.error(f"Failed to compute stablecoin supply ratio: {e}")
            features['stablecoin_supply_ratio'] = np.nan
    else:
        features['stablecoin_supply_ratio'] = np.nan
        
    # Compute DeFi TVL momentum (broadcast to all assets)
    if tvl_data is not None and not tvl_data.empty:
        try:
            tvl_momentum = compute_defi_tvl_momentum_30d(tvl_data)
            # Broadcast to all assets
            features = features.reset_index()
            features = pd.merge(
                features,
                tvl_momentum.reset_index().rename(columns={'defi_tvl_momentum_30d': 'defi_tvl_momentum_30d'}),
                on='timestamp',
                how='left'
            )
            features = features.set_index(['timestamp', 'asset'])
        except Exception as e:
            logger.error(f"Failed to compute TVL momentum: {e}")
            features['defi_tvl_momentum_30d'] = np.nan
    else:
        features['defi_tvl_momentum_30d'] = np.nan
        
    # Compute developer activity
    # Note: This requires mapping assets to repo names
    if github_data is not None and not github_data.empty:
        try:
            dev_activity = compute_dev_activity_30d(github_data)
            # This would need asset-to-repo mapping
            # For now, leave as NaN
            features['dev_activity_30d'] = np.nan
        except Exception as e:
            logger.error(f"Failed to compute dev activity: {e}")
            features['dev_activity_30d'] = np.nan
    else:
        features['dev_activity_30d'] = np.nan
        
    logger.info(f"Computed crypto features: {features.shape}")
    return features.reset_index()


def handle_missing_data(
    df: pd.DataFrame,
    forward_fill_days: int = 7
) -> pd.DataFrame:
    """Handle missing data in crypto features.
    
    Forward-fills up to specified days, then marks as NaN.
    
    Args:
        df: DataFrame with features
        forward_fill_days: Maximum days to forward-fill
        
    Returns:
        DataFrame with missing data handled
    """
    logger.info(f"Handling missing data (forward-fill up to {forward_fill_days} days)")
    
    df = df.sort_values(['asset', 'timestamp']).copy()
    
    # Get feature columns (exclude timestamp and asset)
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'asset', 'close']]
    
    for col in feature_cols:
        # Forward-fill within each asset group
        df[col] = df.groupby('asset')[col].fillna(method='ffill', limit=forward_fill_days)
        
    missing_counts = df[feature_cols].isna().sum()
    if missing_counts.sum() > 0:
        logger.warning(f"Remaining missing values after forward-fill:\n{missing_counts[missing_counts > 0]}")
        
    return df


def main():
    """Main function for testing crypto feature engineering."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create sample data for testing
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    # Sample price data
    prices = pd.DataFrame({
        'timestamp': dates.repeat(3),
        'asset': ['BTC', 'ETH', 'SOL'] * 100,
        'close': np.random.randn(300).cumsum() + 100
    })
    
    # Sample whale balance data
    whale_balances = pd.DataFrame({
        'timestamp': dates.repeat(3),
        'asset': ['BTC', 'ETH', 'SOL'] * 100,
        'top1pct_balance': np.random.randn(300).cumsum() + 1000
    })
    
    # Compute features
    features = compute_all_crypto_features(
        prices=prices,
        whale_balances=whale_balances
    )
    
    logger.info(f"Features shape: {features.shape}")
    logger.info(f"Feature columns: {features.columns.tolist()}")
    logger.info(f"\nSample features:\n{features.head()}")


if __name__ == "__main__":
    main()
