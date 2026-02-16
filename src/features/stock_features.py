"""Stock-specific feature engineering.

This module computes fundamental quality and anomaly signals:
- Gross profitability / assets (Novy-Marx quality factor)
- Asset growth (low growth = higher returns)
- Net operating assets (accruals mean reversion)
- O-score (Ohlson bankruptcy prediction)
- Insider buy/sell ratio
"""

import logging
from typing import Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def compute_gross_profitability_assets(fundamentals: pd.DataFrame) -> pd.Series:
    """Compute (Revenue - COGS) / Total Assets.
    
    Quality factor from Novy-Marx (2013). High profitability predicts higher returns.
    
    Args:
        fundamentals: DataFrame with columns [timestamp, ticker, revenue, gross_profit, total_assets]
        
    Returns:
        Series indexed by (timestamp, ticker) with profitability ratio
    """
    logger.info("Computing gross profitability / assets")
    
    if fundamentals.empty:
        logger.warning("Empty fundamentals DataFrame")
        return pd.Series(dtype=float)
    
    df = fundamentals.copy()
    
    # Compute gross profitability / assets
    # If gross_profit is available, use it; otherwise compute from revenue - COGS
    if 'gross_profit' in df.columns:
        df['gross_profitability_assets'] = df['gross_profit'] / df['total_assets']
    else:
        logger.warning("gross_profit column not found, using revenue as proxy")
        df['gross_profitability_assets'] = df['revenue'] / df['total_assets']
    
    # Handle division by zero
    df['gross_profitability_assets'] = df['gross_profitability_assets'].replace([np.inf, -np.inf], np.nan)
    
    result = df.set_index(['timestamp', 'ticker'])['gross_profitability_assets']
    
    logger.info(f"Computed gross profitability for {df['ticker'].nunique()} stocks")
    return result


def compute_asset_growth_1y(fundamentals: pd.DataFrame) -> pd.Series:
    """Compute year-over-year percentage change in total assets.
    
    Low asset growth predicts higher future returns (asset growth anomaly).
    
    Args:
        fundamentals: DataFrame with columns [timestamp, ticker, total_assets]
        
    Returns:
        Series indexed by (timestamp, ticker) with asset growth percentage
    """
    logger.info("Computing asset growth (1-year)")
    
    if fundamentals.empty:
        logger.warning("Empty fundamentals DataFrame")
        return pd.Series(dtype=float)
    
    df = fundamentals.sort_values(['ticker', 'timestamp']).copy()
    
    # Compute 1-year (4 quarters) percentage change
    df['asset_growth_1y'] = df.groupby('ticker')['total_assets'].pct_change(periods=4) * 100
    
    result = df.set_index(['timestamp', 'ticker'])['asset_growth_1y']
    
    logger.info(f"Computed asset growth for {df['ticker'].nunique()} stocks")
    return result


def compute_net_operating_assets(fundamentals: pd.DataFrame) -> pd.Series:
    """Compute (Operating Assets - Operating Liabilities) / Total Assets.
    
    High accruals (high NOA) predict mean reversion.
    
    Args:
        fundamentals: DataFrame with columns [timestamp, ticker, total_assets, total_liabilities, 
                                              operating_income]
        
    Returns:
        Series indexed by (timestamp, ticker) with NOA ratio
    """
    logger.info("Computing net operating assets")
    
    if fundamentals.empty:
        logger.warning("Empty fundamentals DataFrame")
        return pd.Series(dtype=float)
    
    df = fundamentals.copy()
    
    # Simplified NOA calculation
    # Operating assets ≈ Total assets - Cash
    # Operating liabilities ≈ Total liabilities - Debt
    # For simplicity, use total assets and liabilities as proxy
    
    if 'total_liabilities' in df.columns:
        df['net_operating_assets'] = (df['total_assets'] - df['total_liabilities']) / df['total_assets']
    else:
        logger.warning("total_liabilities column not found, using simplified calculation")
        df['net_operating_assets'] = np.nan
    
    # Handle division by zero
    df['net_operating_assets'] = df['net_operating_assets'].replace([np.inf, -np.inf], np.nan)
    
    result = df.set_index(['timestamp', 'ticker'])['net_operating_assets']
    
    logger.info(f"Computed NOA for {df['ticker'].nunique()} stocks")
    return result


def compute_o_score(fundamentals: pd.DataFrame) -> pd.Series:
    """Compute Ohlson bankruptcy prediction score.
    
    Uses logit model with size, leverage, liquidity, and profitability ratios.
    Lower scores indicate financial distress (potential value opportunities).
    
    Simplified O-Score formula:
    O = -1.32 - 0.407*log(TA) + 6.03*(TL/TA) - 1.43*(WC/TA) + 0.0757*(CL/CA) 
        - 2.37*(NI/TA) - 1.83*(FFO/TL) + 0.285*OENEG - 1.72*INTWO
    
    Args:
        fundamentals: DataFrame with columns [timestamp, ticker, total_assets, total_liabilities,
                                              revenue, operating_income]
        
    Returns:
        Series indexed by (timestamp, ticker) with O-score
    """
    logger.info("Computing O-score (Ohlson bankruptcy model)")
    
    if fundamentals.empty:
        logger.warning("Empty fundamentals DataFrame")
        return pd.Series(dtype=float)
    
    df = fundamentals.copy()
    
    # Simplified O-score using available data
    # Full implementation would require more detailed balance sheet items
    
    try:
        # Size: log of total assets
        log_ta = np.log(df['total_assets'].replace(0, np.nan))
        
        # Leverage: total liabilities / total assets
        leverage = df['total_liabilities'] / df['total_assets']
        
        # Profitability: operating income / total assets
        profitability = df['operating_income'] / df['total_assets']
        
        # Simplified O-score (using available components)
        df['o_score'] = (
            -1.32 
            - 0.407 * log_ta 
            + 6.03 * leverage 
            - 2.37 * profitability
        )
        
        # Handle invalid values
        df['o_score'] = df['o_score'].replace([np.inf, -np.inf], np.nan)
        
    except Exception as e:
        logger.error(f"Failed to compute O-score: {e}")
        df['o_score'] = np.nan
    
    result = df.set_index(['timestamp', 'ticker'])['o_score']
    
    logger.info(f"Computed O-score for {df['ticker'].nunique()} stocks")
    return result


def compute_insider_buy_sell_ratio(insider_trades: pd.DataFrame) -> pd.Series:
    """Compute ratio of insider buy volume to sell volume over 90 days.
    
    High ratio indicates insider conviction (bullish signal).
    
    Args:
        insider_trades: DataFrame with columns [timestamp, ticker, transaction_type, shares, price]
        
    Returns:
        Series indexed by (timestamp, ticker) with buy/sell ratio
    """
    logger.info("Computing insider buy/sell ratio (90-day)")
    
    if insider_trades.empty:
        logger.warning("Empty insider_trades DataFrame")
        return pd.Series(dtype=float)
    
    df = insider_trades.copy()
    
    # Calculate transaction value
    df['value'] = df['shares'] * df['price']
    
    # Separate buys and sells
    buys = df[df['transaction_type'] == 'buy'].groupby(['timestamp', 'ticker'])['value'].sum()
    sells = df[df['transaction_type'] == 'sell'].groupby(['timestamp', 'ticker'])['value'].sum()
    
    # Compute ratio
    ratio = buys / sells
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    
    # Fill missing with 1.0 (neutral)
    ratio = ratio.fillna(1.0)
    
    logger.info(f"Computed insider ratio for {len(ratio)} ticker-date pairs")
    return ratio


def compute_all_stock_features(
    fundamentals: pd.DataFrame,
    insider_trades: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Compute all stock features.
    
    Args:
        fundamentals: Fundamental data with columns [timestamp, ticker, revenue, gross_profit,
                                                     total_assets, total_liabilities, operating_income]
        insider_trades: Insider trading data (optional)
        
    Returns:
        DataFrame with all stock features
    """
    logger.info("Computing all stock features")
    
    # Start with fundamentals
    features = fundamentals[['timestamp', 'ticker']].copy()
    features = features.set_index(['timestamp', 'ticker'])
    
    # Compute gross profitability
    try:
        gross_prof = compute_gross_profitability_assets(fundamentals)
        features = features.join(gross_prof, how='left')
    except Exception as e:
        logger.error(f"Failed to compute gross profitability: {e}")
        features['gross_profitability_assets'] = np.nan
    
    # Compute asset growth
    try:
        asset_growth = compute_asset_growth_1y(fundamentals)
        features = features.join(asset_growth, how='left')
    except Exception as e:
        logger.error(f"Failed to compute asset growth: {e}")
        features['asset_growth_1y'] = np.nan
    
    # Compute net operating assets
    try:
        noa = compute_net_operating_assets(fundamentals)
        features = features.join(noa, how='left')
    except Exception as e:
        logger.error(f"Failed to compute NOA: {e}")
        features['net_operating_assets'] = np.nan
    
    # Compute O-score
    try:
        o_score = compute_o_score(fundamentals)
        features = features.join(o_score, how='left')
    except Exception as e:
        logger.error(f"Failed to compute O-score: {e}")
        features['o_score'] = np.nan
    
    # Compute insider buy/sell ratio
    if insider_trades is not None and not insider_trades.empty:
        try:
            insider_ratio = compute_insider_buy_sell_ratio(insider_trades)
            features = features.join(insider_ratio.rename('insider_buy_sell_ratio'), how='left')
        except Exception as e:
            logger.error(f"Failed to compute insider ratio: {e}")
            features['insider_buy_sell_ratio'] = np.nan
    else:
        features['insider_buy_sell_ratio'] = np.nan
    
    logger.info(f"Computed stock features: {features.shape}")
    return features.reset_index()


def handle_missing_fundamentals(
    df: pd.DataFrame,
    method: str = 'forward_fill'
) -> pd.DataFrame:
    """Handle missing fundamental data.
    
    Fundamental data is typically quarterly, so forward-fill is appropriate.
    
    Args:
        df: DataFrame with stock features
        method: Method to handle missing data ('forward_fill' or 'drop')
        
    Returns:
        DataFrame with missing data handled
    """
    logger.info(f"Handling missing fundamental data (method: {method})")
    
    df = df.sort_values(['ticker', 'timestamp']).copy()
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'ticker']]
    
    if method == 'forward_fill':
        # Forward-fill within each ticker group (fundamentals are quarterly)
        for col in feature_cols:
            df[col] = df.groupby('ticker')[col].fillna(method='ffill')
    elif method == 'drop':
        # Drop rows with any missing fundamentals
        df = df.dropna(subset=feature_cols)
    
    missing_counts = df[feature_cols].isna().sum()
    if missing_counts.sum() > 0:
        logger.warning(f"Remaining missing values:\n{missing_counts[missing_counts > 0]}")
    
    return df


def main():
    """Main function for testing stock feature engineering."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create sample data for testing
    dates = pd.date_range('2024-01-01', periods=20, freq='Q')  # Quarterly data
    
    # Sample fundamental data
    fundamentals = pd.DataFrame({
        'timestamp': dates.repeat(3),
        'ticker': ['AAPL', 'MSFT', 'GOOGL'] * 20,
        'revenue': np.random.randn(60).cumsum() + 100000,
        'gross_profit': np.random.randn(60).cumsum() + 50000,
        'total_assets': np.random.randn(60).cumsum() + 200000,
        'total_liabilities': np.random.randn(60).cumsum() + 100000,
        'operating_income': np.random.randn(60).cumsum() + 30000
    })
    
    # Compute features
    features = compute_all_stock_features(fundamentals=fundamentals)
    
    logger.info(f"Features shape: {features.shape}")
    logger.info(f"Feature columns: {features.columns.tolist()}")
    logger.info(f"\nSample features:\n{features.head()}")


if __name__ == "__main__":
    main()
