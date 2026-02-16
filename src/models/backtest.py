"""Backtesting engine for portfolio simulation."""

import logging
from typing import Dict
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import get_config
from src.models.train import BreakoutPredictor

logger = logging.getLogger(__name__)


def run_backtest(start_date: pd.Timestamp, end_date: pd.Timestamp,
                 top_n: int = 10, rebalance_days: int = 30,
                 initial_capital: float = 100000.0) -> Dict:
    """
    Run portfolio backtest simulation.
    
    Args:
        start_date: Backtest start date
        end_date: Backtest end date
        top_n: Number of assets in portfolio
        rebalance_days: Days between rebalancing
        initial_capital: Starting capital
    
    Returns:
        Dictionary with backtest results
    """
    config = get_config()
    
    # Load features
    features_path = config.features_dir / 'features.parquet'
    if not features_path.exists():
        raise FileNotFoundError(f"Features not found at {features_path}")
    
    features = pd.read_parquet(features_path)
    features = features[(features['timestamp'] >= start_date) & 
                       (features['timestamp'] <= end_date)]
    
    if features.empty:
        raise ValueError(f"No data found between {start_date} and {end_date}")
    
    # Load model
    predictor = BreakoutPredictor()
    predictor.load(config.models_dir)
    
    # Get feature columns
    feature_cols = [col for col in features.columns 
                   if col not in ['timestamp', 'asset', 'close', 'asset_type']]
    
    # Simulate portfolio
    portfolio_values = []
    dates = []
    holdings = {}
    capital = initial_capital
    
    rebalance_dates = pd.date_range(start_date, end_date, freq=f'{rebalance_days}D')
    
    for rebalance_date in rebalance_dates:
        # Get data at rebalance date
        date_features = features[features['timestamp'] <= rebalance_date]
        date_features = date_features.sort_values('timestamp').groupby('asset').tail(1)
        
        if date_features.empty:
            continue
        
        # Predict scores
        X = date_features[feature_cols].fillna(0)
        scores = predictor.predict_proba(X)
        date_features['score'] = scores
        
        # Select top N assets
        top_assets = date_features.nlargest(top_n, 'score')
        
        # Rebalance portfolio (equal weight)
        new_holdings = {}
        position_size = capital / top_n
        
        for _, row in top_assets.iterrows():
            asset = row['asset']
            price = row['close']
            if price > 0:
                shares = position_size / price
                new_holdings[asset] = {'shares': shares, 'entry_price': price}
        
        holdings = new_holdings
        
        # Track portfolio value
        portfolio_value = sum(
            h['shares'] * h['entry_price'] for h in holdings.values()
        )
        portfolio_values.append(portfolio_value)
        dates.append(rebalance_date)
        
        logger.info(f"Rebalanced on {rebalance_date}: {len(holdings)} positions, value=${portfolio_value:,.0f}")
    
    # Calculate metrics
    if len(portfolio_values) < 2:
        raise ValueError("Insufficient data for backtest")
    
    returns = pd.Series(portfolio_values).pct_change().dropna()
    
    total_return = (portfolio_values[-1] - initial_capital) / initial_capital
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 / rebalance_days) if returns.std() > 0 else 0
    
    # Calculate drawdowns
    cumulative = pd.Series(portfolio_values)
    running_max = cumulative.expanding().max()
    drawdowns = (cumulative - running_max) / running_max * 100
    max_drawdown = drawdowns.min()
    
    # Win rate
    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
    
    results = {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'portfolio_values': portfolio_values,
        'dates': dates,
        'drawdowns': drawdowns.tolist(),
        'final_value': portfolio_values[-1],
        'num_rebalances': len(rebalance_dates)
    }
    
    logger.info(f"Backtest complete: Return={total_return:.1%}, Sharpe={sharpe_ratio:.2f}, MaxDD={max_drawdown:.1%}")
    
    return results


def main():
    """CLI entry point for backtesting."""
    logging.basicConfig(level=logging.INFO)
    
    # Run 6-month backtest
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=180)
    
    results = run_backtest(
        start_date=start_date,
        end_date=end_date,
        top_n=10,
        rebalance_days=30
    )
    
    print("\n" + "="*80)
    print("💰 MONEYBALL BACKTEST RESULTS")
    print("="*80)
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Total Return: {results['total_return']:.1%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.1%}")
    print(f"Win Rate: {results['win_rate']:.1%}")
    print(f"Final Value: ${results['final_value']:,.0f}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
