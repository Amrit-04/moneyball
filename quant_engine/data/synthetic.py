"""Deterministic Multi-Asset Market & Factor Dataset Generator.

Generates a realistic, cross-sectional daily panel of 24 assets spanning Equities,
ETFs, Fixed Income, Commodities, and Cryptocurrencies over 1,000+ trading days.
Uses calibrated drift, correlated Geometric Brownian Motion with Merton Jump-Diffusion,
and realistic volume dynamics with 100% offline reproducibility via fixed seed.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


DEFAULT_ASSETS: List[str] = [
    # US Mega-Cap Tech & Growth
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    # US Value, Financials & Healthcare
    "JPM", "V", "JNJ", "WMT", "PG", "XOM", "UNH",
    # ETFs & Macro Assets (Equities, High Yield, Treasuries, Gold)
    "SPY", "QQQ", "IWM", "HYG", "TLT", "GLD",
    # Cryptocurrencies
    "BTC", "ETH", "SOL", "BNB"
]

# Asset class mappings
ASSET_CLASSES: Dict[str, str] = {
    "AAPL": "TECH", "MSFT": "TECH", "GOOGL": "TECH", "AMZN": "TECH",
    "NVDA": "TECH", "META": "TECH", "TSLA": "TECH",
    "JPM": "FIN", "V": "FIN", "JNJ": "HEALTH", "WMT": "CONSUMER",
    "PG": "CONSUMER", "XOM": "ENERGY", "UNH": "HEALTH",
    "SPY": "INDEX", "QQQ": "INDEX", "IWM": "INDEX",
    "HYG": "CREDIT", "TLT": "RATES", "GLD": "COMMODITY",
    "BTC": "CRYPTO", "ETH": "CRYPTO", "SOL": "CRYPTO", "BNB": "CRYPTO"
}


def build_calibrated_covariance(assets: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build calibrated annualized drift vector, volatility vector, and correlation matrix.
    
    Args:
        assets: List of asset ticker symbols.
        
    Returns:
        Tuple of (drift_vector, vol_vector, correlation_matrix)
    """
    n = len(assets)
    drifts = np.zeros(n)
    vols = np.zeros(n)
    
    for i, a in enumerate(assets):
        aclass = ASSET_CLASSES.get(a, "TECH")
        if aclass == "TECH":
            drifts[i] = 0.16
            vols[i] = 0.28
        elif aclass == "FIN":
            drifts[i] = 0.11
            vols[i] = 0.22
        elif aclass == "HEALTH" or aclass == "CONSUMER":
            drifts[i] = 0.08
            vols[i] = 0.16
        elif aclass == "ENERGY":
            drifts[i] = 0.10
            vols[i] = 0.26
        elif aclass == "INDEX":
            drifts[i] = 0.12 if a != "IWM" else 0.09
            vols[i] = 0.18 if a != "IWM" else 0.23
        elif aclass == "CREDIT":
            drifts[i] = 0.06
            vols[i] = 0.10
        elif aclass == "RATES":
            drifts[i] = 0.03
            vols[i] = 0.14
        elif aclass == "COMMODITY":
            drifts[i] = 0.07
            vols[i] = 0.15
        elif aclass == "CRYPTO":
            drifts[i] = 0.28
            vols[i] = 0.65
        else:
            drifts[i] = 0.10
            vols[i] = 0.20
            
    # Base correlation matrix
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            c1, c2 = ASSET_CLASSES.get(assets[i]), ASSET_CLASSES.get(assets[j])
            if c1 == c2:
                base_c = 0.70 if c1 != "CRYPTO" else 0.82
            elif "CRYPTO" in (c1, c2):
                base_c = 0.35 if "TECH" in (c1, c2) or "INDEX" in (c1, c2) else 0.15
            elif "RATES" in (c1, c2):
                base_c = -0.30 if "INDEX" in (c1, c2) or "TECH" in (c1, c2) else -0.10
            elif "COMMODITY" in (c1, c2):
                base_c = 0.10
            else:
                base_c = 0.50
            corr[i, j] = base_c
            corr[j, i] = base_c
            
    # Ensure positive semi-definiteness via eigenvalue clipping
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 1e-4)
    corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    
    return drifts, vols, corr


def generate_deterministic_market_panel(
    assets: Optional[List[str]] = None,
    n_days: int = 1260,
    seed: int = 42,
    start_date: str = "2020-01-02",
) -> Dict[str, pd.DataFrame]:
    """Generate deterministic historical daily market panel (OHLCV + factors).
    
    Uses Merton Jump-Diffusion under correlated Brownian increments with dt = 1/252.
    
    Args:
        assets: List of asset symbols (defaults to DEFAULT_ASSETS: 24 assets).
        n_days: Number of trading days (default 1260 ~ 5 years).
        seed: Random seed for deterministic reproducibility.
        start_date: Starting trading calendar date string.
        
    Returns:
        Dict of DataFrames:
          'close': DataFrame (dates x assets)
          'open': DataFrame (dates x assets)
          'high': DataFrame (dates x assets)
          'low': DataFrame (dates x assets)
          'volume': DataFrame (dates x assets)
          'returns': DataFrame (dates x assets)
    """
    if assets is None:
        assets = list(DEFAULT_ASSETS)
    n_assets = len(assets)
    
    rng = np.random.default_rng(seed)
    drifts, vols, corr = build_calibrated_covariance(assets)
    
    # Cholesky factor for correlated standard normal shocks
    L = np.linalg.cholesky(corr)
    
    dt = 1.0 / 252.0
    sqrt_dt = np.sqrt(dt)
    
    # Correlated Brownian increments
    z_uncorr = rng.standard_normal((n_days, n_assets))
    z_corr = z_uncorr @ L.T
    
    # Merton Jump-Diffusion parameters:
    # lambda_jump = expected jumps per year (e.g. 3 jumps/year)
    # jump_mean = -0.02 (slightly negative skew), jump_std = 0.04
    lambda_jump = 3.0
    jump_mean = -0.015
    jump_std = 0.035
    jump_drift_comp = lambda_jump * (np.exp(jump_mean + 0.5 * jump_std**2) - 1.0)
    
    # Simulate jump counts per day (Poisson)
    jump_prob = lambda_jump * dt
    jump_occurred = rng.uniform(0, 1, size=(n_days, n_assets)) < jump_prob
    jump_sizes = rng.normal(jump_mean, jump_std, size=(n_days, n_assets)) * jump_occurred
    
    # Daily log returns: (mu - jump_comp - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z + Jumps
    drift_term = (drifts - jump_drift_comp - 0.5 * vols**2) * dt
    diffusion_term = vols * sqrt_dt * z_corr
    log_returns = drift_term + diffusion_term + jump_sizes
    
    # Convert to price levels starting from realistic bases
    initial_prices = np.array([
        100.0 if ASSET_CLASSES.get(a) != "CRYPTO" else (10000.0 if a == "BTC" else 500.0)
        for a in assets
    ])
    
    price_paths = np.zeros((n_days + 1, n_assets))
    price_paths[0] = initial_prices
    for t in range(n_days):
        price_paths[t + 1] = price_paths[t] * np.exp(log_returns[t])
        
    close_prices = price_paths[1:]
    
    # Generate intraday Open, High, Low and Volume
    dates = pd.date_range(start=start_date, periods=n_days, freq="B")
    
    # Intraday spreads and volatility
    intraday_vol = vols * sqrt_dt
    open_prices = np.zeros_like(close_prices)
    high_prices = np.zeros_like(close_prices)
    low_prices = np.zeros_like(close_prices)
    volumes = np.zeros_like(close_prices)
    
    # Previous close for open generation
    prev_close = np.vstack([initial_prices, close_prices[:-1]])
    
    for t in range(n_days):
        # Open price fluctuates near previous close
        open_noise = rng.normal(0, intraday_vol * 0.25, size=n_assets)
        open_t = prev_close[t] * np.exp(open_noise)
        close_t = close_prices[t]
        
        # High and Low bounds
        span = np.maximum(open_t, close_t)
        base_low = np.minimum(open_t, close_t)
        
        high_bump = np.abs(rng.normal(0, intraday_vol * 0.5, size=n_assets))
        low_bump = np.abs(rng.normal(0, intraday_vol * 0.5, size=n_assets))
        
        high_prices[t] = span * (1.0 + high_bump)
        low_prices[t] = base_low * np.maximum(0.1, 1.0 - low_bump)
        open_prices[t] = open_t
        
        # Volume: lognormal with surges correlated with absolute returns
        abs_ret = np.abs(log_returns[t])
        vol_surge = 1.0 + 3.0 * (abs_ret / vols)
        base_vol = 1_000_000.0 * (vols / 0.20)
        vol_noise = rng.lognormal(mean=0.0, sigma=0.4, size=n_assets)
        volumes[t] = base_vol * vol_surge * vol_noise
        
    df_close = pd.DataFrame(close_prices, index=dates, columns=assets)
    df_open = pd.DataFrame(open_prices, index=dates, columns=assets)
    df_high = pd.DataFrame(high_prices, index=dates, columns=assets)
    df_low = pd.DataFrame(low_prices, index=dates, columns=assets)
    df_vol = pd.DataFrame(volumes, index=dates, columns=assets)
    df_ret = pd.DataFrame(np.expm1(log_returns), index=dates, columns=assets)
    
    return {
        "close": df_close,
        "open": df_open,
        "high": df_high,
        "low": df_low,
        "volume": df_vol,
        "returns": df_ret,
    }
