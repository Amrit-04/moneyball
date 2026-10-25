"""Market Data Loader and Cache Manager.

Supports:
1. Deterministic synthetic panel generation (24 assets, 1,260 days, 100% reproducible).
2. Local disk caching (Parquet / CSV) for zero-latency offline runs.
3. Live yfinance / crypto pulls with automatic graceful fallback to cache.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np
import pandas as pd

from quant_engine.data.synthetic import (
    DEFAULT_ASSETS,
    generate_deterministic_market_panel,
)
from quant_engine.data.features import compute_factor_panel

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "cache"


class MarketDataLoader:
    """Manages loading, synthetic generation, live download, and caching of market data."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def load_panel(
        self,
        assets: Optional[List[str]] = None,
        n_days: int = 1260,
        seed: int = 42,
        use_cache: bool = True,
        live_pull: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """Load multi-asset OHLCV market panel.
        
        Args:
            assets: List of ticker symbols. Defaults to DEFAULT_ASSETS.
            n_days: Number of trading days (default 1260).
            seed: Random seed for deterministic reproducibility.
            use_cache: If True, check local cache first.
            live_pull: If True, attempt live yfinance download before falling back.
            
        Returns:
            Dict containing 'close', 'open', 'high', 'low', 'volume', 'returns' DataFrames.
        """
        if assets is None:
            assets = list(DEFAULT_ASSETS)
            
        cache_key = f"market_panel_{len(assets)}assets_{n_days}d_s{seed}.parquet"
        cache_path = self.cache_dir / cache_key
        
        if live_pull:
            try:
                logger.info("Attempting live data pull via yfinance...")
                import yfinance as yf
                tickers_str = " ".join([a for a in assets if not a.endswith("-USD") and a not in ["BTC", "ETH", "SOL", "BNB"]])
                # Add USD for crypto tickers if yfinance
                yf_tickers = [f"{a}-USD" if a in ["BTC", "ETH", "SOL", "BNB"] else a for a in assets]
                df_live = yf.download(
                    yf_tickers,
                    period=f"{int(n_days * 1.5)}d",
                    interval="1d",
                    progress=False,
                    group_by="column"
                )
                if not df_live.empty and "Close" in df_live:
                    close_df = df_live["Close"].dropna()
                    if len(close_df) >= 200:
                        logger.info("Live data pull succeeded.")
                        # Format into standard dictionary
                        panel = {
                            "close": close_df.iloc[-n_days:],
                            "open": df_live["Open"].iloc[-n_days:] if "Open" in df_live else close_df.iloc[-n_days:],
                            "high": df_live["High"].iloc[-n_days:] if "High" in df_live else close_df.iloc[-n_days:],
                            "low": df_live["Low"].iloc[-n_days:] if "Low" in df_live else close_df.iloc[-n_days:],
                            "volume": df_live["Volume"].iloc[-n_days:] if "Volume" in df_live else close_df.iloc[-n_days:] * 1000,
                            "returns": close_df.pct_change().iloc[-n_days:],
                        }
                        return panel
            except Exception as e:
                logger.warning(f"Live yfinance pull failed ({e}). Falling back to deterministic pipeline.")
                
        # Cache check
        if use_cache and cache_path.exists():
            try:
                logger.info(f"Loading cached market panel from {cache_path}")
                panel = {}
                for key in ["close", "open", "high", "low", "volume", "returns"]:
                    p_file = self.cache_dir / f"{key}_{cache_key}"
                    if p_file.exists():
                        panel[key] = pd.read_parquet(p_file)
                if len(panel) == 6:
                    return panel
            except Exception as e:
                logger.warning(f"Failed to read cache: {e}. Regenerating...")
                
        # Generate deterministic panel
        logger.info(f"Generating deterministic panel for {len(assets)} assets over {n_days} days (seed={seed})")
        panel = generate_deterministic_market_panel(assets=assets, n_days=n_days, seed=seed)
        
        # Save cache
        if use_cache:
            try:
                for key, df in panel.items():
                    p_file = self.cache_dir / f"{key}_{cache_key}"
                    df.to_parquet(p_file)
                logger.info(f"Cached market panel to {self.cache_dir}")
            except Exception as e:
                logger.warning(f"Failed to write cache: {e}")
                
        return panel

    def load_feature_panel(
        self,
        assets: Optional[List[str]] = None,
        n_days: int = 1260,
        seed: int = 42,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Load or generate the factor feature panel with MultiIndex (date, asset)."""
        cache_key = f"factor_panel_{len(assets or DEFAULT_ASSETS)}assets_{n_days}d_s{seed}.parquet"
        cache_path = self.cache_dir / cache_key
        
        if use_cache and cache_path.exists():
            try:
                logger.info(f"Loading factor panel from cache: {cache_path}")
                return pd.read_parquet(cache_path)
            except Exception as e:
                logger.warning(f"Failed to read cached factor panel: {e}")
                
        panel = self.load_panel(assets=assets, n_days=n_days, seed=seed, use_cache=use_cache)
        factor_df = compute_factor_panel(panel, seed=seed)
        
        if use_cache:
            try:
                factor_df.to_parquet(cache_path)
                logger.info(f"Saved factor panel cache to {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save factor panel cache: {e}")
                
        return factor_df
