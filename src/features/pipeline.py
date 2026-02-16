"""Unified feature engineering pipeline.

This module orchestrates the complete feature computation process:
1. Load raw data
2. Compute crypto features
3. Compute stock features
4. Compute technical features
5. Detect market regimes
6. Create regime interactions
7. Handle outliers and normalize
8. Save feature matrix
"""

import logging
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from scipy import stats

from src.config import get_config
from src.features.crypto_features import compute_all_crypto_features, handle_missing_data
from src.features.stock_features import compute_all_stock_features, handle_missing_fundamentals
from src.features.technical_features import compute_all_technical_features
from src.features.regime import RegimeDetector, create_regime_interactions, prepare_regime_data

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Unified feature engineering pipeline."""
    
    def __init__(self):
        """Initialize feature pipeline."""
        self.config = get_config()
        self.regime_detector = None
        
    def load_raw_data(self) -> dict:
        """Load all raw data from Parquet files.
        
        Returns:
            Dictionary with loaded DataFrames
        """
        logger.info("Loading raw data")
        
        data = {}
        raw_dir = self.config.raw_data_dir
        
        # Load crypto data
        crypto_dir = raw_dir / "coingecko"
        if crypto_dir.exists():
            crypto_files = list(crypto_dir.glob("*.parquet"))
            if crypto_files:
                dfs = [pd.read_parquet(f) for f in crypto_files]
                data['crypto_prices'] = pd.concat(dfs, ignore_index=True)
                logger.info(f"Loaded crypto data: {len(data['crypto_prices'])} records")
        
        # Load stock data
        stock_dir = raw_dir / "yfinance"
        if stock_dir.exists():
            stock_files = list(stock_dir.glob("*.parquet"))
            if stock_files:
                dfs = [pd.read_parquet(f) for f in stock_files]
                data['stock_prices'] = pd.concat(dfs, ignore_index=True)
                logger.info(f"Loaded stock data: {len(data['stock_prices'])} records")
        
        # Load GitHub data
        github_dir = raw_dir / "github"
        if github_dir.exists():
            github_files = list(github_dir.glob("*.parquet"))
            if github_files:
                dfs = [pd.read_parquet(f) for f in github_files]
                data['github'] = pd.concat(dfs, ignore_index=True)
                logger.info(f"Loaded GitHub data: {len(data['github'])} records")
        
        return data
        
    def compute_crypto_features(self, data: dict) -> pd.DataFrame:
        """Compute cryptocurrency features.
        
        Args:
            data: Dictionary with raw data
            
        Returns:
            DataFrame with crypto features
        """
        logger.info("Computing cryptocurrency features")
        
        if 'crypto_prices' not in data:
            logger.warning("No crypto price data available")
            return pd.DataFrame()
        
        crypto_prices = data['crypto_prices']
        
        # Rename coin_id to asset for consistency
        if 'coin_id' in crypto_prices.columns:
            crypto_prices = crypto_prices.rename(columns={'coin_id': 'asset'})
        
        # Compute features
        features = compute_all_crypto_features(
            prices=crypto_prices,
            github_data=data.get('github')
        )
        
        # Handle missing data
        features = handle_missing_data(features, self.config.forward_fill_days)
        
        # Add asset type
        features['asset_type'] = 'crypto'
        
        logger.info(f"Computed crypto features: {features.shape}")
        return features
        
    def compute_stock_features(self, data: dict) -> pd.DataFrame:
        """Compute stock features.
        
        Args:
            data: Dictionary with raw data
            
        Returns:
            DataFrame with stock features
        """
        logger.info("Computing stock features")
        
        if 'stock_prices' not in data:
            logger.warning("No stock price data available")
            return pd.DataFrame()
        
        stock_prices = data['stock_prices']
        
        # Rename ticker to asset for consistency
        stock_prices_renamed = stock_prices.rename(columns={'ticker': 'asset'})
        
        # Compute features
        features = compute_all_stock_features(fundamentals=stock_prices)
        
        # Handle missing data
        features = handle_missing_fundamentals(features, method='forward_fill')
        
        # Rename back to asset
        if 'ticker' in features.columns:
            features = features.rename(columns={'ticker': 'asset'})
        
        # Add asset type
        features['asset_type'] = 'stock'
        
        logger.info(f"Computed stock features: {features.shape}")
        return features
        
    def compute_technical_features(self, data: dict) -> dict:
        """Compute technical features for both crypto and stocks.
        
        Args:
            data: Dictionary with raw data
            
        Returns:
            Dictionary with technical features for crypto and stocks
        """
        logger.info("Computing technical features")
        
        tech_features = {}
        
        # Crypto technical features
        if 'crypto_prices' in data:
            crypto_prices = data['crypto_prices'].copy()
            if 'coin_id' in crypto_prices.columns:
                crypto_prices = crypto_prices.rename(columns={'coin_id': 'asset'})
            
            tech_features['crypto'] = compute_all_technical_features(crypto_prices)
            logger.info(f"Computed crypto technical features: {tech_features['crypto'].shape}")
        
        # Stock technical features
        if 'stock_prices' in data:
            stock_prices = data['stock_prices'].copy()
            if 'ticker' in stock_prices.columns:
                stock_prices = stock_prices.rename(columns={'ticker': 'asset'})
            
            tech_features['stock'] = compute_all_technical_features(stock_prices)
            logger.info(f"Computed stock technical features: {tech_features['stock'].shape}")
        
        return tech_features
        
    def detect_regimes(self, data: dict) -> pd.Series:
        """Detect market regimes using BTC and VIX.
        
        Args:
            data: Dictionary with raw data
            
        Returns:
            Series with regime labels
        """
        logger.info("Detecting market regimes")
        
        if 'crypto_prices' not in data:
            logger.warning("No crypto data for regime detection")
            return pd.Series()
        
        # Get BTC prices
        crypto_prices = data['crypto_prices']
        btc_prices = crypto_prices[crypto_prices['coin_id'] == 'bitcoin'][['timestamp', 'close']].copy()
        
        if btc_prices.empty:
            logger.warning("No Bitcoin data for regime detection")
            return pd.Series()
        
        # Prepare data
        btc_returns, vix_proxy = prepare_regime_data(btc_prices)
        
        # Train regime detector
        self.regime_detector = RegimeDetector(n_states=self.config.regime_n_states)
        self.regime_detector.fit(btc_returns, vix_proxy)
        
        # Predict regimes
        regimes = self.regime_detector.predict_regime(btc_returns, vix_proxy)
        
        logger.info(f"Detected regimes: {regimes.value_counts().to_dict()}")
        return regimes
        
    def winsorize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using winsorization.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with winsorized features
        """
        logger.info("Winsorizing outliers")
        
        percentiles = self.config.winsorize_percentiles
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['timestamp', 'asset_type']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        for col in numeric_cols:
            if df[col].notna().sum() > 0:
                lower = np.percentile(df[col].dropna(), percentiles[0])
                upper = np.percentile(df[col].dropna(), percentiles[1])
                df[col] = df[col].clip(lower, upper)
        
        logger.info(f"Winsorized {len(numeric_cols)} features")
        return df
        
    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features using z-score within asset groups.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with normalized features
        """
        logger.info("Normalizing features (z-score)")
        
        window_days = self.config.zscore_window_days
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['timestamp', 'asset_type']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        df = df.sort_values(['asset', 'timestamp']).copy()
        
        for col in numeric_cols:
            # Compute rolling z-score within each asset
            df[f'{col}_zscore'] = df.groupby('asset')[col].transform(
                lambda x: (x - x.rolling(window_days, min_periods=30).mean()) / 
                         x.rolling(window_days, min_periods=30).std()
            )
        
        logger.info(f"Normalized {len(numeric_cols)} features")
        return df
        
    def run_pipeline(self, save_output: bool = True) -> pd.DataFrame:
        """Run complete feature engineering pipeline.
        
        Args:
            save_output: Whether to save feature matrix to disk
            
        Returns:
            Complete feature matrix
        """
        logger.info("=" * 60)
        logger.info("STARTING FEATURE ENGINEERING PIPELINE")
        logger.info("=" * 60)
        
        # Load raw data
        data = self.load_raw_data()
        
        if not data:
            logger.error("No raw data available. Run data fetching first.")
            return pd.DataFrame()
        
        # Compute technical features first (needed for both asset types)
        tech_features = self.compute_technical_features(data)
        
        # Compute crypto features
        crypto_features = self.compute_crypto_features(data)
        
        # Compute stock features
        stock_features = self.compute_stock_features(data)
        
        # Merge technical features
        if not crypto_features.empty and 'crypto' in tech_features:
            crypto_features = pd.merge(
                crypto_features,
                tech_features['crypto'],
                on=['timestamp', 'asset'],
                how='left'
            )
        
        if not stock_features.empty and 'stock' in tech_features:
            stock_features = pd.merge(
                stock_features,
                tech_features['stock'],
                on=['timestamp', 'asset'],
                how='left'
            )
        
        # Combine crypto and stock features
        all_features = []
        if not crypto_features.empty:
            all_features.append(crypto_features)
        if not stock_features.empty:
            all_features.append(stock_features)
        
        if not all_features:
            logger.error("No features computed")
            return pd.DataFrame()
        
        features = pd.concat(all_features, ignore_index=True)
        logger.info(f"Combined features: {features.shape}")
        
        # Detect regimes
        regimes = self.detect_regimes(data)
        
        if not regimes.empty:
            # Broadcast regimes to daily data
            regime_df = regimes.reset_index()
            regime_df.columns = ['timestamp', 'regime']
            
            # Merge with forward-fill
            features = features.sort_values('timestamp')
            features = pd.merge_asof(
                features,
                regime_df,
                on='timestamp',
                direction='backward'
            )
            
            # Create regime interactions
            features = create_regime_interactions(features, features['regime'])
        
        # Handle outliers
        features = self.winsorize_features(features)
        
        # Normalize features
        features = self.normalize_features(features)
        
        # Save to disk
        if save_output:
            output_dir = self.config.features_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / "features.parquet"
            features.to_parquet(output_path, index=False)
            logger.info(f"Saved features to {output_path}")
        
        logger.info("=" * 60)
        logger.info("FEATURE ENGINEERING PIPELINE COMPLETE")
        logger.info(f"Final feature matrix: {features.shape}")
        logger.info("=" * 60)
        
        return features


def main():
    """Main function for running feature pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    pipeline = FeaturePipeline()
    features = pipeline.run_pipeline(save_output=True)
    
    if not features.empty:
        logger.info(f"\nFeature columns: {features.columns.tolist()}")
        logger.info(f"\nSample features:\n{features.head()}")
        
        # Summary statistics
        logger.info(f"\nAsset types: {features['asset_type'].value_counts().to_dict()}")
        logger.info(f"Date range: {features['timestamp'].min()} to {features['timestamp'].max()}")
        logger.info(f"Unique assets: {features['asset'].nunique()}")


if __name__ == "__main__":
    main()
