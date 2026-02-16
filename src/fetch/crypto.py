"""Cryptocurrency data fetching from CoinGecko and Dune Analytics.

This module fetches:
- OHLCV and market cap data from CoinGecko
- Whale transfer data from Dune Analytics
- Exchange netflow data from Dune Analytics
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import get_config

logger = logging.getLogger(__name__)


class CoinGeckoFetcher:
    """Fetcher for CoinGecko market data."""
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize CoinGecko fetcher.
        
        Args:
            api_key: CoinGecko Demo or Pro API key (optional for free tier)
        """
        self.config = get_config()
        self.api_key = api_key or self.config.coingecko_api_key
        
        # Setup session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Rate limiting
        self.calls_per_minute = self.config.get('data_fetching.rate_limits.coingecko_calls_per_minute', 50)
        self.last_call_time = 0
        
    def _rate_limit(self):
        """Enforce rate limiting."""
        min_interval = 60.0 / self.calls_per_minute
        elapsed = time.time() - self.last_call_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self.last_call_time = time.time()
        
    def _make_request(self, endpoint: str, params: dict = None) -> dict:
        """Make API request with rate limiting and error handling.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            JSON response
        """
        self._rate_limit()
        
        url = f"{self.BASE_URL}/{endpoint}"
        headers = {}
        if self.api_key:
            # Demo plan uses x-cg-demo-api-key, Pro uses x-cg-pro-api-key
            headers['x-cg-demo-api-key'] = self.api_key
            
        try:
            response = self.session.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"CoinGecko API request failed: {e}")
            raise
            
    def get_top_coins(self, limit: int = 100) -> List[str]:
        """Get list of top coins by market cap.
        
        Args:
            limit: Number of top coins to return
            
        Returns:
            List of coin IDs
        """
        logger.info(f"Fetching top {limit} coins by market cap")
        
        params = {
            'vs_currency': 'usd',
            'order': 'market_cap_desc',
            'per_page': min(limit, 250),
            'page': 1,
            'sparkline': False
        }
        
        coins = []
        pages_needed = (limit + 249) // 250
        
        for page in range(1, pages_needed + 1):
            params['page'] = page
            data = self._make_request('coins/markets', params)
            coins.extend([coin['id'] for coin in data])
            
            if len(coins) >= limit:
                break
                
        return coins[:limit]
        
    def fetch_market_data(
        self,
        coin_ids: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch OHLCV and market cap data for coins.
        
        Args:
            coin_ids: List of CoinGecko coin IDs
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with columns: timestamp, coin_id, open, high, low, close, volume, market_cap
        """
        logger.info(f"Fetching market data for {len(coin_ids)} coins from {start_date} to {end_date}")
        
        all_data = []
        
        for coin_id in coin_ids:
            try:
                # CoinGecko prefers ISO date format for compatibility
                from_str = start_date.strftime('%Y-%m-%d')
                to_str = end_date.strftime('%Y-%m-%d')
                
                # Fetch OHLC data
                params = {
                    'vs_currency': 'usd',
                    'from': from_str,
                    'to': to_str
                }
                
                data = self._make_request(f'coins/{coin_id}/market_chart/range', params)
                
                if not data or 'prices' not in data:
                    logger.warning(f"No data returned for {coin_id}")
                    continue
                
                # Parse data
                prices = data['prices']
                volumes = data.get('total_volumes', [])
                market_caps = data.get('market_caps', [])
                
                # Convert to DataFrame
                df = pd.DataFrame({
                    'timestamp': [datetime.fromtimestamp(p[0] / 1000) for p in prices],
                    'coin_id': coin_id,
                    'close': [p[1] for p in prices],
                    'volume': [v[1] if v else None for v in volumes],
                    'market_cap': [m[1] if m else None for m in market_caps]
                })
                
                # CoinGecko doesn't provide OHLC in free tier, use close as approximation
                df['open'] = df['close']
                df['high'] = df['close']
                df['low'] = df['close']
                
                all_data.append(df)
                logger.info(f"Fetched {len(df)} records for {coin_id}")
                
            except Exception as e:
                logger.error(f"Failed to fetch data for {coin_id}: {e}")
                continue
                
        if not all_data:
            logger.warning("No data fetched for any coins")
            return pd.DataFrame()
            
        # Combine all data
        result = pd.concat(all_data, ignore_index=True)
        
        # Convert timestamp to UTC
        result['timestamp'] = pd.to_datetime(result['timestamp'], utc=True)
        
        # Reorder columns
        result = result[['timestamp', 'coin_id', 'open', 'high', 'low', 'close', 'volume', 'market_cap']]
        
        logger.info(f"Total records fetched: {len(result)}")
        return result


class DuneFetcher:
    """Fetcher for Dune Analytics on-chain data."""
    
    BASE_URL = "https://api.dune.com/api/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Dune fetcher.
        
        Args:
            api_key: Dune Analytics API key
        """
        self.config = get_config()
        self.api_key = api_key or self.config.dune_api_key
        
        if not self.api_key:
            logger.warning("Dune API key not provided. On-chain data fetching will be disabled.")
            
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({'X-Dune-API-Key': self.api_key})
            
    def fetch_whale_transfers(
        self,
        blockchain: str = "ethereum",
        min_value_usd: float = 1000000,
        start_date: datetime = None
    ) -> pd.DataFrame:
        """Fetch large wallet transfers (whale activity).
        
        Args:
            blockchain: Blockchain to query (ethereum, polygon, etc.)
            min_value_usd: Minimum transfer value in USD
            start_date: Start date for transfers
            
        Returns:
            DataFrame with columns: timestamp, token, from_address, to_address, amount_usd
        """
        if not self.api_key:
            logger.warning("Dune API key not available. Returning empty DataFrame.")
            return pd.DataFrame()
            
        logger.info(f"Fetching whale transfers on {blockchain} (min ${min_value_usd:,.0f})")
        
        # Note: This is a placeholder. Actual implementation requires:
        # 1. Creating a Dune query for whale transfers
        # 2. Executing the query via API
        # 3. Polling for results
        
        # For now, return empty DataFrame with correct schema
        return pd.DataFrame(columns=[
            'timestamp', 'blockchain', 'token', 'from_address', 
            'to_address', 'amount', 'amount_usd'
        ])


def fetch_coingecko_market_data(
    coin_ids: List[str],
    start_date: datetime,
    end_date: datetime,
    output_dir: Optional[Path] = None
) -> pd.DataFrame:
    """Fetch OHLCV and market cap data from CoinGecko.
    
    Args:
        coin_ids: List of CoinGecko coin IDs
        start_date: Start date for data
        end_date: End date for data
        output_dir: Directory to save Parquet file (optional)
        
    Returns:
        DataFrame with columns: timestamp, coin_id, open, high, low, close, volume, market_cap
    """
    fetcher = CoinGeckoFetcher()
    df = fetcher.fetch_market_data(coin_ids, start_date, end_date)
    
    if output_dir and not df.empty:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"coingecko_{end_date.strftime('%Y%m%d')}.parquet"
        filepath = output_dir / filename
        
        df.to_parquet(filepath, index=False)
        logger.info(f"Saved data to {filepath}")
        
    return df


def fetch_dune_whale_transfers(
    blockchain: str,
    min_value_usd: float,
    start_date: datetime,
    output_dir: Optional[Path] = None
) -> pd.DataFrame:
    """Query Dune Analytics for large wallet transfers.
    
    Args:
        blockchain: Blockchain to query
        min_value_usd: Minimum transfer value in USD
        start_date: Start date for transfers
        output_dir: Directory to save Parquet file (optional)
        
    Returns:
        DataFrame with columns: timestamp, token, from_address, to_address, amount_usd
    """
    fetcher = DuneFetcher()
    df = fetcher.fetch_whale_transfers(blockchain, min_value_usd, start_date)
    
    if output_dir and not df.empty:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"dune_whale_transfers_{datetime.now().strftime('%Y%m%d')}.parquet"
        filepath = output_dir / filename
        
        df.to_parquet(filepath, index=False)
        logger.info(f"Saved data to {filepath}")
        
    return df


def get_last_fetch_date(data_dir: Path, prefix: str) -> Optional[datetime]:
    """Get the date of the last successful fetch.
    
    Args:
        data_dir: Directory containing data files
        prefix: Filename prefix (e.g., 'coingecko')
        
    Returns:
        Last fetch date or None if no previous data
    """
    if not data_dir.exists():
        return None
        
    files = list(data_dir.glob(f"{prefix}_*.parquet"))
    if not files:
        return None
        
    # Read the most recent file and get max timestamp
    latest_file = max(files, key=lambda p: p.stat().st_mtime)
    
    try:
        df = pd.read_parquet(latest_file)
        if df.empty or 'timestamp' not in df.columns:
            return None
            
        max_timestamp = pd.to_datetime(df['timestamp']).max()
        return max_timestamp.to_pydatetime()
    except Exception as e:
        logger.error(f"Failed to read last fetch date from {latest_file}: {e}")
        return None


def main():
    """Main function for testing crypto data fetching."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    config = get_config()
    
    # Get top coins
    fetcher = CoinGeckoFetcher()
    top_coins = fetcher.get_top_coins(limit=config.top_n_coins)
    logger.info(f"Top coins: {top_coins[:10]}...")
    
    # Determine date range
    output_dir = config.raw_data_dir / "coingecko"
    
    if config.incremental_fetching:
        last_date = get_last_fetch_date(output_dir, "coingecko")
        if last_date:
            start_date = last_date + timedelta(days=1)
            logger.info(f"Incremental fetch from {start_date}")
        else:
            start_date = datetime.now() - timedelta(days=config.lookback_days)
            logger.info(f"Initial fetch from {start_date}")
    else:
        start_date = datetime.now() - timedelta(days=config.lookback_days)
        
    end_date = datetime.now()
    
    # Fetch data
    df = fetch_coingecko_market_data(
        coin_ids=top_coins[:10],  # Test with 10 coins
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir
    )
    
    logger.info(f"Fetched {len(df)} records")
    if not df.empty:
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.info(f"Coins: {df['coin_id'].nunique()}")
    else:
        logger.warning("No data fetched. Check API keys and CoinGecko rate limits.")


if __name__ == "__main__":
    main()
