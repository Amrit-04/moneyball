"""Stock data fetching from yfinance and SEC EDGAR/BSE-NSE.

This module fetches:
- Price and fundamental data from yfinance (US or Indian markets)
- Insider trading data from SEC EDGAR (US) or BSE/NSE (India)
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import pandas as pd
import yfinance as yf
import requests

from src.config import get_config

logger = logging.getLogger(__name__)


class MarketSelector:
    """Selects best stock market (US or India) based on data availability."""
    
    def __init__(self):
        """Initialize market selector."""
        self.config = get_config()
        
    def test_market_availability(self, market: str) -> Tuple[bool, int, str]:
        """Test if market data is available and assess quality.
        
        Args:
            market: Market code ('US' or 'IN')
            
        Returns:
            Tuple of (is_available, num_assets_found, error_message)
        """
        logger.info(f"Testing {market} market availability...")
        
        # Test tickers for each market
        test_tickers = {
            'US': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
            'IN': ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']
        }
        
        if market not in test_tickers:
            return False, 0, f"Unknown market: {market}"
            
        tickers = test_tickers[market]
        successful = 0
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1mo")
                
                if not hist.empty and len(hist) >= 10:
                    successful += 1
                    
            except Exception as e:
                logger.debug(f"Failed to fetch {ticker}: {e}")
                continue
                
        is_available = successful >= 3  # At least 3 out of 5 should work
        
        if is_available:
            logger.info(f"{market} market: ✓ Available ({successful}/{len(tickers)} test tickers)")
        else:
            logger.warning(f"{market} market: ✗ Limited availability ({successful}/{len(tickers)} test tickers)")
            
        return is_available, successful, ""
        
    def select_market(self) -> str:
        """Select best market based on configuration and availability.
        
        Returns:
            Market code ('US' or 'IN')
        """
        configured_market = self.config.stock_market
        
        if configured_market in ['US', 'IN']:
            logger.info(f"Using configured market: {configured_market}")
            return configured_market
            
        # Auto-select based on availability
        logger.info("Auto-selecting market based on data availability...")
        
        us_available, us_count, _ = self.test_market_availability('US')
        in_available, in_count, _ = self.test_market_availability('IN')
        
        if us_available and not in_available:
            logger.info("Selected US market")
            return 'US'
        elif in_available and not us_available:
            logger.info("Selected Indian market")
            return 'IN'
        elif us_available and in_available:
            # Both available, prefer US for better data quality
            logger.info("Both markets available, preferring US market")
            return 'US'
        else:
            # Neither fully available, use US as fallback
            logger.warning("Limited availability for both markets, defaulting to US")
            return 'US'


class StockFetcher:
    """Fetcher for stock price and fundamental data."""
    
    def __init__(self, market: Optional[str] = None):
        """Initialize stock fetcher.
        
        Args:
            market: Market code ('US' or 'IN'), auto-selected if None
        """
        self.config = get_config()
        
        if market is None:
            selector = MarketSelector()
            market = selector.select_market()
            
        self.market = market
        logger.info(f"Initialized StockFetcher for {self.market} market")
        
    def get_small_cap_tickers(self, limit: int = 200) -> List[str]:
        """Get list of small-cap stock tickers.
        
        Args:
            limit: Maximum number of tickers to return
            
        Returns:
            List of ticker symbols
        """
        logger.info(f"Fetching small-cap tickers for {self.market} market")
        
        if self.market == 'US':
            return self._get_us_small_caps(limit)
        else:
            return self._get_indian_small_caps(limit)
            
    def _get_us_small_caps(self, limit: int) -> List[str]:
        """Get US small-cap tickers.
        
        Note: This is a simplified implementation. In production, you'd want to:
        - Use a proper stock screener API
        - Filter by market cap range ($50M - $2B)
        - Update the list periodically
        """
        # Placeholder: Common small-cap tickers
        # In production, fetch from a screener API or maintain a curated list
        small_caps = [
            'CRWD', 'DDOG', 'NET', 'SNOW', 'ZS', 'OKTA', 'ESTC', 'MDB',
            'TEAM', 'WDAY', 'NOW', 'VEEV', 'ZM', 'DOCU', 'TWLO', 'SHOP',
            'SQ', 'PYPL', 'ROKU', 'PINS', 'SNAP', 'UBER', 'LYFT', 'DASH',
            'ABNB', 'COIN', 'HOOD', 'SOFI', 'AFRM', 'UPST', 'LC', 'OPEN'
        ]
        
        return small_caps[:limit]
        
    def _get_indian_small_caps(self, limit: int) -> List[str]:
        """Get Indian small-cap tickers (NSE).
        
        Note: Indian tickers need .NS or .BO suffix for yfinance
        """
        # Placeholder: Common Indian small-cap tickers
        small_caps = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
            'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
            'LT.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'TITAN.NS',
            'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'BAJFINANCE.NS', 'NESTLEIND.NS', 'WIPRO.NS'
        ]
        
        return small_caps[:limit]
        
    def fetch_stock_data(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch stock price and fundamental data.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with columns: timestamp, ticker, market, open, high, low, close, 
                                   volume, revenue, gross_profit, total_assets, operating_income
        """
        logger.info(f"Fetching data for {len(tickers)} stocks from {start_date} to {end_date}")
        
        all_data = []
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                
                # Fetch price data
                hist = stock.history(start=start_date, end=end_date)
                
                if hist.empty:
                    logger.warning(f"No price data for {ticker}")
                    continue
                    
                # Create DataFrame
                df = pd.DataFrame({
                    'timestamp': hist.index,
                    'ticker': ticker,
                    'market': self.market,
                    'open': hist['Open'].values,
                    'high': hist['High'].values,
                    'low': hist['Low'].values,
                    'close': hist['Close'].values,
                    'volume': hist['Volume'].values
                })
                
                # Fetch fundamental data
                try:
                    info = stock.info
                    financials = stock.financials
                    balance_sheet = stock.balance_sheet
                    
                    # Extract key metrics (these will be the same for all rows)
                    revenue = None
                    gross_profit = None
                    total_assets = None
                    operating_income = None
                    
                    if not financials.empty:
                        if 'Total Revenue' in financials.index:
                            revenue = financials.loc['Total Revenue'].iloc[0] if len(financials.columns) > 0 else None
                        if 'Gross Profit' in financials.index:
                            gross_profit = financials.loc['Gross Profit'].iloc[0] if len(financials.columns) > 0 else None
                        if 'Operating Income' in financials.index:
                            operating_income = financials.loc['Operating Income'].iloc[0] if len(financials.columns) > 0 else None
                            
                    if not balance_sheet.empty:
                        if 'Total Assets' in balance_sheet.index:
                            total_assets = balance_sheet.loc['Total Assets'].iloc[0] if len(balance_sheet.columns) > 0 else None
                            
                    df['revenue'] = revenue
                    df['gross_profit'] = gross_profit
                    df['total_assets'] = total_assets
                    df['operating_income'] = operating_income
                    
                except Exception as e:
                    logger.debug(f"Could not fetch fundamentals for {ticker}: {e}")
                    df['revenue'] = None
                    df['gross_profit'] = None
                    df['total_assets'] = None
                    df['operating_income'] = None
                    
                all_data.append(df)
                logger.info(f"Fetched {len(df)} records for {ticker}")
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Failed to fetch data for {ticker}: {e}")
                continue
                
        if not all_data:
            logger.warning("No data fetched for any stocks")
            return pd.DataFrame()
            
        # Combine all data
        result = pd.concat(all_data, ignore_index=True)
        
        # Convert timestamp to UTC
        result['timestamp'] = pd.to_datetime(result['timestamp'], utc=True)
        
        # Add total_liabilities column (placeholder)
        result['total_liabilities'] = None
        
        logger.info(f"Total records fetched: {len(result)}")
        return result


class InsiderTradeFetcher:
    """Fetcher for insider trading data."""
    
    def __init__(self, market: str):
        """Initialize insider trade fetcher.
        
        Args:
            market: Market code ('US' or 'IN')
        """
        self.market = market
        self.config = get_config()
        
    def fetch_insider_trades(
        self,
        tickers: List[str],
        start_date: datetime
    ) -> pd.DataFrame:
        """Fetch insider trading data.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date for trades
            
        Returns:
            DataFrame with columns: timestamp, ticker, insider_name, transaction_type, shares, price
        """
        if self.market == 'US':
            return self._fetch_sec_insider_trades(tickers, start_date)
        else:
            return self._fetch_indian_insider_trades(tickers, start_date)
            
    def _fetch_sec_insider_trades(
        self,
        tickers: List[str],
        start_date: datetime
    ) -> pd.DataFrame:
        """Fetch SEC Form 4 insider trading data.
        
        Note: This is a placeholder. Full implementation would require:
        - Parsing SEC EDGAR XML files
        - Or using a paid API service
        """
        logger.info(f"Fetching SEC insider trades for {len(tickers)} tickers")
        logger.warning("SEC insider trade fetching not fully implemented - returning empty DataFrame")
        
        return pd.DataFrame(columns=[
            'timestamp', 'ticker', 'insider_name', 'transaction_type', 'shares', 'price'
        ])
        
    def _fetch_indian_insider_trades(
        self,
        tickers: List[str],
        start_date: datetime
    ) -> pd.DataFrame:
        """Fetch Indian insider trading data from BSE/NSE.
        
        Note: This is a placeholder. Indian insider data has limited free API access.
        """
        logger.info(f"Fetching Indian insider trades for {len(tickers)} tickers")
        logger.warning("Indian insider trade fetching not fully implemented - returning empty DataFrame")
        
        return pd.DataFrame(columns=[
            'timestamp', 'ticker', 'insider_name', 'transaction_type', 'shares', 'price'
        ])


def fetch_yfinance_data(
    tickers: List[str],
    market: str,
    start_date: datetime,
    end_date: datetime,
    output_dir: Optional[Path] = None
) -> pd.DataFrame:
    """Fetch stock price and fundamental data from yfinance.
    
    Args:
        tickers: List of ticker symbols
        market: Market code ('US' or 'IN')
        start_date: Start date for data
        end_date: End date for data
        output_dir: Directory to save Parquet file (optional)
        
    Returns:
        DataFrame with stock data
    """
    fetcher = StockFetcher(market=market)
    df = fetcher.fetch_stock_data(tickers, start_date, end_date)
    
    if output_dir and not df.empty:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"yfinance_{market}_{end_date.strftime('%Y%m%d')}.parquet"
        filepath = output_dir / filename
        
        df.to_parquet(filepath, index=False)
        logger.info(f"Saved data to {filepath}")
        
    return df


def fetch_sec_insider_trades(
    tickers: List[str],
    start_date: datetime,
    output_dir: Optional[Path] = None
) -> pd.DataFrame:
    """Fetch Form 4 insider trading data from SEC EDGAR (US stocks only).
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date for trades
        output_dir: Directory to save Parquet file (optional)
        
    Returns:
        DataFrame with insider trades
    """
    fetcher = InsiderTradeFetcher(market='US')
    df = fetcher.fetch_insider_trades(tickers, start_date)
    
    if output_dir and not df.empty:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"sec_insider_trades_{datetime.now().strftime('%Y%m%d')}.parquet"
        filepath = output_dir / filename
        
        df.to_parquet(filepath, index=False)
        logger.info(f"Saved data to {filepath}")
        
    return df


def fetch_indian_insider_trades(
    tickers: List[str],
    start_date: datetime,
    output_dir: Optional[Path] = None
) -> pd.DataFrame:
    """Fetch insider trading data from BSE/NSE (Indian stocks only).
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date for trades
        output_dir: Directory to save Parquet file (optional)
        
    Returns:
        DataFrame with insider trades
    """
    fetcher = InsiderTradeFetcher(market='IN')
    df = fetcher.fetch_insider_trades(tickers, start_date)
    
    if output_dir and not df.empty:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"indian_insider_trades_{datetime.now().strftime('%Y%m%d')}.parquet"
        filepath = output_dir / filename
        
        df.to_parquet(filepath, index=False)
        logger.info(f"Saved data to {filepath}")
        
    return df


def main():
    """Main function for testing stock data fetching."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    config = get_config()
    
    # Select market
    selector = MarketSelector()
    market = selector.select_market()
    
    # Get tickers
    fetcher = StockFetcher(market=market)
    tickers = fetcher.get_small_cap_tickers(limit=10)  # Test with 10 stocks
    logger.info(f"Test tickers: {tickers}")
    
    # Fetch data
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    
    output_dir = config.raw_data_dir / "yfinance"
    
    df = fetch_yfinance_data(
        tickers=tickers,
        market=market,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir
    )
    
    logger.info(f"Fetched {len(df)} records")
    if not df.empty:
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.info(f"Tickers: {df['ticker'].nunique()}")


if __name__ == "__main__":
    main()
