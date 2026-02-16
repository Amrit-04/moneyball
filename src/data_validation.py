"""Data validation and quality checks for Moneyball system.

This module validates:
- Data completeness (minimum history, asset counts)
- Timestamp formats and consistency
- Data quality metrics
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

from src.config import get_config

logger = logging.getLogger(__name__)


class DataValidator:
    """Validator for data completeness and quality."""
    
    def __init__(self):
        """Initialize data validator."""
        self.config = get_config()
        self.validation_results = {}
        
    def validate_all(self) -> Dict[str, any]:
        """Run all validation checks.
        
        Returns:
            Dictionary with validation results
        """
        logger.info("Starting data validation...")
        
        results = {
            'timestamp': datetime.now(),
            'crypto_validation': self.validate_crypto_data(),
            'stock_validation': self.validate_stock_data(),
            'overall_status': 'pending'
        }
        
        # Determine overall status
        crypto_ok = results['crypto_validation']['status'] == 'pass'
        stock_ok = results['stock_validation']['status'] == 'pass'
        
        if crypto_ok and stock_ok:
            results['overall_status'] = 'pass'
            logger.info("All validation checks passed")
        elif crypto_ok or stock_ok:
            results['overall_status'] = 'partial'
            logger.warning("Some validation checks failed")
        else:
            results['overall_status'] = 'fail'
            logger.error("All validation checks failed")
            
        self.validation_results = results
        return results
        
    def validate_crypto_data(self) -> Dict[str, any]:
        """Validate cryptocurrency data.
        
        Returns:
            Dictionary with crypto validation results
        """
        logger.info("Validating cryptocurrency data...")
        
        crypto_dir = self.config.raw_data_dir / "coingecko"
        
        if not crypto_dir.exists():
            logger.warning(f"Crypto data directory not found: {crypto_dir}")
            return {
                'status': 'fail',
                'reason': 'Data directory not found',
                'asset_count': 0,
                'date_range_days': 0,
                'issues': ['Directory does not exist']
            }
            
        # Find all parquet files
        parquet_files = list(crypto_dir.glob("*.parquet"))
        
        if not parquet_files:
            logger.warning("No crypto data files found")
            return {
                'status': 'fail',
                'reason': 'No data files found',
                'asset_count': 0,
                'date_range_days': 0,
                'issues': ['No parquet files in directory']
            }
            
        # Load and analyze data
        try:
            dfs = [pd.read_parquet(f) for f in parquet_files]
            df = pd.concat(dfs, ignore_index=True)
            
            # Validate timestamps
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            
            # Calculate metrics
            asset_count = df['coin_id'].nunique()
            min_date = df['timestamp'].min()
            max_date = df['timestamp'].max()
            date_range_days = (max_date - min_date).days
            
            # Check thresholds
            min_assets = self.config.min_crypto_assets
            min_months = self.config.min_history_months
            min_days = min_months * 30
            
            issues = []
            
            if asset_count < min_assets:
                issues.append(f"Insufficient assets: {asset_count} < {min_assets}")
                
            if date_range_days < min_days:
                issues.append(f"Insufficient history: {date_range_days} days < {min_days} days")
                
            # Check for missing data
            missing_assets = df[df['close'].isna()]['coin_id'].unique()
            if len(missing_assets) > 0:
                issues.append(f"Missing price data for {len(missing_assets)} assets")
                
            status = 'pass' if not issues else 'fail'
            
            result = {
                'status': status,
                'asset_count': asset_count,
                'date_range_days': date_range_days,
                'min_date': min_date.isoformat(),
                'max_date': max_date.isoformat(),
                'total_records': len(df),
                'issues': issues
            }
            
            if status == 'pass':
                logger.info(f"Crypto validation passed: {asset_count} assets, {date_range_days} days")
            else:
                logger.warning(f"Crypto validation failed: {', '.join(issues)}")
                
            return result
            
        except Exception as e:
            logger.error(f"Failed to validate crypto data: {e}")
            return {
                'status': 'fail',
                'reason': f'Validation error: {str(e)}',
                'asset_count': 0,
                'date_range_days': 0,
                'issues': [str(e)]
            }
            
    def validate_stock_data(self) -> Dict[str, any]:
        """Validate stock data.
        
        Returns:
            Dictionary with stock validation results
        """
        logger.info("Validating stock data...")
        
        stock_dir = self.config.raw_data_dir / "yfinance"
        
        if not stock_dir.exists():
            logger.warning(f"Stock data directory not found: {stock_dir}")
            return {
                'status': 'fail',
                'reason': 'Data directory not found',
                'asset_count': 0,
                'date_range_days': 0,
                'issues': ['Directory does not exist']
            }
            
        # Find all parquet files
        parquet_files = list(stock_dir.glob("*.parquet"))
        
        if not parquet_files:
            logger.warning("No stock data files found")
            return {
                'status': 'fail',
                'reason': 'No data files found',
                'asset_count': 0,
                'date_range_days': 0,
                'issues': ['No parquet files in directory']
            }
            
        # Load and analyze data
        try:
            dfs = [pd.read_parquet(f) for f in parquet_files]
            df = pd.concat(dfs, ignore_index=True)
            
            # Validate timestamps
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            
            # Calculate metrics
            asset_count = df['ticker'].nunique()
            min_date = df['timestamp'].min()
            max_date = df['timestamp'].max()
            date_range_days = (max_date - min_date).days
            
            # Check thresholds
            min_assets = self.config.min_stock_assets
            min_months = self.config.min_history_months
            min_days = min_months * 30
            
            issues = []
            
            if asset_count < min_assets:
                issues.append(f"Insufficient assets: {asset_count} < {min_assets}")
                
            if date_range_days < min_days:
                issues.append(f"Insufficient history: {date_range_days} days < {min_days} days")
                
            # Check for missing data
            missing_assets = df[df['close'].isna()]['ticker'].unique()
            if len(missing_assets) > 0:
                issues.append(f"Missing price data for {len(missing_assets)} assets")
                
            status = 'pass' if not issues else 'fail'
            
            result = {
                'status': status,
                'asset_count': asset_count,
                'date_range_days': date_range_days,
                'min_date': min_date.isoformat(),
                'max_date': max_date.isoformat(),
                'total_records': len(df),
                'market': df['market'].iloc[0] if 'market' in df.columns else 'unknown',
                'issues': issues
            }
            
            if status == 'pass':
                logger.info(f"Stock validation passed: {asset_count} assets, {date_range_days} days")
            else:
                logger.warning(f"Stock validation failed: {', '.join(issues)}")
                
            return result
            
        except Exception as e:
            logger.error(f"Failed to validate stock data: {e}")
            return {
                'status': 'fail',
                'reason': f'Validation error: {str(e)}',
                'asset_count': 0,
                'date_range_days': 0,
                'issues': [str(e)]
            }
            
    def validate_timestamps(self, df: pd.DataFrame, column: str = 'timestamp') -> Tuple[bool, List[str]]:
        """Validate timestamp format and consistency.
        
        Args:
            df: DataFrame to validate
            column: Timestamp column name
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        if column not in df.columns:
            issues.append(f"Timestamp column '{column}' not found")
            return False, issues
            
        # Check if timestamps are datetime64[ns]
        if not pd.api.types.is_datetime64_any_dtype(df[column]):
            issues.append(f"Column '{column}' is not datetime type")
            
        # Check for timezone awareness (should be UTC)
        if df[column].dt.tz is None:
            issues.append(f"Timestamps are not timezone-aware")
        elif str(df[column].dt.tz) != 'UTC':
            issues.append(f"Timestamps are not in UTC (found: {df[column].dt.tz})")
            
        # Check for null timestamps
        null_count = df[column].isna().sum()
        if null_count > 0:
            issues.append(f"Found {null_count} null timestamps")
            
        # Check for future timestamps
        future_count = (df[column] > pd.Timestamp.now(tz='UTC')).sum()
        if future_count > 0:
            issues.append(f"Found {future_count} future timestamps")
            
        is_valid = len(issues) == 0
        return is_valid, issues
        
    def generate_report(self) -> str:
        """Generate validation report.
        
        Returns:
            Formatted validation report
        """
        if not self.validation_results:
            return "No validation results available. Run validate_all() first."
            
        results = self.validation_results
        
        report = []
        report.append("=" * 60)
        report.append("MONEYBALL DATA VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Timestamp: {results['timestamp']}")
        report.append(f"Overall Status: {results['overall_status'].upper()}")
        report.append("")
        
        # Crypto validation
        crypto = results['crypto_validation']
        report.append("CRYPTOCURRENCY DATA")
        report.append("-" * 60)
        report.append(f"Status: {crypto['status'].upper()}")
        report.append(f"Assets: {crypto['asset_count']}")
        report.append(f"Date Range: {crypto.get('date_range_days', 0)} days")
        if 'min_date' in crypto:
            report.append(f"Period: {crypto['min_date']} to {crypto['max_date']}")
        if crypto.get('issues'):
            report.append("Issues:")
            for issue in crypto['issues']:
                report.append(f"  - {issue}")
        report.append("")
        
        # Stock validation
        stock = results['stock_validation']
        report.append("STOCK DATA")
        report.append("-" * 60)
        report.append(f"Status: {stock['status'].upper()}")
        report.append(f"Assets: {stock['asset_count']}")
        report.append(f"Market: {stock.get('market', 'unknown').upper()}")
        report.append(f"Date Range: {stock.get('date_range_days', 0)} days")
        if 'min_date' in stock:
            report.append(f"Period: {stock['min_date']} to {stock['max_date']}")
        if stock.get('issues'):
            report.append("Issues:")
            for issue in stock['issues']:
                report.append(f"  - {issue}")
        report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)


def validate_data() -> Dict[str, any]:
    """Run data validation and return results.
    
    Returns:
        Validation results dictionary
    """
    validator = DataValidator()
    results = validator.validate_all()
    
    # Print report
    report = validator.generate_report()
    print(report)
    
    # Save report to file
    config = get_config()
    report_path = config.log_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write(report)
        
    logger.info(f"Validation report saved to {report_path}")
    
    return results


def main():
    """Main function for running data validation."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    results = validate_data()
    
    # Exit with appropriate code
    if results['overall_status'] == 'pass':
        exit(0)
    elif results['overall_status'] == 'partial':
        exit(1)
    else:
        exit(2)


if __name__ == "__main__":
    main()
