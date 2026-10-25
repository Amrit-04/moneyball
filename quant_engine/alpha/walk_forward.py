"""Purged & Embargoed Walk-Forward Cross-Validation.

Implements Marcos López de Prado's Purging and Embargoing methodology for financial
time-series alpha models:
- Purging: Eliminates training observations whose forward-looking labels overlap
  with test evaluation periods.
- Embargoing: Enforces a cooldown period immediately following test intervals
  to mitigate autoregressive residual leakage.
"""

from typing import Generator, List, Optional, Tuple
import numpy as np
import pandas as pd


class PurgedWalkForwardCV:
    """Purged & Embargoed Walk-Forward Cross-Validation for panel or time-series data."""
    
    def __init__(
        self,
        n_splits: int = 5,
        train_window: int = 504,     # 2 years of trading days
        test_window: int = 63,       # 1 quarter (~3 months)
        purge_window: int = 5,       # 5 days (label overlap horizon)
        embargo_window: int = 5,     # 5 days embargo after test window
    ):
        """Initialize the Purged Walk-Forward CV split generator.
        
        Args:
            n_splits: Target number of walk-forward rolling splits.
            train_window: Number of trading days in rolling training set.
            test_window: Number of trading days in out-of-sample evaluation set.
            purge_window: Number of days purged from training tail to prevent target leakage.
            embargo_window: Number of days embargoed after testing before re-entering training.
        """
        self.n_splits = n_splits
        self.train_window = train_window
        self.test_window = test_window
        self.purge_window = purge_window
        self.embargo_window = embargo_window
        
    def split(
        self,
        panel_df: pd.DataFrame,
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex, pd.DatetimeIndex], None, None]:
        """Yield (train_df, test_df, train_dates, test_dates) tuples across walk-forward folds.
        
        Args:
            panel_df: DataFrame indexed by either DatetimeIndex or MultiIndex (date, asset).
            
        Yields:
            Tuple of:
              - train_df: Purged training data slice
              - test_df: Out-of-sample test data slice
              - train_dates: Unique trading dates in training split
              - test_dates: Unique trading dates in test split
        """
        if isinstance(panel_df.index, pd.MultiIndex):
            dates = pd.DatetimeIndex(panel_df.index.get_level_values(0).unique()).sort_values()
        else:
            dates = pd.DatetimeIndex(panel_df.index.unique()).sort_values()
            
        total_days = len(dates)
        required_span = self.train_window + self.test_window + self.embargo_window
        if total_days < required_span:
            raise ValueError(
                f"Total days ({total_days}) is less than required minimum span ({required_span})."
            )
            
        # Determine step size between folds
        available_days = total_days - self.train_window - self.test_window
        step = max(1, available_days // self.n_splits)
        
        for fold in range(self.n_splits):
            train_start_idx = fold * step
            train_end_idx = train_start_idx + self.train_window - self.purge_window
            
            test_start_idx = train_start_idx + self.train_window
            test_end_idx = test_start_idx + self.test_window
            
            if test_end_idx > total_days:
                break
                
            train_dates = dates[train_start_idx:train_end_idx]
            test_dates = dates[test_start_idx:test_end_idx]
            
            # Extract slices
            if isinstance(panel_df.index, pd.MultiIndex):
                train_mask = panel_df.index.get_level_values(0).isin(train_dates)
                test_mask = panel_df.index.get_level_values(0).isin(test_dates)
                train_slice = panel_df[train_mask]
                test_slice = panel_df[test_mask]
            else:
                train_slice = panel_df.loc[train_dates]
                test_slice = panel_df.loc[test_dates]
                
            yield train_slice, test_slice, train_dates, test_dates
