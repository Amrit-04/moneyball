"""Quantitative Alpha Statistical Metrics.

Computes:
1. Daily Cross-Sectional Spearman Rank Information Coefficient (Rank IC)
2. IC Information Ratio (ICIR - annualized signal consistency)
3. Directional Hit Rate (% sign concordance)
4. Newey-West Autocorrelation & Heteroskedasticity Consistent (HAC) t-statistic & p-value
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


def compute_daily_rank_ic(
    predictions_df: pd.DataFrame,
    targets_df: pd.DataFrame
) -> pd.Series:
    """Compute cross-sectional Spearman Rank IC for each trading date.
    
    Args:
        predictions_df: DataFrame of alpha predictions (index=date, columns=assets).
        targets_df: DataFrame of forward realized returns (index=date, columns=assets).
        
    Returns:
        pd.Series of daily Rank ICs indexed by date.
    """
    common_dates = predictions_df.index.intersection(targets_df.index)
    daily_ic = []
    valid_dates = []
    
    for d in common_dates:
        p_row = predictions_df.loc[d].dropna()
        t_row = targets_df.loc[d].dropna()
        common_assets = p_row.index.intersection(t_row.index)
        
        if len(common_assets) >= 5:
            # Spearman rank correlation
            corr, _ = stats.spearmanr(p_row.loc[common_assets], t_row.loc[common_assets])
            if not np.isnan(corr):
                daily_ic.append(corr)
                valid_dates.append(d)
                
    return pd.Series(daily_ic, index=pd.DatetimeIndex(valid_dates), name="Rank_IC")


def compute_newey_west_stats(ic_series: pd.Series, max_lags: int = 5) -> Tuple[float, float]:
    """Compute Newey-West (HAC) adjusted t-statistic and two-sided p-value for IC series.
    
    Args:
        ic_series: Time series of Rank IC values.
        max_lags: Maximum lag for Bartlett kernel (default 5 for weekly horizon).
        
    Returns:
        Tuple of (t_stat, p_value)
    """
    clean_ic = ic_series.dropna().values
    if len(clean_ic) < 10:
        return 0.0, 1.0
        
    # Regress IC on a constant
    X = np.ones((len(clean_ic), 1))
    ols_model = sm.OLS(clean_ic, X)
    res = ols_model.fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})
    
    t_stat = float(res.tvalues[0])
    p_val = float(res.pvalues[0])
    return t_stat, p_val


def compute_alpha_performance_metrics(
    predictions_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    annualization_factor: float = np.sqrt(252.0)
) -> Dict[str, float]:
    """Calculate comprehensive suite of statistical alpha metrics.
    
    Returns:
        Dict containing:
          - mean_rank_ic: Mean Spearman Rank IC
          - std_rank_ic: Standard deviation of Rank IC
          - icir: Information Ratio of IC (annualized)
          - hit_rate: Directional accuracy (% sign agreement)
          - newey_west_tstat: HAC-adjusted t-statistic
          - newey_west_pvalue: HAC-adjusted p-value
    """
    ic_series = compute_daily_rank_ic(predictions_df, targets_df)
    
    if len(ic_series) == 0:
        return {
            "mean_rank_ic": 0.0,
            "std_rank_ic": 0.0,
            "icir": 0.0,
            "hit_rate": 0.50,
            "newey_west_tstat": 0.0,
            "newey_west_pvalue": 1.0,
            "n_observations": 0
        }
        
    mean_ic = float(ic_series.mean())
    std_ic = float(ic_series.std()) if ic_series.std() > 0 else 1e-6
    icir = float((mean_ic / std_ic) * annualization_factor)
    
    # Hit rate across all valid prediction/target pairs
    common_idx = predictions_df.index.intersection(targets_df.index)
    p_flat = predictions_df.loc[common_idx].values.flatten()
    t_flat = targets_df.loc[common_idx].values.flatten()
    valid_mask = ~np.isnan(p_flat) & ~np.isnan(t_flat)
    
    p_valid = p_flat[valid_mask]
    t_valid = t_flat[valid_mask]
    
    hit_rate = float(np.mean(np.sign(p_valid) == np.sign(t_valid))) if len(p_valid) > 0 else 0.50
    t_stat, p_val = compute_newey_west_stats(ic_series)
    
    return {
        "mean_rank_ic": round(mean_ic, 4),
        "std_rank_ic": round(std_ic, 4),
        "icir": round(icir, 3),
        "hit_rate": round(hit_rate, 4),
        "newey_west_tstat": round(t_stat, 3),
        "newey_west_pvalue": round(p_val, 4),
        "n_observations": int(len(ic_series))
    }
