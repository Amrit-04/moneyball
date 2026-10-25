"""Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR) Calculations.

Implements exact sample quantile VaR and Conditional Expectation (Expected Shortfall)
with asymptotic standard error estimates and mathematical bound verification.
"""

from typing import Dict, Tuple, Union
import numpy as np


def compute_var_cvar(
    returns: np.ndarray,
    confidence_levels: Tuple[float, ...] = (0.95, 0.99),
) -> Dict[str, float]:
    """Compute VaR and CVaR (Expected Shortfall) at specified confidence levels.
    
    Expressed as positive loss percentages:
      VaR_alpha = -Quantile(returns, 1 - alpha)
      CVaR_alpha = -Mean(returns[returns <= -VaR_alpha])
      
    Args:
        returns: 1D array of portfolio returns across simulated paths.
        confidence_levels: Tuple of confidence thresholds (e.g. 0.95, 0.99).
        
    Returns:
        Dict mapping metric names to positive percentage values.
    """
    clean_ret = np.asarray(returns).ravel()
    n_paths = len(clean_ret)
    results = {}
    
    for alpha in confidence_levels:
        q_level = (1.0 - alpha) * 100.0
        # Value-at-Risk is the negative of the lower quantile
        var_threshold = -np.percentile(clean_ret, q_level)
        
        # Tail returns that breach VaR
        tail_losses = clean_ret[clean_ret <= -var_threshold]
        if len(tail_losses) == 0:
            cvar = var_threshold
        else:
            cvar = -np.mean(tail_losses)
            
        # Mathematical invariance: CVaR must be >= VaR
        if cvar < var_threshold:
            cvar = var_threshold
            
        alpha_pct = int(alpha * 100)
        results[f"VaR_{alpha_pct}"] = float(var_threshold)
        results[f"CVaR_{alpha_pct}"] = float(cvar)
        
        # Asymptotic Monte Carlo Standard Error of CVaR
        if len(tail_losses) > 1:
            se_cvar = float(np.std(tail_losses, ddof=1) / np.sqrt(len(tail_losses)))
        else:
            se_cvar = 0.0
        results[f"CVaR_{alpha_pct}_SE"] = se_cvar
        
    results["n_paths"] = int(n_paths)
    return results
