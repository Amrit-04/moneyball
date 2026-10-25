"""Machine Learning Alpha Predictors.

Wrappers for cross-sectional return prediction models:
1. Ridge Regression (L2 regularized linear model)
2. Random Forest Regressor (ensemble decision trees)
3. Gradient Boosting / XGBoost Regressor (boosting trees)
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "mom_5d",
    "mom_20d",
    "mom_60d",
    "realized_vol_20d",
    "rsi_14d",
    "bollinger_z_20d",
    "volume_shock",
    "sentiment_proxy",
]


class AlphaModelSuite:
    """Manages training and prediction across candidate alpha architectures."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Any] = {
            "Ridge": Ridge(alpha=10.0, random_state=random_state),
            "RandomForest": RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=20,
                random_state=random_state,
                n_jobs=-1
            ),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                random_state=random_state
            )
        }
        self.scalers: Dict[str, StandardScaler] = {
            name: StandardScaler() for name in self.models
        }
        self.is_fitted: Dict[str, bool] = {name: False for name in self.models}
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AlphaModelSuite":
        """Fit all models on feature matrix X and target return y."""
        X_clean = X[FEATURE_COLUMNS].copy().fillna(0.0)
        y_clean = y.copy().fillna(0.0)
        
        # Clip extreme outliers in return labels for stability
        y_clipped = np.clip(y_clean, -0.20, 0.20)
        
        for name, model in self.models.items():
            scaler = self.scalers[name]
            X_scaled = scaler.fit_transform(X_clean)
            model.fit(X_scaled, y_clipped)
            self.is_fitted[name] = True
            
        return self

    def predict(self, X: pd.DataFrame, model_name: str = "GradientBoosting") -> np.ndarray:
        """Generate out-of-sample alpha score predictions."""
        if not self.is_fitted.get(model_name, False):
            raise RuntimeError(f"Model {model_name} is not fitted.")
            
        X_clean = X[FEATURE_COLUMNS].copy().fillna(0.0)
        scaler = self.scalers[model_name]
        X_scaled = scaler.transform(X_clean)
        model = self.models[model_name]
        return model.predict(X_scaled)

    def get_feature_importances(self, model_name: str = "GradientBoosting") -> Dict[str, float]:
        """Extract normalized feature importance weights."""
        if not self.is_fitted.get(model_name, False):
            raise RuntimeError(f"Model {model_name} is not fitted.")
            
        model = self.models[model_name]
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "coef_"):
            imp = np.abs(model.coef_)
        else:
            imp = np.ones(len(FEATURE_COLUMNS))
            
        imp = imp / (np.sum(imp) + 1e-12)
        return dict(zip(FEATURE_COLUMNS, imp))
