"""Model training with walk-forward validation.

This module handles:
- Target labeling (breakout events)
- XGBoost and LightGBM training
- Hyperparameter tuning with Optuna
- Walk-forward validation
- Model serialization
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, precision_score, recall_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import optuna
import shap

from src.config import get_config

logger = logging.getLogger(__name__)


def label_breakouts(
    prices: pd.DataFrame,
    threshold: float = 0.15,
    horizon_days: int = 30
) -> pd.Series:
    """Create binary labels for breakout events.
    
    Args:
        prices: DataFrame with columns [timestamp, asset, close]
        threshold: Minimum return to qualify as breakout (default 0.15 = 15%)
        horizon_days: Forward-looking window (default 30 days)
        
    Returns:
        Series with 1 if max forward return > threshold, else 0
    """
    logger.info(f"Labeling breakouts (threshold={threshold}, horizon={horizon_days} days)")
    
    df = prices.sort_values(['asset', 'timestamp']).copy()
    
    # Compute forward returns for each day in horizon
    forward_returns = []
    for days in range(1, horizon_days + 1):
        df[f'return_{days}d'] = df.groupby('asset')['close'].pct_change(periods=days).shift(-days)
        forward_returns.append(f'return_{days}d')
    
    # Find maximum forward return
    df['max_forward_return'] = df[forward_returns].max(axis=1)
    
    # Create binary label
    df['breakout_30d'] = (df['max_forward_return'] > threshold).astype(int)
    
    # Drop rows without sufficient forward data
    df = df[df['max_forward_return'].notna()]
    
    # Clean up temporary columns
    df = df.drop(columns=forward_returns + ['max_forward_return'])
    
    breakout_rate = df['breakout_30d'].mean()
    logger.info(f"Breakout rate: {breakout_rate:.2%} ({df['breakout_30d'].sum()} / {len(df)})")
    
    return df.set_index(['timestamp', 'asset'])['breakout_30d']


class BreakoutPredictor:
    """Ensemble model for predicting 30-day breakouts."""
    
    def __init__(self):
        """Initialize predictor."""
        self.xgb_model = None
        self.lgb_model = None
        self.feature_names = None
        self.config = get_config()
        
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        optimize_hyperparams: bool = True
    ) -> Dict:
        """Train XGBoost and LightGBM models.
        
        Args:
            X_train: Feature matrix
            y_train: Binary labels (1 if breakout, else 0)
            optimize_hyperparams: Whether to run Optuna tuning
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training models on {len(X_train)} samples")
        
        self.feature_names = X_train.columns.tolist()
        
        # Calculate class weights
        pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        logger.info(f"Positive class weight: {pos_weight:.2f}")
        
        # Train XGBoost
        if optimize_hyperparams:
            xgb_params = self._optimize_xgboost(X_train, y_train, pos_weight)
        else:
            xgb_params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'scale_pos_weight': pos_weight
            }
        
        self.xgb_model = xgb.XGBClassifier(**xgb_params, random_state=42)
        self.xgb_model.fit(X_train, y_train)
        
        # Train LightGBM
        if optimize_hyperparams:
            lgb_params = self._optimize_lightgbm(X_train, y_train, pos_weight)
        else:
            lgb_params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'scale_pos_weight': pos_weight
            }
        
        self.lgb_model = lgb.LGBMClassifier(**lgb_params, random_state=42)
        self.lgb_model.fit(X_train, y_train)
        
        # Compute training metrics
        y_pred_xgb = self.xgb_model.predict(X_train)
        y_pred_lgb = self.lgb_model.predict(X_train)
        y_pred_ensemble = ((self.xgb_model.predict_proba(X_train)[:, 1] + 
                           self.lgb_model.predict_proba(X_train)[:, 1]) / 2 > 0.5).astype(int)
        
        metrics = {
            'xgb_auc': roc_auc_score(y_train, self.xgb_model.predict_proba(X_train)[:, 1]),
            'lgb_auc': roc_auc_score(y_train, self.lgb_model.predict_proba(X_train)[:, 1]),
            'ensemble_auc': roc_auc_score(y_train, (self.xgb_model.predict_proba(X_train)[:, 1] + 
                                                     self.lgb_model.predict_proba(X_train)[:, 1]) / 2),
            'ensemble_precision': precision_score(y_train, y_pred_ensemble),
            'ensemble_recall': recall_score(y_train, y_pred_ensemble)
        }
        
        logger.info(f"Training metrics: {metrics}")
        return metrics
    
    def _optimize_xgboost(self, X, y, pos_weight) -> Dict:
        """Optimize XGBoost hyperparameters using Optuna."""
        logger.info("Optimizing XGBoost hyperparameters")
        
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'scale_pos_weight': pos_weight
            }
            
            model = xgb.XGBClassifier(**params, random_state=42)
            model.fit(X, y)
            y_pred = model.predict_proba(X)[:, 1]
            return roc_auc_score(y, y_pred)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config.optuna_trials, show_progress_bar=False)
        
        logger.info(f"Best XGBoost AUC: {study.best_value:.4f}")
        return study.best_params
    
    def _optimize_lightgbm(self, X, y, pos_weight) -> Dict:
        """Optimize LightGBM hyperparameters using Optuna."""
        logger.info("Optimizing LightGBM hyperparameters")
        
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'scale_pos_weight': pos_weight
            }
            
            model = lgb.LGBMClassifier(**params, random_state=42)
            model.fit(X, y)
            y_pred = model.predict_proba(X)[:, 1]
            return roc_auc_score(y, y_pred)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config.optuna_trials, show_progress_bar=False)
        
        logger.info(f"Best LightGBM AUC: {study.best_value:.4f}")
        return study.best_params
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict breakout probability using ensemble average.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of probabilities
        """
        if self.xgb_model is None or self.lgb_model is None:
            raise ValueError("Models not trained. Call train() first.")
        
        xgb_probs = self.xgb_model.predict_proba(X)[:, 1]
        lgb_probs = self.lgb_model.predict_proba(X)[:, 1]
        
        return (xgb_probs + lgb_probs) / 2
    
    def explain(self, X: pd.DataFrame, max_features: int = 10) -> pd.DataFrame:
        """Compute SHAP values for feature importance.
        
        Args:
            X: Feature matrix
            max_features: Maximum features to return
            
        Returns:
            DataFrame with top features and SHAP values
        """
        logger.info("Computing SHAP values")
        
        # Use XGBoost for SHAP (faster than LightGBM)
        explainer = shap.TreeExplainer(self.xgb_model)
        shap_values = explainer.shap_values(X)
        
        # Get mean absolute SHAP values
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'shap_importance': mean_shap
        }).sort_values('shap_importance', ascending=False).head(max_features)
        
        return importance_df
    
    def save(self, path: Path):
        """Save models to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.xgb_model, path / 'xgb_model.pkl')
        joblib.dump(self.lgb_model, path / 'lgb_model.pkl')
        joblib.dump(self.feature_names, path / 'feature_names.pkl')
        
        logger.info(f"Models saved to {path}")
    
    def load(self, path: Path):
        """Load models from disk."""
        path = Path(path)
        
        self.xgb_model = joblib.load(path / 'xgb_model.pkl')
        self.lgb_model = joblib.load(path / 'lgb_model.pkl')
        self.feature_names = joblib.load(path / 'feature_names.pkl')
        
        logger.info(f"Models loaded from {path}")


def walk_forward_validation(
    features: pd.DataFrame,
    labels: pd.Series,
    train_months: int = 6,
    test_months: int = 1
) -> Dict:
    """Perform walk-forward cross-validation.
    
    Args:
        features: Feature matrix with datetime index
        labels: Binary target labels
        train_months: Size of training window
        test_months: Size of test window
        
    Returns:
        Dictionary with fold-wise metrics and aggregated performance
    """
    logger.info(f"Walk-forward validation ({train_months}m train, {test_months}m test)")
    
    # Merge features and labels
    df = features.join(labels.rename('target'), how='inner')
    df = df.sort_index()
    
    # Get date range
    start_date = df.index.get_level_values('timestamp').min()
    end_date = df.index.get_level_values('timestamp').max()
    
    # Create folds
    folds = []
    current_date = start_date + timedelta(days=train_months * 30)
    
    while current_date + timedelta(days=test_months * 30) <= end_date:
        train_end = current_date
        test_start = current_date
        test_end = current_date + timedelta(days=test_months * 30)
        
        folds.append({
            'train_start': start_date,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end
        })
        
        current_date += timedelta(days=test_months * 30)
    
    logger.info(f"Created {len(folds)} folds")
    
    # Train and evaluate on each fold
    fold_results = []
    
    for i, fold in enumerate(folds):
        logger.info(f"Fold {i+1}/{len(folds)}: Train {fold['train_start']} to {fold['train_end']}, "
                   f"Test {fold['test_start']} to {fold['test_end']}")
        
        # Split data
        train_mask = (df.index.get_level_values('timestamp') >= fold['train_start']) & \
                     (df.index.get_level_values('timestamp') < fold['train_end'])
        test_mask = (df.index.get_level_values('timestamp') >= fold['test_start']) & \
                    (df.index.get_level_values('timestamp') < fold['test_end'])
        
        train_df = df[train_mask]
        test_df = df[test_mask]
        
        if len(train_df) == 0 or len(test_df) == 0:
            logger.warning(f"Fold {i+1}: Insufficient data, skipping")
            continue
        
        X_train = train_df.drop(columns=['target'])
        y_train = train_df['target']
        X_test = test_df.drop(columns=['target'])
        y_test = test_df['target']
        
        # Train model
        predictor = BreakoutPredictor()
        predictor.train(X_train, y_train, optimize_hyperparams=False)
        
        # Evaluate
        y_pred_proba = predictor.predict_proba(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        fold_metrics = {
            'fold': i + 1,
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0)
        }
        
        fold_results.append(fold_metrics)
        logger.info(f"Fold {i+1} metrics: AUC={fold_metrics['roc_auc']:.4f}, "
                   f"Precision={fold_metrics['precision']:.4f}, Recall={fold_metrics['recall']:.4f}")
    
    # Aggregate results
    if fold_results:
        results = {
            'folds': fold_results,
            'mean_roc_auc': np.mean([f['roc_auc'] for f in fold_results]),
            'mean_precision': np.mean([f['precision'] for f in fold_results]),
            'mean_recall': np.mean([f['recall'] for f in fold_results]),
            'std_roc_auc': np.std([f['roc_auc'] for f in fold_results])
        }
        
        logger.info(f"Walk-forward validation complete:")
        logger.info(f"  Mean ROC-AUC: {results['mean_roc_auc']:.4f} ± {results['std_roc_auc']:.4f}")
        logger.info(f"  Mean Precision: {results['mean_precision']:.4f}")
        logger.info(f"  Mean Recall: {results['mean_recall']:.4f}")
        
        # Check threshold
        config = get_config()
        if results['mean_roc_auc'] < config.min_roc_auc:
            logger.warning(f"ROC-AUC {results['mean_roc_auc']:.4f} below threshold {config.min_roc_auc}")
        else:
            logger.info(f"ROC-AUC {results['mean_roc_auc']:.4f} meets threshold {config.min_roc_auc}")
        
        return results
    else:
        logger.error("No folds completed successfully")
        return {}


def main():
    """Main function for model training."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    config = get_config()
    
    # Load features
    features_path = config.features_dir / 'features.parquet'
    if not features_path.exists():
        logger.error(f"Features not found at {features_path}. Run feature pipeline first.")
        return
    
    logger.info(f"Loading features from {features_path}")
    features = pd.read_parquet(features_path)
    
    # Load prices for labeling
    # This is simplified - in production, load from raw data
    prices = features[['timestamp', 'asset', 'close']].copy()
    
    # Label breakouts
    labels = label_breakouts(prices, threshold=config.breakout_threshold, 
                            horizon_days=config.breakout_horizon_days)
    
    # Prepare features - exclude non-numeric columns (e.g. regime string)
    exclude_cols = ['timestamp', 'asset', 'close', 'asset_type', 'regime']
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    features_indexed = features.set_index(['timestamp', 'asset'])[feature_cols]
    
    # Run walk-forward validation
    results = walk_forward_validation(
        features_indexed,
        labels,
        train_months=config.train_months,
        test_months=config.test_months
    )
    
    # Train final model on all data
    logger.info("Training final model on all data")
    df = features_indexed.join(labels.rename('target'), how='inner')
    X = df.drop(columns=['target'])
    y = df['target']
    
    predictor = BreakoutPredictor()
    predictor.train(X, y, optimize_hyperparams=True)
    
    # Save model
    predictor.save(config.models_dir)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
