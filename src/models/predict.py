"""Inference pipeline for generating investment recommendations."""

import logging
from pathlib import Path
from typing import List, Dict
import pandas as pd

from src.config import get_config
from src.models.train import BreakoutPredictor
from src.features.regime import RegimeDetector
from src.agents.orchestrator import OrchestratorAgent
from src.agents.quant_agent import QuantAgent
from src.agents.onchain_agent import OnChainAgent
from src.agents.sentiment_agent import SentimentAgent
from src.agents.cio_agent import CIOAgent
from src.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class InferencePipeline:
    """End-to-end inference pipeline for investment recommendations."""
    
    def __init__(self):
        self.config = get_config()
        self.audit_logger = AuditLogger(self.config.log_dir)
        
        # Load model
        self.predictor = BreakoutPredictor()
        self.predictor.load(self.config.models_dir)
        
        # Initialize agents
        self.orchestrator = OrchestratorAgent(self.audit_logger)
        self.quant_agent = QuantAgent(self.audit_logger)
        self.onchain_agent = OnChainAgent(self.audit_logger)
        self.sentiment_agent = SentimentAgent(self.audit_logger)
        self.cio_agent = CIOAgent(self.audit_logger)
        
        # Load regime detector
        self.regime_detector = RegimeDetector()
        
        logger.info("Inference pipeline initialized")
    
    def generate_recommendations(self, top_n: int = 20, 
                                asset_types: List[str] = None) -> pd.DataFrame:
        """Generate top N investment recommendations."""
        # Load features
        features_path = self.config.features_dir / 'features.parquet'
        if not features_path.exists():
            raise FileNotFoundError(f"Features not found at {features_path}")
        
        features = pd.read_parquet(features_path)
        logger.info(f"Loaded {len(features)} feature rows")
        
        # Filter by asset type
        if asset_types:
            features = features[features['asset_type'].isin(asset_types)]
        
        # Get latest data per asset
        features = features.sort_values('timestamp').groupby('asset').tail(1)
        
        # Predict breakout scores - use model's feature names (numeric only)
        feature_cols = [c for c in self.predictor.feature_names if c in features.columns]
        X = features[feature_cols].fillna(0)
        
        scores = self.predictor.predict_proba(X)
        features['breakout_score'] = scores
        
        # Get top N
        top_assets = features.nlargest(top_n, 'breakout_score')
        
        # Detect current regime
        regime = self._detect_regime()
        
        # Generate memos for each asset
        recommendations = []
        for idx, row in top_assets.iterrows():
            memo = self._generate_memo(row, regime)
            recommendations.append({
                'asset': row['asset'],
                'asset_type': row.get('asset_type', 'unknown'),
                'breakout_score': row['breakout_score'],
                'current_price': row.get('close', 0),
                'regime': regime,
                'memo': memo
            })
        
        return pd.DataFrame(recommendations)
    
    def _detect_regime(self) -> str:
        """Detect current market regime."""
        try:
            # Load recent price data
            raw_dir = self.config.raw_data_dir
            crypto_path = raw_dir / 'crypto_prices.parquet'
            
            if crypto_path.exists():
                prices = pd.read_parquet(crypto_path)
                btc_prices = prices[prices['asset'] == 'bitcoin'].sort_values('timestamp')
                
                if len(btc_prices) > 30:
                    returns = btc_prices['close'].pct_change().dropna()
                    self.regime_detector.fit(returns.values.reshape(-1, 1))
                    current_regime = self.regime_detector.predict(returns.tail(1).values.reshape(-1, 1))[0]
                    
                    regime_map = {0: "Bull", 1: "Bear", 2: "Sideways"}
                    return regime_map.get(current_regime, "Unknown")
        except Exception as e:
            logger.warning(f"Regime detection failed: {e}")
        
        return "Unknown"
    
    def _generate_memo(self, asset_row: pd.Series, regime: str) -> str:
        """Generate investment memo for a single asset."""
        asset = asset_row['asset']
        asset_type = asset_row.get('asset_type', 'unknown')
        
        # Plan workflow
        workflow = self.orchestrator.plan_workflow(asset, asset_type)
        
        # Execute workflow
        quant_analysis = None
        onchain_analysis = None
        sentiment_analysis = None
        
        for task in workflow:
            if task == "quant":
                quant_analysis = self.quant_agent.analyze(asset, asset_row)
            elif task == "onchain" and asset_type == "crypto":
                onchain_analysis = self.onchain_agent.analyze(asset, asset_row)
            elif task == "sentiment":
                sentiment_analysis = self.sentiment_agent.analyze(asset)
            elif task == "cio":
                memo = self.cio_agent.synthesize(
                    asset=asset,
                    quant_analysis=quant_analysis,
                    onchain_analysis=onchain_analysis,
                    sentiment_analysis=sentiment_analysis,
                    regime=regime,
                    breakout_score=asset_row['breakout_score']
                )
                return memo
        
        return "Memo generation failed"


def main():
    """CLI entry point for generating recommendations."""
    logging.basicConfig(level=logging.INFO)
    
    pipeline = InferencePipeline()
    recommendations = pipeline.generate_recommendations(top_n=20)
    
    print("\n" + "="*80)
    print("MONEYBALL - TOP INVESTMENT OPPORTUNITIES")
    print("="*80 + "\n")
    
    for idx, rec in recommendations.iterrows():
        print(f"\n{'='*80}")
        print(f"#{idx+1} | {rec['asset'].upper()} | Score: {rec['breakout_score']:.1%}")
        print(f"{'='*80}")
        print(rec['memo'])
    
    # Save to file
    config = get_config()
    output_path = config.log_dir / f"recommendations_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv"
    recommendations.to_csv(output_path, index=False)
    print(f"\nRecommendations saved to {output_path}")


if __name__ == "__main__":
    main()
