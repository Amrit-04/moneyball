"""Sentiment agent for Reddit and social media analysis."""

import logging
from typing import Dict
from datetime import datetime
import pandas as pd

from src.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class SentimentAgent:
    """Specialist agent for sentiment and social media analysis."""
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
    
    def analyze(self, asset: str, sentiment_data: pd.DataFrame = None) -> Dict:
        """Analyze Reddit sentiment and social signals."""
        start_time = datetime.now()
        
        if sentiment_data is None or sentiment_data.empty:
            analysis = {
                "asset": asset,
                "reddit_mentions": 0,
                "avg_sentiment": 0.0,
                "sentiment_trend": "neutral",
                "signals": ["no_data"]
            }
        else:
            # Filter for this asset
            asset_sentiment = sentiment_data[sentiment_data["asset"] == asset]
            
            analysis = {
                "asset": asset,
                "reddit_mentions": len(asset_sentiment),
                "avg_sentiment": float(asset_sentiment["sentiment"].mean()) if not asset_sentiment.empty else 0.0,
                "sentiment_trend": "neutral"
            }
            
            # Sentiment signals
            signals = []
            if analysis["reddit_mentions"] > 100:
                signals.append("high_social_activity")
            
            if analysis["avg_sentiment"] > 0.3:
                signals.append("positive_sentiment")
                analysis["sentiment_trend"] = "bullish"
            elif analysis["avg_sentiment"] < -0.3:
                signals.append("negative_sentiment")
                analysis["sentiment_trend"] = "bearish"
            
            analysis["signals"] = signals
        
        latency = (datetime.now() - start_time).total_seconds() * 1000
        self.audit_logger.log_event(
            event_type="sentiment_analysis",
            agent="sentiment",
            asset=asset,
            details=analysis,
            latency_ms=latency
        )
        
        logger.info(f"Sentiment analysis for {asset}: {analysis.get('sentiment_trend', 'neutral')}")
        return analysis
