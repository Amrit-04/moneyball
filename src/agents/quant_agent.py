"""Quant agent for price and fundamental analysis."""

import logging
from typing import Dict
from datetime import datetime
import pandas as pd

from src.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class QuantAgent:
    """Specialist agent for quantitative price and fundamental analysis."""
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
    
    def analyze(self, asset: str, features: pd.Series) -> Dict:
        """Analyze price action and fundamentals."""
        start_time = datetime.now()
        
        analysis = {
            "asset": asset,
            "current_price": float(features.get("close", 0)),
            "rsi_14": float(features.get("rsi_14", 50)),
            "macd": float(features.get("macd", 0)),
            "bollinger_pct_b": float(features.get("bollinger_pct_b", 0.5)),
            "volume_change_7d": float(features.get("volume_change_7d", 0)),
            "momentum_1m": float(features.get("momentum_1m", 0)),
            "momentum_3m": float(features.get("momentum_3m", 0))
        }
        
        # Technical signals
        signals = []
        if analysis["rsi_14"] < 30:
            signals.append("oversold_rsi")
        elif analysis["rsi_14"] > 70:
            signals.append("overbought_rsi")
        
        if analysis["macd"] > 0:
            signals.append("bullish_macd")
        
        if analysis["bollinger_pct_b"] < 0.2:
            signals.append("near_lower_band")
        elif analysis["bollinger_pct_b"] > 0.8:
            signals.append("near_upper_band")
        
        if analysis["momentum_1m"] > 0.05:
            signals.append("strong_momentum")
        
        analysis["signals"] = signals
        analysis["signal_count"] = len(signals)
        
        latency = (datetime.now() - start_time).total_seconds() * 1000
        self.audit_logger.log_event(
            event_type="quant_analysis",
            agent="quant",
            asset=asset,
            details=analysis,
            latency_ms=latency
        )
        
        logger.info(f"Quant analysis for {asset}: {len(signals)} signals")
        return analysis
