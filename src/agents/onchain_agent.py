"""On-chain agent for whale and DEX data analysis."""

import logging
from typing import Dict
from datetime import datetime
import pandas as pd

from src.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class OnChainAgent:
    """Specialist agent for on-chain whale and DEX analysis."""
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
    
    def analyze(self, asset: str, features: pd.Series) -> Dict:
        """Analyze on-chain whale activity and DEX flows."""
        start_time = datetime.now()
        
        analysis = {
            "asset": asset,
            "whale_accumulation_7d": float(features.get("whale_accumulation_7d", 0)),
            "exchange_netflow_divergence": float(features.get("exchange_netflow_divergence", 0)),
            "stablecoin_supply_ratio": float(features.get("stablecoin_supply_ratio", 0)),
            "defi_tvl_momentum_30d": float(features.get("defi_tvl_momentum_30d", 0)),
            "dev_activity_30d": float(features.get("dev_activity_30d", 0))
        }
        
        # On-chain signals
        signals = []
        if analysis["whale_accumulation_7d"] > 0.05:
            signals.append("whale_accumulation")
        
        if analysis["exchange_netflow_divergence"] < -0.1:
            signals.append("exchange_outflow")  # Bullish
        
        if analysis["stablecoin_supply_ratio"] > 0.15:
            signals.append("high_buying_power")
        
        if analysis["defi_tvl_momentum_30d"] > 0.1:
            signals.append("tvl_growth")
        
        if analysis["dev_activity_30d"] > 50:
            signals.append("active_development")
        
        analysis["signals"] = signals
        analysis["signal_count"] = len(signals)
        
        latency = (datetime.now() - start_time).total_seconds() * 1000
        self.audit_logger.log_event(
            event_type="onchain_analysis",
            agent="onchain",
            asset=asset,
            details=analysis,
            latency_ms=latency
        )
        
        logger.info(f"On-chain analysis for {asset}: {len(signals)} signals")
        return analysis
