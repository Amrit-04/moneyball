"""CIO Agent for investment memo synthesis."""

import logging
from typing import Dict
from datetime import datetime
import google.generativeai as genai
from src.config import get_config
from src.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class CIOAgent:
    """Chief Investment Officer agent for synthesis and memo generation."""
    
    def __init__(self, audit_logger: AuditLogger = None):
        self.config = get_config()
        self.audit_logger = audit_logger
        
        if self.config.gemini_api_key:
            genai.configure(api_key=self.config.gemini_api_key)
            self.model = genai.GenerativeModel(self.config.llm_model)
        else:
            self.model = None
            logger.warning("Gemini API key not configured, using template fallback")
    
    def synthesize(self, asset: str, quant_analysis: Dict, onchain_analysis: Dict = None, 
                   sentiment_analysis: Dict = None, regime: str = "Unknown", 
                   breakout_score: float = 0.0) -> str:
        """Generate investment memo from specialist outputs."""
        start_time = datetime.now()
        
        if not self.model:
            memo = self._template_memo(asset, quant_analysis, onchain_analysis, 
                                      sentiment_analysis, regime, breakout_score)
        else:
            memo = self._llm_memo(asset, quant_analysis, onchain_analysis, 
                                 sentiment_analysis, regime, breakout_score)
        
        if self.audit_logger:
            latency = (datetime.now() - start_time).total_seconds() * 1000
            self.audit_logger.log_event(
                event_type="memo_generated",
                agent="cio",
                asset=asset,
                details={"memo_length": len(memo), "breakout_score": breakout_score},
                latency_ms=latency
            )
        
        return memo
    
    def _llm_memo(self, asset: str, quant: Dict, onchain: Dict, sentiment: Dict, 
                  regime: str, score: float) -> str:
        """Generate memo using Gemini LLM."""
        prompt = f"""Generate a concise investment memo for {asset}.

**Breakout Score:** {score:.1%}
**Market Regime:** {regime}

**Quant Analysis:**
{self._format_dict(quant)}

**On-Chain Analysis:**
{self._format_dict(onchain) if onchain else "N/A"}

**Sentiment Analysis:**
{self._format_dict(sentiment) if sentiment else "N/A"}

Format the memo as:
# Investment Memo: {asset}

## Conviction: [High/Medium/Low]

## Key Signals
- [List 3-5 most important signals]

## Risk Factors
- [List 2-3 key risks]

## Recommendation
[1-2 sentence actionable recommendation]
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return self._template_memo(asset, quant, onchain, sentiment, regime, score)
    
    def _template_memo(self, asset: str, quant: Dict, onchain: Dict, sentiment: Dict, 
                       regime: str, score: float) -> str:
        """Fallback template-based memo."""
        # Determine conviction
        if score > 0.7:
            conviction = "High"
        elif score > 0.5:
            conviction = "Medium"
        else:
            conviction = "Low"
        
        # Collect signals
        signals = []
        if quant:
            signals.extend([f"Quant: {s}" for s in quant.get("signals", [])])
        if onchain:
            signals.extend([f"On-chain: {s}" for s in onchain.get("signals", [])])
        if sentiment:
            signals.extend([f"Sentiment: {s}" for s in sentiment.get("signals", [])])
        
        signals_text = "\n".join([f"- {s}" for s in signals[:5]]) if signals else "- No strong signals"
        
        return f"""# Investment Memo: {asset}

## Conviction: {conviction}
**Breakout Score:** {score:.1%}  
**Market Regime:** {regime}

## Key Signals
{signals_text}

## Quantitative Metrics
- Current Price: ${quant.get('current_price', 0):.2f}
- RSI(14): {quant.get('rsi_14', 50):.1f}
- 1M Momentum: {quant.get('momentum_1m', 0):.1%}

## Risk Factors
- Market volatility in {regime} regime
- Liquidity risk for small-cap asset
- Model prediction uncertainty

## Recommendation
{"Strong buy signal - consider position sizing based on conviction." if score > 0.7 
 else "Monitor for entry opportunity - wait for confirmation." if score > 0.5
 else "Insufficient signal strength - pass for now."}
"""
    
    def _format_dict(self, d: Dict) -> str:
        """Format dictionary for LLM prompt."""
        if not d:
            return "N/A"
        return "\n".join([f"- {k}: {v}" for k, v in d.items() if k != "signals"])
