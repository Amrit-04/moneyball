"""Orchestrator agent for workflow planning and task routing."""

import logging
from typing import Dict, List
from datetime import datetime

from src.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class OrchestratorAgent:
    """Orchestrates workflow between specialist agents."""
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
    
    def plan_workflow(self, asset: str, asset_type: str) -> List[str]:
        """Plan analysis workflow based on asset type."""
        start_time = datetime.now()
        
        workflow = ["quant"]  # Always run quant analysis
        
        if asset_type == "crypto":
            workflow.extend(["onchain", "sentiment"])
        elif asset_type == "stock":
            workflow.append("sentiment")
        
        workflow.append("cio")  # Always synthesize at end
        
        latency = (datetime.now() - start_time).total_seconds() * 1000
        self.audit_logger.log_event(
            event_type="workflow_planned",
            agent="orchestrator",
            asset=asset,
            details={"workflow": workflow, "asset_type": asset_type},
            latency_ms=latency
        )
        
        logger.info(f"Planned workflow for {asset}: {workflow}")
        return workflow
    
    def route_task(self, task: str, asset: str, data: Dict) -> str:
        """Route task to appropriate specialist agent."""
        routing = {
            "quant": "QuantAgent",
            "onchain": "OnChainAgent",
            "sentiment": "SentimentAgent",
            "cio": "CIOAgent"
        }
        
        agent = routing.get(task, "Unknown")
        logger.info(f"Routing {task} for {asset} to {agent}")
        
        return agent
