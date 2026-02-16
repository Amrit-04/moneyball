"""Audit logging for Moneyball system."""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class AuditLogger:
    """Centralized logger for agent actions and model decisions."""
    
    def __init__(self, log_dir: Path = Path("logs")):
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True)
        self.current_log_file = self.log_dir / f"audit_{datetime.now():%Y%m%d}.jsonl"
    
    def log_event(self, event_type: str, agent: str, asset: str, details: dict, latency_ms: float = None):
        """Log a single event to JSONL file."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "agent": agent,
            "asset": asset,
            "details": details,
            "latency_ms": latency_ms
        }
        
        with open(self.current_log_file, "a") as f:
            f.write(json.dumps(event) + "\n")
