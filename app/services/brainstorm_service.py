import time
from datetime import datetime, timezone
from typing import List, Dict, Any
from collections import deque

class BrainstormService:
    """
    Captures and streams real-time AI thinking steps, analysis milestones,
    risk calculations, and decision rationales to the Web Dashboard.
    """
    def __init__(self, max_entries: int = 150):
        self.logs: deque = deque(maxlen=max_entries)
        # Add initial greeting logs
        self.add_log(
            level="SYSTEM",
            category="BOOT",
            symbol="SYSTEM",
            message="🚀 AI Brainstorm Core Initialized. Multi-School Analysis Engine Ready (Classical / Fibonacci / SMC)."
        )

    def add_log(self, level: str, category: str, symbol: str, message: str, details: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        level: 'INFO' | 'ANALYSIS' | 'BRAINSTORM' | 'SIGNAL' | 'RISK' | 'EXECUTION' | 'SYSTEM'
        category: 'CLASSICAL' | 'FIBONACCI' | 'SMC' | 'RISK_GATE' | 'TRADE_DECISION' | 'BOOT'
        """
        now = datetime.now(timezone.utc)
        entry = {
            "id": f"log_{int(time.time()*1000)}_{len(self.logs)}",
            "timestamp": now.strftime("%H:%M:%S.%f")[:-3],
            "iso_time": now.isoformat(),
            "level": level,
            "category": category,
            "symbol": symbol,
            "message": message,
            "details": details or {}
        }
        self.logs.append(entry)
        return entry

    def get_recent_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        return list(self.logs)[-limit:]

    def clear(self):
        self.logs.clear()

brainstorm_service = BrainstormService()
