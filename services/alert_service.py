"""
PlantIQ AI Brain - Alert Service
Alert generation and management.
"""
from datetime import datetime
from typing import Dict, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class AlertService:
    """Manages alerts from all AI models."""

    def __init__(self):
        self._alerts: List[Dict] = []

    def add_alert(self, alert_type: str, severity: str, title: str,
                  message: str, source: str, zone_id: str = None):
        """Add a new alert."""
        self._alerts.append({
            "id": len(self._alerts) + 1,
            "type": alert_type,
            "severity": severity,
            "title": title,
            "message": message,
            "source": source,
            "zone_id": zone_id,
            "acknowledged": False,
            "created_at": datetime.now().isoformat(),
        })

    def get_active_alerts(self, severity: str = None) -> List[Dict]:
        """Get all unacknowledged alerts."""
        alerts = [a for a in self._alerts if not a["acknowledged"]]
        if severity:
            alerts = [a for a in alerts if a["severity"] == severity]
        return sorted(alerts, key=lambda x: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(x["severity"], 4))

    def acknowledge_alert(self, alert_id: int) -> bool:
        """Mark an alert as acknowledged."""
        for alert in self._alerts:
            if alert["id"] == alert_id:
                alert["acknowledged"] = True
                alert["acknowledged_at"] = datetime.now().isoformat()
                return True
        return False

    def get_alert_summary(self) -> Dict:
        """Get alert count summary."""
        active = [a for a in self._alerts if not a["acknowledged"]]
        return {
            "total_active": len(active),
            "critical": sum(1 for a in active if a["severity"] == "critical"),
            "high": sum(1 for a in active if a["severity"] == "high"),
            "medium": sum(1 for a in active if a["severity"] == "medium"),
            "low": sum(1 for a in active if a["severity"] == "low"),
        }


# Singleton
alert_service = AlertService()
