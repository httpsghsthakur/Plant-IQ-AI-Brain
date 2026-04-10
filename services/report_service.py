"""
PlantIQ AI Brain - Report Service
Report generation (daily/weekly/monthly).
"""
from datetime import datetime
from typing import Dict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class ReportService:
    """Generates formatted reports from model outputs."""

    def format_daily_report(self, report_data: Dict) -> str:
        """Format daily report data into a readable text report."""
        lines = []
        lines.append("=" * 60)
        lines.append(f"  PlantIQ AI Brain - Daily Report")
        lines.append(f"  {report_data.get('nursery_name', config.NURSERY_NAME)}")
        lines.append(f"  Date: {report_data.get('report_date', datetime.now().strftime('%Y-%m-%d'))}")
        lines.append("=" * 60)

        summary = report_data.get("summary", {})
        lines.append(f"\n[*] Summary:")
        lines.append(f"   Total Plants: {summary.get('total_plants', 'N/A')}")
        lines.append(f"   Healthy: {summary.get('healthy_plants_pct', 'N/A')}%")

        # Urgent actions
        urgent = report_data.get("urgent_actions", [])
        if urgent:
            lines.append(f"\n🚨 URGENT Actions ({len(urgent)}):")
            for action in urgent:
                lines.append(f"   ❗ [{action.get('category', '')}] {action.get('title', '')}")
                lines.append(f"      {action.get('description', '')}")

        # Important actions
        important = report_data.get("important_actions", [])
        if important:
            lines.append(f"\n[WARN]  Important Actions ({len(important)}):")
            for action in important:
                lines.append(f"   ⚡ [{action.get('category', '')}] {action.get('title', '')}")

        # Optimization
        optimization = report_data.get("optimization_actions", [])
        if optimization:
            lines.append(f"\n💡 Optimization Opportunities ({len(optimization)}):")
            for action in optimization:
                lines.append(f"   💰 {action.get('title', '')}: {action.get('impact', '')}")

        lines.append(f"\n{'=' * 60}")
        lines.append(f"  Generated: {report_data.get('generated_at', datetime.now().isoformat())}")
        lines.append("=" * 60)

        return "\n".join(lines)


# Singleton
report_service = ReportService()
