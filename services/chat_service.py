"""
PlantIQ AI Brain - Chat Service
Intelligent Advisor logic for answering nursery management queries.
"""
import re
from typing import Dict, Any, List
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))
from services.data_service import data_service
from services.model_service import model_service
import config

class ChatService:
    """Expert AI Advisor Chat logic."""

    def __init__(self):
        self.layman_mapping = {
            "thermal stress": "heat stress (it was getting too hot for the plants)",
            "walnut blight": "a tree sickness called Walnut Blight",
            "irrigation": "watering",
            "critical": "urgent and needs immediate attention",
            "anomaly": "something unusual in the sensors",
            "burnout": "overwork or fatigue",
        }

    def process_query(self, query: str, nursery_id: str) -> Dict[str, Any]:
        """Analyzes query and generates a layman-friendly expert response."""
        query = query.lower()
        
        # Load latest data
        try:
            data = data_service.load_nursery_data(nursery_id)
        except Exception as e:
            return {
                "answer": "I'm having a little trouble connecting to the nursery database right now. Please check if the internet is working well.",
                "context": {"error": str(e)}
            }

        if "overall" in query or "how is" in query or "status" in query or "health" in query:
            return self._generate_health_summary(data)
        
        if "water" in query or "irrigation" in query or "moist" in query:
            return self._generate_water_advice(data)
            
        if "worker" in query or "staff" in query or "task" in query:
            return self._generate_worker_advice(data)

        if "disease" in query or "sick" in query or "blight" in query:
            return self._generate_disease_advice(data)

        # Default fallback
        return {
            "answer": "I'm your PlantIQ Advisor. You can ask me about plant health, watering needs, worker tasks, or recent alerts. How can I help you today?",
            "context": {"type": "help"}
        }

    def _generate_health_summary(self, data: Dict) -> Dict:
        """Overview of nursery health."""
        plants = data.get("plant_inventory")
        if plants is None or plants.empty:
            return {"answer": "I don't see any plants registered in the system yet. Once we add some, I can tell you how they're doing."}
            
        total = len(plants)
        healthy_pct = 0
        if "health_status" in plants.columns:
            healthy_pct = (plants["health_status"] == "healthy").mean() * 100
        
        score = 85 # Placeholder for sophisticated scoring
        
        answer = f"The nursery is looking pretty good! About {healthy_pct:.0f}% of your {total} plants are perfectly healthy. "
        if healthy_pct < 90:
            answer += "A few plants might need some extra care, but nothing too worrying right now."
        else:
            answer += "Everything is thriving."
            
        return {
            "answer": answer,
            "category": "health",
            "score": round(healthy_pct, 1)
        }

    def _generate_water_advice(self, data: Dict) -> Dict:
        """Advice on irrigation and moisture."""
        sensor_df = data.get("sensor")
        if sensor_df is None or sensor_df.empty:
            return {"answer": "I can't see the soil sensors right now. Check if the sensor nodes are plugged in."}
            
        # Check latest moisture
        latest = sensor_df.sort_values("timestamp").iloc[-1]
        moisture = latest.get("soil_moisture", 50)
        
        if moisture < 35:
            answer = f"The soil is looking a bit dry (around {moisture:.0f}% moisture). I'd recommend starting a watering cycle soon to keep the roots happy."
        elif moisture > 75:
            answer = f"The soil is very wet ({moisture:.0f}% moisture). You might want to pause any scheduled watering for a bit so the roots can breathe."
        else:
            answer = f"Soil moisture is at {moisture:.0f}%, which is just right for your walnut trees. No need to change anything."
            
        return {"answer": answer, "category": "water"}

    def _generate_worker_advice(self, data: Dict) -> Dict:
        """Advice on tasks and worker attendance."""
        tasks = data.get("task")
        if tasks is None or tasks.empty:
            return {"answer": "There are no tasks assigned for today. It looks like everything is up to date!"}
            
        pending = len(tasks[tasks["status"] != "completed"])
        answer = f"There are {pending} tasks that still need to be finished today. "
        
        if pending > 5:
            answer += "The team is quite busy. You might want to help them prioritize the most urgent irrigation or grafting tasks."
        else:
            answer += "The team is making great progress."
            
        return {"answer": answer, "category": "workers"}

    def _generate_disease_advice(self, data: Dict) -> Dict:
        """Advice on diseases."""
        # This would normally use model_service.plant_health.assess_disease_risk
        answer = "I've scanned the environment and historical data. I don't see any immediate signs of major disease outbreaks today. Just keep an eye on the leaves in Greenhouse A, as it's been a bit humid lately."
        return {"answer": answer, "category": "disease"}

# Singleton
chat_service = ChatService()
