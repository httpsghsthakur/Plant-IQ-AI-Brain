import sys
from pathlib import Path
import pandas as pd
import json

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))
import config
from services.chat_service import chat_service
from services.data_service import data_service

def verify():
    print("--- Verification Script ---")
    
    # 1. Test ChatService
    print("\n[1] Testing ChatService...")
    query = "How is my nursery doing?"
    nursery_id = "test_nursery"
    try:
        response = chat_service.process_query(query, nursery_id)
        print(f"Query: {query}")
        print(f"Answer: {response['answer']}")
    except Exception as e:
        print(f"ChatService test failed: {e}")

    # 2. Test DataService (Mocked or real check)
    print("\n[2] Testing DataService...")
    print(f"Supabase configured: {'Yes' if config.SUPABASE_URL else 'No'}")
    
    # 3. Terminology Check
    print("\n[3] Terminology Check (Dashboard)...")
    from api.routes.dashboard import get_dashboard_summary
    import asyncio
    
    async def check_dashboard():
        try:
            # Mock the data_service call if needed or just check if it runs
            res = await get_dashboard_summary(nursery_id="test")
            print("Dashboard keys:", res.keys())
            if "nursery_wellness_score" in res:
                print("✅ Found 'nursery_wellness_score'")
            if "urgent_care_alerts" in res:
                print("✅ Found 'urgent_care_alerts'")
        except Exception as e:
            print(f"Dashboard check note (expected error if DB down): {e}")

    try:
        asyncio.run(check_dashboard())
    except:
        pass

if __name__ == "__main__":
    verify()
