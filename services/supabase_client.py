import os
from supabase import create_client, Client
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Initialize Supabase client
supabase: Client = None

if config.SUPABASE_URL and config.SUPABASE_KEY:
    try:
        supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
        print("✅ Supabase client initialized successfully")
    except Exception as e:
        print(f"⚠️ Failed to initialize Supabase client: {e}")
else:
    print("⚠️ SUPABASE_URL or SUPABASE_KEY not set in environment or config")

def get_supabase() -> Client:
    """Returns the initialized Supabase client."""
    if not supabase:
        raise ValueError("Supabase client is not initialized. Please configure SUPABASE_URL and SUPABASE_KEY.")
    return supabase
