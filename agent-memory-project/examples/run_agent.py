import os
import sys
from dotenv import load_dotenv

# --- Add project root to Python path ---
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)
# ---------------------------------------

# 1. Import the main manager and the specific providers you want
from memory_lib import MemoryManager, BaseDbProvider
from memory_lib.models import OpenAIProvider
from memory_lib.db import SqliteProvider, PostgresProvider


def print_memories(db: BaseDbProvider, user_id: str):
    """Helper function to print current memories for a user."""
    print(f"\n--- Current Memories for {user_id} ---")
    memories = db.get_memories(user_id)
    if not memories:
        print("  <No memories found.>")
        return
    
    for mem in memories:
        print(f"  - [ID: {mem.memory_id[:8]}] {mem.content}")

def clear_all_memories(db: BaseDbProvider, user_id: str):
    """Helper to clear memory for a specific user."""
    print(f"Clearing all memories for {user_id}...")
    memories = db.get_memories(user_id)
    for mem in memories:
        db.delete_memory(mem.memory_id)

def run_demo(manager: MemoryManager, db: BaseDbProvider, user_id: str):
    """Runs the demo scenarios on a given manager."""
    
    # Start clean
    clear_all_memories(db, user_id)
    print_memories(db, user_id)
    
    # --- SCENARIO 1: ADD ---
    print("\n>>> Processing: 'Hi, my name is Alex and I live in Toronto.'")
    manager.process_message(user_id, "Hi, my name is Alex and I live in Toronto.")
    print_memories(db, user_id)

    # --- SCENARIO 2: UPDATE ---
    print("\n>>> Processing: 'I just moved to New York.'")
    manager.process_message(user_id, "I just moved to New York.")
    print_memories(db, user_id)
    
    # --- SCENARIO 3: DELETE ---
    print("\n>>> Processing: 'Please forget where I live.'")
    manager.process_message(user_id, "Please forget where I live.")
    print_memories(db, user_id)
    
    print("\nDemo finished.")

def main():
    # Load .env file from the project root
    load_dotenv(os.path.join(project_root, '.env'))
    
    # --- 2. Instantiate your providers directly ---
    try:
        # --- AI Model ---
        # You can swap this for "OpenAzure()" when you create it
        model = OpenAIProvider(model="gpt-4o-mini")
        
        # --- DB (Example 1: SQLite) ---
        print("\n" + "="*50)
        print("--- RUNNING DEMO WITH SQLITE ---")
        print("="*50)
        db_sqlite = SqliteProvider(db_path="my_agent_memory.db")
        
        # 3. Inject providers into the Manager
        manager_sqlite = MemoryManager(model=model, db=db_sqlite)
        
        # 4. Run the demo
        run_demo(manager_sqlite, db_sqlite, user_id="user_sqlite_001")

        
        # --- DB (Example 2: Postgres) ---
        # print("\n" + "="*50)
        # print("--- RUNNING DEMO WITH POSTGRES ---")
        # print("="*50)
        
        # # Get DB URL from .env or use a default
        # DB_URL = os.environ.get("DB_URL", "postgresql+psycopg://ai:ai@localhost:5532/ai")

        # db_postgres = PostgresProvider(connection_string=DB_URL)
        
        # # 3. Inject *different* DB provider into a *new* Manager
        # manager_postgres = MemoryManager(model=model, db=db_postgres)
        
        # # 4. Run the same demo
        # run_demo(manager_postgres, db_postgres, user_id="user_postgres_002")

    except Exception as e:
        print(f"\nFatal Error: {e}")
        print("Please check your .env file, API keys, and database connection strings.")
        print("If using Postgres, ensure the server is running and the database/user exists.")

if __name__ == "__main__":
    main()