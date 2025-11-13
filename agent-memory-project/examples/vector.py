import os
import sys
from dotenv import load_dotenv
from typing import Literal

# --- Add project root to Python path ---
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)
# ---------------------------------------

# Import the main manager classes
from memory_lib.core.vector_memory import VectorMemoryManager

# Import the providers we will inject
from memory_lib.models import OpenAIProvider, OpenAIEmbedder
from memory_lib.db import SqliteProvider, ChromaProvider
from memory_lib.interfaces import BaseDbProvider


# --- Helper for Traditional (SQL) Demo ---
def print_sql_memories(db: BaseDbProvider, user_id: str):
    print(f"\n--- Current SQL Memories for {user_id} ---")
    memories = db.get_memories(user_id)
    if not memories:
        print("  <No memories found.>")
        return
    for mem in memories:
        print(f"  - [ID: {mem.memory_id[:8]}] {mem.content}")



# --- Helper for Vector (Chroma) Demo ---
def print_vector_memories(manager: VectorMemoryManager, user_id: str, query: str):
    print(f"\n--- Searching Vector Memories for '{query}' (User: {user_id}) ---")
    memories = manager.search(user_id, query, limit=1)
    if not memories:
        print("  <No relevant memories found.>")
        return
    for mem in memories:
        print(f"  - [Score: {mem.score:.4f}] [ID: {mem.id[:8]}] {mem.content}")


def main():
    load_dotenv(os.path.join(project_root, '.env'))
    
    # --- 1. Instantiate ALL providers ---
    try:
        model = OpenAIProvider(model="gpt-4o-mini")
        embedder = OpenAIEmbedder()
        vector_db = ChromaProvider(path="./vector_memory_db")

    except Exception as e:
        print(f"Fatal Error: {e}")
        print("Please check your .env file and API keys.")
        return


    print("\n" + "="*50)
    print("--- RUNNING DEMO WITH 'VECTOR' (mem0-style) MANAGER ---")
    print("="*50)
    
    # 2. Inject Model, Embedder, and Vector DB into the new manager
    vector_manager = VectorMemoryManager(model=model, vector_db=vector_db, embedder=embedder)
    USER_ID_VECTOR = "user_vector_002"
    
    # 3. Clear old memories
    old_memories = vector_manager.search(USER_ID_VECTOR, "find all memories", limit=100)
    for mem in old_memories:
        vector_db.delete(mem.id)
    
    print_vector_memories(vector_manager, USER_ID_VECTOR, "where user lives")
    
    # --- SCENARIO 1: ADD ---
    print("\n>>> Processing: 'Hi, my name is Chloe. I'm a marine biologist.'")
    vector_manager.process_message(USER_ID_VECTOR, "Hi, my name is Chloe. I'm a marine biologist.")
    print_vector_memories(vector_manager, USER_ID_VECTOR, "who is chloe?")

    # --- SCENARIO 2: ADD ---
    print("\n>>> Processing: 'I live in San Diego and I love to surf.'")
    vector_manager.process_message(USER_ID_VECTOR, "I live in San Diego and I love to surf.")
    print_vector_memories(vector_manager, USER_ID_VECTOR, "where does chloe live?")
    
    # --- SCENARIO 3: UPDATE ---
    print("\n>>> Processing: 'I'm actually moving to Hawaii next month.'")
    vector_manager.process_message(USER_ID_VECTOR, "I'm actually moving to Hawaii next month.")
    print_vector_memories(vector_manager, USER_ID_VECTOR, "where does chloe live?")
    
    
    # --- SCENARIO 4: NO ACTION (Redundant) ---
    print("\n>>> Processing: 'Just a reminder, I'm a biologist.'")
    vector_manager.process_message(USER_ID_VECTOR, "Just a reminder, I'm a biologist.")
    print_vector_memories(vector_manager, USER_ID_VECTOR, "what does chloe do?")

if __name__ == "__main__":
    main()