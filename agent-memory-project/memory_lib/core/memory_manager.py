from ..interfaces import BaseModelProvider, BaseDbProvider
from ..schemas import Message, UserMemory, MemoryUpdatePlan, MemoryAction
from typing import List

class MemoryManager:
    """
    The core class that manages memory by orchestrating
    the AI model and the Database.
    """
    def __init__(self, model: BaseModelProvider, db: BaseDbProvider):
        if not isinstance(model, BaseModelProvider):
            raise TypeError("model must be an instance of BaseModelProvider")
        if not isinstance(db, BaseDbProvider):
            raise TypeError("db must be an instance of BaseDbProvider")
            
        self.model = model
        self.db = db
        print("[MemoryManager] Initialized successfully.")

    def _build_prompt(self, user_id: str, new_message: str) -> List[Message]:
        """Builds the system prompt for the AI to make a memory decision."""
        print(f"[MemoryManager] Building prompt for user: {user_id}")
        existing_memories = self.db.get_memories(user_id)
        
        prompt = """
        You are a highly intelligent memory manager for an AI assistant.
        Your task is to analyze a new user message and the list of existing memories.
        You must decide what actions to take: ADD, UPDATE, or DELETE memories to
        keep the user's profile accurate and up-to-date.
        
        Follow these rules precisely:
        1.  **ADD**: Use this for new, distinct facts about the user (name, preferences, new events).
        2.  **UPDATE**: Use this to modify an existing memory with new, superseding information. 
            Provide the 'memory_id' of the memory to update. 
            (e.g., User moved from "Paris" to "Berlin").
        3.  **DELETE**: Use this ONLY if the user explicitly asks to forget something. 
            Provide the 'memory_id' of the memory to delete.
        4.  **NO ACTION**: If no new information or change is present, return an empty plan: {"plan": []}.

        ---
        EXISTING MEMORIES:
        """
        
        if not existing_memories:
            prompt += "<No memories exist yet.>\n"
        else:
            for mem in existing_memories:
                prompt += f"- [ID: {mem.memory_id}] {mem.content}\n"
        
        prompt += f"""
        ---
        NEW USER MESSAGE:
        "{new_message}"
        ---
        
        Based on this message and the existing memories, what is your memory update plan?
        """
        
        return [Message(role="system", content=prompt)]

    def process_message(self, user_id: str, new_message: str):
        """
        Processes a new message, updates memories, and returns the actions taken.
        This is the core agentic CRUD logic.
        """
        
        # 1. Build the prompt
        messages = self._build_prompt(user_id, new_message)
        
        # 2. Get structured response from AI
        print("[MemoryManager] Requesting memory plan from AI...")
        update_plan = self.model.get_structured_completion(messages, MemoryUpdatePlan)
        
        if not isinstance(update_plan, MemoryUpdatePlan):
            print(f"[MemoryManager] Error: Model did not return a valid MemoryUpdatePlan.")
            return

        print(f"[MemoryManager] Received plan with {len(update_plan.plan)} action(s).")
        
        # 3. Execute the plan
        for action in update_plan.plan:
            try:
                if action.action == "ADD":
                    if not action.content:
                        print("[MemoryManager] Skipping ADD: No content.")
                        continue
                    new_mem = UserMemory(user_id=user_id, content=action.content)
                    self.db.upsert_memory(new_mem)
                    
                elif action.action == "UPDATE":
                    if not action.memory_id or not action.content:
                        print("[MemoryManager] Skipping UPDATE: Missing ID or content.")
                        continue
                    mem = UserMemory(memory_id=action.memory_id, user_id=user_id, content=action.content)
                    self.db.upsert_memory(mem)
                    
                elif action.action == "DELETE":
                    if not action.memory_id:
                        print("[MemoryManager] Skipping DELETE: Missing ID.")
                        continue
                    self.db.delete_memory(action.memory_id)
            except Exception as e:
                print(f"[MemoryManager] Error executing action {action.action}: {e}")