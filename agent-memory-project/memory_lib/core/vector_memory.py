from ..interfaces import BaseModelProvider, BaseVectorStore, BaseEmbedder
from ..schemas import (
    Message, VectorMemory, RetrievedMemory, FactExtractPlan, 
    VectorMemoryAction, VectorMemoryUpdatePlan
)
from typing import Dict, List, Optional
from datetime import datetime

class VectorMemoryManager:
    """
    Manages "infinite" memory using a vector store and an "extract-search-merge"
    pattern, inspired by mem0.
    """
    def __init__(self, 
                 model: BaseModelProvider, 
                 vector_db: BaseVectorStore, 
                 embedder: BaseEmbedder,
                 search_limit: int = 3):
        
        if not isinstance(model, BaseModelProvider):
            raise TypeError("model must be an instance of BaseModelProvider")
        if not isinstance(vector_db, BaseVectorStore):
            raise TypeError("vector_db must be an instance of BaseVectorStore")
        if not isinstance(embedder, BaseEmbedder):
            raise TypeError("embedder must be an instance of BaseEmbedder")
            
        self.model = model
        self.vector_db = vector_db
        self.embedder = embedder
        self.search_limit = search_limit
        print("[VectorMemoryManager] Initialized successfully.")

    def _extract_facts(self, new_message: str) -> List[str]:
        """Step 1: Use LLM to extract new facts from the message."""
        print(f"[VectorMemoryManager] Step 1: Extracting facts from message...")
        prompt = f"""
        You are an information extractor. Extract all key facts, statements, or
        preferences from the following user message.
        
        USER MESSAGE:
        "{new_message}"
        
        Return ONLY a JSON object matching the FactExtractPlan schema.
        """
        messages = [Message(role="system", content=prompt)]
        
        try:
            response = self.model.get_structured_completion(messages, FactExtractPlan)
            if isinstance(response, FactExtractPlan):
                facts = [fact.fact for fact in response.facts]
                print(f"[VectorMemoryManager] Extracted facts: {facts}")
                return facts
        except Exception as e:
            print(f"[VectorMemoryManager] Error extracting facts: {e}")
            
        return [] # Return empty list on failure

    def _search_relevant_memories(self, user_id: str, facts: List[str]) -> List[RetrievedMemory]:
        """Step 2: Embed facts and search for relevant memories."""
        print(f"[VectorMemoryManager] Step 2: Searching for relevant memories...")
        all_retrieved: Dict[str, RetrievedMemory] = {}
        
        for fact in facts:
            embedding = self.embedder.embed_text(fact)
            results = self.vector_db.search(user_id, embedding, self.search_limit)
            for res in results:
                # Use a dict to automatically de-duplicate by memory ID
                all_retrieved[res.id] = res
                
        retrieved_list = list(all_retrieved.values())
        print(f"[VectorMemoryManager] Found {len(retrieved_list)} relevant memories.")
        return retrieved_list

    def _get_memory_update_plan(self, new_facts: List[str], old_memories: List[RetrievedMemory]) -> VectorMemoryUpdatePlan:
        """Step 3: Ask LLM to merge new facts and old memories into a plan."""
        print(f"[VectorMemoryManager] Step 3: Generating memory update plan...")
        
        # --- [FIX 1] --- Updated Prompt ---
        prompt = """
        You are a memory consolidation expert. Your job is to merge new facts with
        existing memories to create an accurate and non-redundant memory profile.
        
        You must return a 'plan' of actions: ADD, UPDATE, DELETE, or NONE.
        
        RULES:
        -   **ADD**: Use for a `new_fact` not covered by `old_memory`. 
            The 'content' field MUST be the new memory to add (e.g., "User lives in San Diego").
        -   **UPDATE**: Use when a `new_fact` supersedes an `old_memory`. 
            Provide the 'id' of the old memory and the new, consolidated 'content' (e.g., "Chloe is a marine biologist.").
        -   **DELETE**: Use ONLY if a `new_fact` explicitly states to forget 
            an `old_memory`. Provide the 'id' of the memory to delete.
        -   **NONE**: Use if a `new_fact` is already perfectly covered by an
            `old_memory` and no action is needed.
        
        ---
        EXISTING MEMORIES (from search):
        """
        
        if not old_memories:
            prompt += "<No relevant memories found.>\n"
        else:
            for mem in old_memories:
                prompt += f"- [ID: {mem.id}] {mem.content}\n"
        
        prompt += f"""
        ---
        NEW FACTS (from message):
        """
        if not new_facts:
            prompt += "<No new facts extracted.>\n"
        else:
            for fact in new_facts:
                prompt += f"- {fact}\n"
        
        prompt += """
        ---
        Based on these, generate the memory update plan.
        Provide ONLY a JSON object matching the VectorMemoryUpdatePlan schema.
        """
        # --- [END FIX 1] ---
        
        messages = [Message(role="system", content=prompt)]
        
        try:
            response = self.model.get_structured_completion(messages, VectorMemoryUpdatePlan)
            if isinstance(response, VectorMemoryUpdatePlan):
                print(f"[VectorMemoryManager] Plan received: {response.model_dump_json(indent=2)}")
                return response
        except Exception as e:
            print(f"[VectorMemoryManager] Error getting update plan: {e}")
            
        return VectorMemoryUpdatePlan(plan=[]) # Return empty plan on failure

    def _execute_plan(self, user_id: str, plan: VectorMemoryUpdatePlan):
        """Step 4: Execute the ADD, UPDATE, or DELETE actions."""
        print(f"[VectorMemoryManager] Step 4: Executing {len(plan.plan)} actions...")
        for action in plan.plan:
            try:
                if action.action == "ADD":
                    # --- [FIX 2] --- Fallback Logic ---
                    # If AI provides null content, fall back to the original fact.
                    content_to_add = action.content or action.original_fact 
                    if not content_to_add:
                        print(f"[MemoryManager] Skipping ADD: No content or original_fact for '{action.original_fact}'")
                        continue
                    # --- [END FIX 2] ---
                        
                    new_mem = VectorMemory(user_id=user_id, content=content_to_add)
                    embedding = self.embedder.embed_text(new_mem.content)
                    self.vector_db.upsert(new_mem, embedding)
                
                elif action.action == "UPDATE":
                    if not action.id or not action.content:
                        print(f"[MemoryManager] Skipping UPDATE: Missing ID or content for '{action.original_fact}'")
                        continue
                    
                    updated_mem = VectorMemory(
                        id=action.id, 
                        user_id=user_id, 
                        content=action.content, 
                        updated_at=datetime.now()
                    )
                    embedding = self.embedder.embed_text(updated_mem.content)
                    self.vector_db.upsert(updated_mem, embedding)
                
                elif action.action == "DELETE":
                    if not action.id:
                        print(f"[MemoryManager] Skipping DELETE: Missing ID for '{action.original_fact}'")
                        continue
                    self.vector_db.delete(action.id)
                
                elif action.action == "NONE":
                    print(f"[VectorMemoryManager] Action: NONE for '{action.original_fact}'")
                    
            except Exception as e:
                print(f"[VectorMemoryManager] Error executing action {action.action}: {e}")

    def process_message(self, user_id: str, new_message: str):
        """
        Processes a new message using the full vector memory pipeline.
        """
        # Step 1: Extract facts from the new message
        new_facts = self._extract_facts(new_message)
        if not new_facts:
            print("[VectorMemoryManager] No facts extracted. Nothing to do.")
            return

        # Step 2: Search for relevant memories
        relevant_memories = self._search_relevant_memories(user_id, new_facts)
        
        # Step 3: Get an update plan from the LLM
        update_plan = self._get_memory_update_plan(new_facts, relevant_memories)
        
        # Step 4: Execute the plan
        self._execute_plan(user_id, update_plan)

    def search(self, user_id: str, query: str, limit: int = 5) -> List[RetrievedMemory]:
        """Directly search the vector memory for a user."""
        print(f"[VectorMemoryManager] Performing direct search for user '{user_id}'...")
        embedding = self.embedder.embed_text(query)
        return self.vector_db.search(user_id, embedding, limit)