import chromadb
from typing import List, Dict, Any
from ..interfaces import BaseVectorStore
from ..schemas import VectorMemory, RetrievedMemory
from datetime import datetime

class ChromaProvider(BaseVectorStore):
    """A concrete implementation of BaseVectorStore using ChromaDB."""
    
    def __init__(self, path: str = "./chroma_db", collection_name: str = "agent_memory"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        print(f"[ChromaProvider] Connected to collection '{collection_name}' at path: {path}")

    # --- NEW FUNCTION ---
    def get_all_memories(self, user_id: str) -> List[VectorMemory]:
        """Gets all memories for a user, without embeddings."""
        results = self.collection.get(where={"user_id": user_id})
        
        memories = []
        ids = results.get('ids', [])
        metadatas = results.get('metadatas', [])
        
        for i in range(len(ids)):
            meta = metadatas[i]
            memories.append(VectorMemory(
                id=ids[i],
                user_id=meta.get('user_id', user_id),
                content=meta.get('content', ''),
                created_at=datetime.fromisoformat(meta.get('created_at')),
                updated_at=datetime.fromisoformat(meta.get('updated_at')),
            ))
        return memories
    # --- END NEW FUNCTION ---

    def search(self, user_id: str, embedding: List[float], limit: int) -> List[RetrievedMemory]:
        """Search for similar memories for a user."""
        where_filter = {"user_id": user_id}

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=limit,
            where=where_filter
        )
        
        retrieved_memories = []
        ids = results.get('ids', [[]])[0]
        distances = results.get('distances', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        
        for i in range(len(ids)):
            retrieved_memories.append(
                RetrievedMemory(
                    id=ids[i],
                    score=distances[i],
                    content=metadatas[i].get('content', ''),
                    user_id=metadatas[i].get('user_id', '')
                )
            )
        return retrieved_memories

    def upsert(self, memory: VectorMemory, embedding: List[float]):
        """Create or update a memory in the vector store."""
        print(f"[ChromaProvider] UPSERT Memory: {memory.id}")
        
        payload = {
            "user_id": memory.user_id,
            "content": memory.content,
            "created_at": memory.created_at.isoformat(),
            "updated_at": memory.updated_at.isoformat(),
            **memory.metadata
        }
        
        self.collection.upsert(
            ids=[memory.id],
            embeddings=[embedding],
            metadatas=[payload]
        )

    def delete(self, memory_id: str):
        """Delete a memory by its ID."""
        print(f"[ChromaProvider] DELETE Memory: {memory_id}")
        try:
            self.collection.delete(ids=[memory_id])
        except Exception as e:
            print(f"[ChromaProvider] Error deleting {memory_id}: {e}. May not exist.")