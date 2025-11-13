from abc import ABC, abstractmethod
from typing import List, Type
from .schemas import Message, UserMemory, BaseModel, VectorMemory, RetrievedMemory

class BaseModelProvider(ABC):
    """Interface for any AI model provider."""
    @abstractmethod
    def get_structured_completion(self, messages: List[Message], output_model: Type[BaseModel]) -> BaseModel:
        """
        Gets a structured JSON response from the AI,
        parsed into the given Pydantic model.
        """
        pass

class BaseDbProvider(ABC):
    """Interface for any database provider."""
    @abstractmethod
    def get_memories(self, user_id: str) -> List[UserMemory]:
        """Get all memories for a specific user."""
        pass

    @abstractmethod
    def upsert_memory(self, memory: UserMemory):
        """Create a new memory or update an existing one."""
        pass
    
    @abstractmethod
    def delete_memory(self, memory_id: str):
        """Delete a memory by its ID."""
        pass



class BaseEmbedder(ABC):
    """Interface for any embedding model."""
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embeds a single string of text."""
        pass

class BaseVectorStore(ABC):
    """Interface for any vector database provider."""
    @abstractmethod
    def search(self, user_id: str, embedding: List[float], limit: int) -> List[RetrievedMemory]:
        """Search for similar memories for a user."""
        pass
    
    @abstractmethod
    def upsert(self, memory: VectorMemory, embedding: List[float]):
        """Create or update a memory in the vector store."""
        pass

    @abstractmethod
    def delete(self, memory_id: str):
        """Delete a memory by its ID."""
        pass
    
    @abstractmethod
    def get_all_memories(self, user_id: str) -> List[VectorMemory]:
        """Gets all memories for a user, without embeddings."""
        pass