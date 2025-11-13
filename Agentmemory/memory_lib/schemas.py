from pydantic import BaseModel, Field  # <-- CORRECT IMPORT
from typing import Any, Dict, List, Optional, Literal
from uuid import uuid4
from datetime import datetime

class Message(BaseModel):
    """A simple message object for the AI."""
    role: str
    content: str

class UserMemory(BaseModel):
    """Represents a single piece of memory in the database."""
    memory_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    content: str
    updated_at: datetime = Field(default_factory=datetime.now)

class MemoryAction(BaseModel):
    """A single action to modify memory."""
    action: Literal["ADD", "UPDATE", "DELETE"]
    memory_id: Optional[str] = Field(None, description="Required for UPDATE/DELETE actions.")
    content: Optional[str] = Field(None, description="Required for ADD/UPDATE actions.")

class MemoryUpdatePlan(BaseModel):
    """The full plan of actions the AI decides to take."""
    plan: List[MemoryAction] = Field(description="A list of 0 or more actions to modify memory.")




class VectorMemory(BaseModel):
    """
    Represents a memory item in the vector store.
    The 'payload' in ChromaDB will match this.
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    content: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RetrievedMemory(BaseModel):
    """A memory retrieved from the vector store, including its similarity score."""
    id: str
    content: str
    score: float
    user_id: str

class Fact(BaseModel):
    """A single fact extracted from a user message."""
    fact: str
    
class FactExtractPlan(BaseModel):
    """The list of facts extracted from a message."""
    facts: List[Fact] = Field(description="A list of 0 or more facts extracted from the user's message.")

class VectorMemoryAction(BaseModel):
    """A single action for the vector memory manager."""
    action: Literal["ADD", "UPDATE", "DELETE", "NONE"]
    id: Optional[str] = Field(None, description="The ID of the memory to UPDATE or DELETE.")
    content: Optional[str] = Field(None, description="The new content for an ADD or UPDATE action.")
    original_fact: str = Field(description="The new fact that prompted this action.")

class VectorMemoryUpdatePlan(BaseModel):
    """The full plan of actions for the vector memory."""
    plan: List[VectorMemoryAction] = Field(description="A list of actions to modify vector memory.")