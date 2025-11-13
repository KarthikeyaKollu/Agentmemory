# memory_lib/__init__.py

# Export the main manager class
from .core.memory_manager import MemoryManager

# Export the core schemas
from .schemas import Message, UserMemory, MemoryAction, MemoryUpdatePlan

# Export the base interfaces for type-hinting or creating custom providers
from .interfaces import BaseModelProvider, BaseDbProvider

__all__ = [
    "MemoryManager",
    "Message",
    "UserMemory",
    "MemoryAction",
    "MemoryUpdatePlan",
    "BaseModelProvider",
    "BaseDbProvider"
]