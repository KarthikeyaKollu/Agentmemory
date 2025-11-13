# 🧠 Agent Memory

## Overview

**Agent Memory** makes it easy to store, manage, and retrieve user-specific memories using different database providers.
It provides a simple API for handling user memories, including both **standard text memories** and **vector-based (semantic)** memories.

---

## ✨ Features

* Store and retrieve **user-specific memories**
* Supports both **normal** and **vector-based** memory formats
* Works with multiple database providers (e.g., SQLite, Chroma)
* Easily extendable and simple to use

---

## 🚀 Quick Start

### 📝 Example 1 — Normal Memory (SQLite)

This example shows how to store and retrieve text-based user memories using SQLite.

```python
from memory_lib.models.openai_provider import OpenAIProvider
from memory_lib.db.sqlite_provider import SqliteProvider
from memory_lib.core.memory_manager import MemoryManager

def main():
    # Initialize model and SQLite memory storage
    model = OpenAIProvider(model="gpt-4o-mini")
    db = SqliteProvider(db_path="my_agent_memory.db")

    # Create a memory manager
    manager = MemoryManager(model=model, db=db)
    user_id = "user_001"

    # Add and update memories
    manager.process_message(user_id, "Hi, my name is Alex and I live in Toronto.")
    manager.process_message(user_id, "I just moved to New York.")
    manager.process_message(user_id, "Please forget where I live.")

    # Retrieve memories
    memories = db.get_memories(user_id)
    print("User Memories:", memories or "<No memories found.>")

if __name__ == "__main__":
    main()
```

---

### 🧬 Example 2 — Vector Memory (Chroma)

This example shows how to use vector-based memory (semantic search).

```python
from memory_lib.models.openai_provider import OpenAIProvider
from memory_lib.embeddings.openai_embedder import OpenAIEmbedder
from memory_lib.db.chroma_provider import ChromaProvider
from memory_lib.core.vector_memory_manager import VectorMemoryManager

def main():
    # Initialize model, embedder, and vector DB
    model = OpenAIProvider(model="gpt-4o-mini")
    embedder = OpenAIEmbedder()
    vector_db = ChromaProvider(path="./vector_memory_db")

    # Create a vector-based memory manager
    manager = VectorMemoryManager(model=model, vector_db=vector_db, embedder=embedder)
    user_id = "user_vector_001"

    # Add and update memories
    manager.process_message(user_id, "Hi, my name is Chloe. I'm a marine biologist.")
    manager.process_message(user_id, "I live in San Diego and I love to surf.")
    manager.process_message(user_id, "I'm moving to Hawaii next month.")

    # Search for information
    results = manager.search(user_id, "Where does Chloe live?")
    print("Search Results:", results or "<No memories found.>")

if __name__ == "__main__":
    main()
```

---

## ⚙️ Helper Functions

### `print_memories(db, user_id)`

Retrieves and prints all memories for the given user.
If no memories exist, it prints `<No memories found.>`.

### `print_sql_memories(db, user_id)`

Same as above, but for **vector-based** databases.

---

## 🧩 Extending the Project

To support a new database or memory type:

1. Create a class that implements `BaseDbProvider`.
2. Implement required methods (like `get_memories(user_id)`).
3. Use it with `MemoryManager` or `VectorMemoryManager`.

---

## 🤝 Contributing

1. Fork this repository
2. Make your changes
3. Submit a pull request 🚀

---

## 📜 License

This project is licensed under the **MIT License**.
See the `LICENSE` file for details.
