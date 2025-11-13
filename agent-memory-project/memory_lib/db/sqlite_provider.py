import sqlite3
import json
from datetime import datetime
from typing import List
from ..interfaces import BaseDbProvider
from ..schemas import UserMemory

class SqliteProvider(BaseDbProvider):
    """A real implementation of the DB provider using SQLite."""
    def __init__(self, db_path: str = "agent_memory.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_table()
        print(f"[SqliteProvider] Connected to DB: {db_path}")

    def _create_table(self):
        with self.conn:
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS user_memories (
                memory_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """)
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON user_memories (user_id)")

    def get_memories(self, user_id: str) -> List[UserMemory]:
        """Gets all memories for a user, ordered by time."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT memory_id, user_id, content, updated_at FROM user_memories WHERE user_id = ? ORDER BY updated_at", 
            (user_id,)
        )
        rows = cursor.fetchall()
        return [UserMemory(
            memory_id=row[0], 
            user_id=row[1], 
            content=row[2], 
            updated_at=datetime.fromisoformat(row[3])
        ) for row in rows]

    def upsert_memory(self, memory: UserMemory):
        """Creates or updates a memory."""
        print(f"[SqliteProvider] UPSERT Memory: {memory.memory_id}")
        memory.updated_at = datetime.now() # Always update timestamp on write
        with self.conn:
            self.conn.execute("""
            INSERT INTO user_memories (memory_id, user_id, content, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                content=excluded.content,
                updated_at=excluded.updated_at
            """, (memory.memory_id, memory.user_id, memory.content, memory.updated_at.isoformat()))

    def delete_memory(self, memory_id: str):
        """Deletes a memory."""
        print(f"[SqliteProvider] DELETE Memory: {memory_id}")
        with self.conn:
            self.conn.execute("DELETE FROM user_memories WHERE memory_id = ?", (memory_id,))