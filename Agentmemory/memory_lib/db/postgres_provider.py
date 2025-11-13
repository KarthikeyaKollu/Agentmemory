import psycopg2
from datetime import datetime
from typing import List, Optional
from ..interfaces import BaseDbProvider
from ..schemas import UserMemory

class PostgresProvider(BaseDbProvider):
    """A real implementation of the DB provider using PostgreSQL."""
    
    def __init__(self, connection_string: str):
        """
        Initializes the provider with a PostgreSQL connection string.
        Example: "postgresql://user:password@localhost:5432/mydb"
        """
        # We need to convert the 'postgresql://' URL to 'dbname=...'
        # This is a simple parser for the user's example format
        try:
            from urllib.parse import urlparse
            url = urlparse(connection_string.replace("postgresql+psycopg", "postgresql"))
            self.conn_string = (
                f"dbname='{url.path[1:]}' "
                f"user='{url.username}' "
                f"password='{url.password}' "
                f"host='{url.hostname}' "
                f"port='{url.port}'"
            )
        except Exception as e:
            print(f"[PostgresProvider] Could not parse connection string, trying as-is. Error: {e}")
            self.conn_string = connection_string # Fallback
            
        self._create_table()
        print(f"[PostgresProvider] Connected to DB: {url.hostname}:{url.port}")

    def _get_conn(self):
        """Establishes a new connection."""
        return psycopg2.connect(self.conn_string)

    def _create_table(self):
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                CREATE TABLE IF NOT EXISTS user_memories (
                    memory_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL
                )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON user_memories (user_id)")

    def get_memories(self, user_id: str) -> List[UserMemory]:
        """Gets all memories for a user, ordered by time."""
        memories = []
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT memory_id, user_id, content, updated_at FROM user_memories WHERE user_id = %s ORDER BY updated_at", 
                    (user_id,)
                )
                rows = cur.fetchall()
                for row in rows:
                    memories.append(UserMemory(
                        memory_id=row[0], 
                        user_id=row[1], 
                        content=row[2], 
                        updated_at=row[3]
                    ))
        return memories

    def upsert_memory(self, memory: UserMemory):
        """Creates or updates a memory using ON CONFLICT."""
        print(f"[PostgresProvider] UPSERT Memory: {memory.memory_id}")
        memory.updated_at = datetime.now()
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                INSERT INTO user_memories (memory_id, user_id, content, updated_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (memory_id) DO UPDATE SET
                    content = EXCLUDED.content,
                    updated_at = EXCLUDED.updated_at
                """, (memory.memory_id, memory.user_id, memory.content, memory.updated_at))

    def delete_memory(self, memory_id: str):
        """Deletes a memory."""
        print(f"[PostgresProvider] DELETE Memory: {memory_id}")
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM user_memories WHERE memory_id = %s", (memory_id,))