import json
import aiosqlite
from typing import Any
from agentic_core.interfaces import IPersistenceProvider

class SQLitePersistenceProvider(IPersistenceProvider):
    """Zero-config, local durable storage implementation."""
    def __init__(self, db_path: str = "agentic_persistence.db"):
        self.db_path = db_path
        
    async def _init_db(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                CREATE TABLE IF NOT EXISTS checkpoints (
                    session_id TEXT PRIMARY KEY,
                    state TEXT
                )
            ''')
            await db.execute('''
                CREATE TABLE IF NOT EXISTS node_results (
                    session_id TEXT,
                    node_id TEXT,
                    result TEXT,
                    PRIMARY KEY (session_id, node_id)
                )
            ''')
            await db.commit()

    async def save_checkpoint(self, session_id: str, state: dict[str, Any]) -> None:
        await self._init_db()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO checkpoints (session_id, state) VALUES (?, ?)",
                (session_id, json.dumps(state))
            )
            await db.commit()

    async def load_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        await self._init_db()
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT state FROM checkpoints WHERE session_id = ?", (session_id,)) as cursor:
                row = await cursor.fetchone()
                if row:
                    state = json.loads(row[0])
                    async with db.execute("SELECT node_id, result FROM node_results WHERE session_id = ?", (session_id,)) as cur2:
                        async for n_row in cur2:
                            node_id, res_str = n_row
                            state.setdefault("results", {})[node_id] = json.loads(res_str)
                    return state
        return None

    async def save_node_result(self, session_id: str, node_id: str, result: dict[str, Any]) -> None:
        await self._init_db()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO node_results (session_id, node_id, result) VALUES (?, ?, ?)",
                (session_id, node_id, json.dumps(result))
            )
            await db.commit()
