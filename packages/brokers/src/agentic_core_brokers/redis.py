import json
import asyncio
from typing import Any
from agentic_core.interfaces import ITaskBroker

class RedisTaskBroker(ITaskBroker):
    """
    Redis-backed task queue for distributed GraphAgentRunner execution.
    Requires `redis.asyncio`.
    """
    def __init__(self, redis_url: str, queue_name: str = "agentic_tasks"):
        try:
            import redis.asyncio as redis
        except ImportError:
            raise ImportError("Please install redis: `pip install callai-agentic_core-brokers[redis]`")
        self.redis = redis.from_url(redis_url)
        self.queue_name = queue_name
        self._active_tasks = 0
        self._join_event = asyncio.Event()
        self._join_event.set()

    async def put(self, item: Any) -> None:
        priority, node_id = item
        await self.redis.zadd(self.queue_name, {json.dumps(node_id): priority})
        self._active_tasks += 1
        self._join_event.clear()

    async def get(self) -> Any:
        while True:
            result = await self.redis.zpopmin(self.queue_name, count=1)
            if result:
                node_id_json, priority = result[0]
                return (priority, json.loads(node_id_json))
            await asyncio.sleep(0.1)

    def task_done(self) -> None:
        self._active_tasks -= 1
        if self._active_tasks <= 0:
            self._join_event.set()

    async def join(self) -> None:
        await self._join_event.wait()
