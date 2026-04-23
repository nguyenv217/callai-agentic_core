from typing import List, Dict, Any, Optional
import asyncio
import math

from agentic_core.config import ConfigurationError

try:
    from sqlalchemy import Column, String, JSON, Integer, Text, text
    from sqlalchemy.orm import declarative_base
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.ext.asyncio import async_sessionmaker
except ImportError:
    raise ConfigurationError('`sqlalchemy` is not installed. Please install with `pip install SQLAlchemy`.')

from ..core import IVectorStore

Base = declarative_base()

class VectorEntry(Base):
    __tablename__ = 'vector_entries'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_id = Column(String(64), unique=True, nullable=False)
    text = Column(Text, nullable=False)
    meta_data = Column(JSON, nullable=True)
    embedding_bytes = Column(Text, nullable=False)

class SQLiteVectorStore(IVectorStore):
    def __init__(
        self,
        db_path: str = None,
        table_name: str = 'vector_entries',
        distance_metric: str = 'cosine'
    ):
        if db_path is None:
            db_path = 'rag_vectors.db'
        
        self.db_path = db_path
        self.table_name = table_name
        self.distance_metric = distance_metric
        
        async_url = f'sqlite+aiosqlite:///{db_path}'
        self._engine = create_async_engine(async_url, echo=False)
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        self._init_lock = asyncio.Lock()
        self._initialized = False
    
    async def _ensure_initialized(self):
        if not self._initialized:
            async with self._init_lock:
                if not self._initialized:
                    async with self._engine.begin() as conn:
                        await conn.run_sync(Base.metadata.create_all)
                    self._initialized = True
    
    @staticmethod
    def _serialize_vector(embedding: List[float]) -> str:
        import json
        return json.dumps(embedding)
    
    @staticmethod
    def _deserialize_vector(serialized: str) -> List[float]:
        import json
        return json.loads(serialized)
    
    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
    
    @staticmethod
    def _euclidean_distance(a: List[float], b: List[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    
    async def add(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]]
    ) -> None:
        await self._ensure_initialized()
        
        import uuid
        async with self._session_factory() as session:
            for text, embedding, meta in zip(texts, embeddings, metadata):
                entry = VectorEntry(
                    chunk_id=str(uuid.uuid4()),
                    text=text,
                    meta_data=meta,
                    embedding_bytes=self._serialize_vector(embedding)
                )
                session.add(entry)
            await session.commit()
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        await self._ensure_initialized()
        
        async with self._session_factory() as session:
            result = await session.execute(
                text(f'SELECT chunk_id, text, meta_data, embedding_bytes FROM {self.table_name}')
            )
            rows = result.fetchall()
        
        scored = []
        for row in rows:
            stored_embedding = self._deserialize_vector(row.embedding_bytes)
            
            if self.distance_metric == 'euclidean':
                score = -self._euclidean_distance(query_embedding, stored_embedding)
            else:
                score = self._cosine_similarity(query_embedding, stored_embedding)
            
            scored.append({
                'text': row.text,
                'metadata': row.meta_data or {},
                'score': score
            })
        
        scored.sort(key=lambda x: x['score'], reverse=True)
        return scored[:top_k]
    
    async def count(self) -> int:
        await self._ensure_initialized()
        async with self._session_factory() as session:
            result = await session.execute(text(f'SELECT COUNT(*) FROM {self.table_name}'))
            return result.scalar()
    
    async def delete_all(self) -> None:
        await self._ensure_initialized()
        async with self._session_factory() as session:
            await session.execute(text(f'DELETE FROM {self.table_name}'))
            await session.commit()
    
    async def close(self):
        await self._engine.dispose()
