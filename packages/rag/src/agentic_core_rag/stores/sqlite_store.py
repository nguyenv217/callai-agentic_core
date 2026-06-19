from typing import Any
import asyncio
import math

from agentic_core.config import ConfigurationError
from ..core import IVectorStore

try:
    from sqlalchemy import Column, String, JSON, Integer, Text, text
    from sqlalchemy.orm import declarative_base
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.ext.asyncio import async_sessionmaker
except ImportError:
    raise ConfigurationError('`sqlalchemy` is not installed. Please install with `pip install SQLAlchemy`.')
    
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
        
        self._init_lock = None
        self._initialized = False
        self._cached_chunk_ids = []
        self._cached_embeddings = None
    
    async def _ensure_initialized(self):
        if self._init_lock is None:
            self._init_lock = asyncio.Lock()
            
        if not self._initialized:
            async with self._init_lock:
                if not self._initialized:
                    async with self._engine.begin() as conn:
                        await conn.run_sync(Base.metadata.create_all)
                    await self._load_cache_locked()
                    self._initialized = True

    async def _load_cache_locked(self):
        import numpy as np
        async with self._session_factory() as session:
            from sqlalchemy import text
            result = await session.execute(
                text(f'SELECT chunk_id, embedding_bytes FROM {self.table_name}')
            )
            rows = result.fetchall()
            
        if rows:
            self._cached_chunk_ids = [r.chunk_id for r in rows]
            self._cached_embeddings = np.array([self._deserialize_vector(r.embedding_bytes) for r in rows])
        else:
            self._cached_chunk_ids = []
            self._cached_embeddings = np.empty((0, 0))
    
    @staticmethod
    def _serialize_vector(embedding: list[float]) -> str:
        import json
        return json.dumps(embedding)
    
    @staticmethod
    def _deserialize_vector(serialized: str) -> list[float]:
        import json
        return json.loads(serialized)
    
    async def add(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadata: list[dict[str, Any]]
    ) -> None:
        await self._ensure_initialized()
        
        import uuid
        import numpy as np
        new_ids = []
        async with self._session_factory() as session:
            for text_content, embedding, meta in zip(texts, embeddings, metadata):
                chunk_id = str(uuid.uuid4())
                new_ids.append(chunk_id)
                entry = VectorEntry(
                    chunk_id=chunk_id,
                    text=text_content,
                    meta_data=meta,
                    embedding_bytes=self._serialize_vector(embedding)
                )
                session.add(entry)
            await session.commit()
            
        async with self._init_lock:
            self._cached_chunk_ids.extend(new_ids)
            if self._cached_embeddings is None or self._cached_embeddings.size == 0:
                self._cached_embeddings = np.array(embeddings)
            else:
                self._cached_embeddings = np.vstack((self._cached_embeddings, embeddings))
    
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 3
    ) -> list[dict[str, Any]]:
        await self._ensure_initialized()
        import numpy as np
        
        if self._cached_embeddings is None or self._cached_embeddings.size == 0:
            return []

        embeddings = self._cached_embeddings
        query_vec = np.array(query_embedding)
        
        if self.distance_metric == 'euclidean':
            diff = embeddings - query_vec
            scores = -np.linalg.norm(diff, axis=1)
        else:
            dot_products = np.dot(embeddings, query_vec)
            norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec)
            # Compute cosine similarity, avoiding division by zero
            scores = np.divide(dot_products, norms, out=np.zeros_like(dot_products), where=norms!=0)
            
        # Extract the top_k indices efficiently
        n_results = min(top_k, len(scores))
        top_k_indices = np.argsort(scores)[::-1][:n_results]
        
        scored_ids = [
            (self._cached_chunk_ids[idx], float(scores[idx])) 
            for idx in top_k_indices
        ]

        if not scored_ids:
            return []

        # Fetch heavy text and metadata ONLY for the top_k
        top_chunk_ids = [s[0] for s in scored_ids]
        scores_map = {s[0]: s[1] for s in scored_ids}
        
        async with self._session_factory() as session:
            placeholders = ', '.join([f":id_{i}" for i in range(len(top_chunk_ids))]) # this prevent SQL injection
            params = {f"id_{i}": chunk_id for i, chunk_id in enumerate(top_chunk_ids)}
            
            query = text(f'SELECT chunk_id, text, meta_data FROM {self.table_name} WHERE chunk_id IN ({placeholders})')
            result = await session.execute(query, params)
            final_rows = result.fetchall()

        output = []
        row_lookup = {row.chunk_id: row for row in final_rows}
        for chunk_id in top_chunk_ids:
            row = row_lookup.get(chunk_id)
            if row:
                output.append({
                    'text': row.text,
                    'metadata': row.meta_data or {},
                    'score': scores_map[chunk_id]
                })
                
        return output
    
    async def count(self) -> int:
        await self._ensure_initialized()
        async with self._session_factory() as session:
            result = await session.execute(text(f'SELECT COUNT(*) FROM {self.table_name}'))
            return result.scalar()
    
    async def delete_all(self):
        await self._ensure_initialized()
        import numpy as np
        async with self._init_lock:
            async with self._session_factory() as session:
                await session.execute(text(f'DELETE FROM {self.table_name}'))
                await session.commit()
            self._cached_chunk_ids = []
            self._cached_embeddings = np.empty((0, 0))
    
    async def close(self):
        await self._engine.dispose()
