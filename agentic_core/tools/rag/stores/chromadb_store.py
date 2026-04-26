from typing import List, Dict, Any, Optional
import uuid

from agentic_core.config import ConfigurationError
from ..core import IVectorStore

try:
    import chromadb
except ImportError:
    raise ConfigurationError("ChromaDB is not installed. Please install with `pip install chromadb`")    
    

class ChromaDBVectorStore(IVectorStore):
    def __init__(
        self,
        persist_directory: str = None,
        collection_name: str = None,
        distance_metric: str = 'cosine'
    ):
        if collection_name is None:
            collection_name = f'rag_collection_{uuid.uuid4().hex[:8]}'
        
        self._collection_name = collection_name
        self._distance_metric = distance_metric
        
        if persist_directory:
            from chromadb.config import Settings
            client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
        else:
            client = chromadb.EphemeralClient()
        
        try:
            self._collection = client.get_collection(name=collection_name)
        except Exception:
            metadata = {'hnsw:space': distance_metric} if distance_metric else None
            self._collection = client.create_collection(
                name=collection_name,
                metadata=metadata,
                get_or_create=False
            )
    
    async def add(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]]
    ) -> None:
        ids = [str(uuid.uuid4()) for _ in texts]
        
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadata
        )
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        output = []
        if results['documents'] and len(results['documents']) > 0 and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                distance = results['distances'][0][i] if results.get('distances') and results['distances'][0] else 0
                metadata = results['metadatas'][0][i] if results.get('metadatas') and results['metadatas'][0] else {}
                
                if self._distance_metric == 'euclidean':
                    score = -distance
                else:
                    score = 1 - distance
                
                output.append({
                    'text': doc,
                    'metadata': metadata or {},
                    'score': score
                })
        
        return output
    
    def count(self) -> int:
        return self._collection.count()
    
    def delete(self) -> None:
        self._collection.delete(where={})
