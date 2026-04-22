from typing import Protocol, List, Dict, Any
from dataclasses import dataclass

@dataclass
class RAGConfig:
    """Configuration for the RAG Suite."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    default_top_k: int = 3
    distance_metric: str = "cosine"
    # System prompt injected when RAG tools are loaded
    suite_prompt: str = (
        "You have access to an internal knowledge base. "
        "Before answering domain-specific questions, ALWAYS use 'knowledge_search' to retrieve facts. "
        "Cite your sources based on the retrieved metadata."
    )

class IEmbeddingProvider(Protocol):
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Convert a list of strings into a list of vector embeddings."""
        ...

class IVectorStore(Protocol):
    async def add(self, texts: List[str], embeddings: List[List[float]], metadata: List[Dict[str, Any]]):
        """Store text chunks, their vectors, and metadata."""
        ...

    async def search(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """Return the top_k most similar chunks. Format: [{'text': str, 'metadata': dict, 'score': float}]"""
        ...
