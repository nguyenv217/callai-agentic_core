from __future__ import annotations
from typing import TYPE_CHECKING

from .providers.embedders import LocalEmbedder, MockEmbedder, OllamaEmbedder, OpenAIEmbedder, GeminiEmbedder
from agentic_core.tools.manager import ToolManager
from .core import RAGConfig, IVectorStore, IEmbeddingProvider
from .tools import RAGConfig, SearchKnowledgeTool, IngestKnowledgeTool

try:
    from .stores.sqlite_store import SQLiteVectorStore
except ImportError:
    class SQLiteVectorStore:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("SQLAlchemy and aiosqlite are required for SQLiteVectorStore. Install with `pip install callai-agentic_core[rag-sqlite]`")

try:
    from .stores.chromadb_store import ChromaDBVectorStore
except ImportError:
    class ChromaDBVectorStore:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("ChromaDB is required for ChromaDBVectorStore. Install with `pip install callai-agentic_core[rag-chroma]`")


def register_rag_suite(
    tool_manager: ToolManager, 
    embedder: IEmbeddingProvider, 
    store: IVectorStore, 
    config: RAGConfig = None
):
    config = config or RAGConfig()
    search_tool = SearchKnowledgeTool(store, embedder, config)
    ingest_tool = IngestKnowledgeTool(store, embedder, config)
    tool_manager.register_tool(search_tool)
    tool_manager.register_tool(ingest_tool)
    tool_manager.add_toolset(
        name='rag_suite',
        tools=[search_tool.name, ingest_tool.name],
        prompt=config.suite_prompt
    )

__all__ = [
    'RAGConfig',
    'IVectorStore',
    'IEmbeddingProvider',
    'SearchKnowledgeTool',
    'IngestKnowledgeTool',
    # === Preserve typehinting ===
    'ChromaDBVectorStore',
    'SQLiteVectorStore',
    'OpenAIEmbedder',
    'OllamaEmbedder',
    'LocalEmbedder',
    'MockEmbedder',
    'GeminiEmbedder',
    'register_rag_suite',
]
