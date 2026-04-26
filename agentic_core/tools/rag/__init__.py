from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .stores.sqlite_store import SQLiteVectorStore
    from .stores.chromadb_store import ChromaDBVectorStore
    from .providers.embedders import LocalEmbedder, MockEmbedder, OllamaEmbedder, OpenAIEmbedder

from ..manager import ToolManager
from .core import RAGConfig, IVectorStore, IEmbeddingProvider
from .tools import RAGConfig, SearchKnowledgeTool, IngestKnowledgeTool

_LOOKUP = {
    "SQLiteVectorStore": ".stores.sqlite_store",
    "ChromaDBVectorStore": ".stores.chromadb_store",
    "OpenAIEmbedder": ".providers.embedders",
    "OllamaEmbedder": ".providers.embedders",
    "LocalEmbedder": ".providers.embedders",
    "MockEmbedder": ".providers.embedders"
}

def __getattr__(name):
    """
    Lazy-load the module when the attribute is accessed.
    This is used because some of the modules require optional dependencies.
    """
    if name in _LOOKUP:
        module_path = _LOOKUP[name]

        import importlib
        module = importlib.import_module(module_path, __package__)
        
        val = getattr(module, name)
        
        # Cache it in the global scope so next time is faster
        globals()[name] = val
        return val
    
    raise AttributeError(f"module {__name__} has no attribute {name}")


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
    'register_rag_suite',
]
