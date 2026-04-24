from ..manager import ToolManager
from .core import RAGConfig, IVectorStore, IEmbeddingProvider
from .tools import SearchKnowledgeTool, IngestKnowledgeTool
from .stores.chromadb_store import ChromaDBVectorStore
from .stores.sqlite_store import SQLiteVectorStore
from .providers.embedders import OpenAIEmbedder, OllamaEmbedder, LocalEmbedder, MockEmbedder

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
    'ChromaDBVectorStore',
    'SQLiteVectorStore',
    'OpenAIEmbedder',
    'OllamaEmbedder',
    'LocalEmbedder',
    'MockEmbedder',
    'register_rag_suite',
]
