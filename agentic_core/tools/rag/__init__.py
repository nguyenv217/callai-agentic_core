from ..manager import ToolManager
from .core import RAGConfig, IVectorStore, IEmbeddingProvider

_LOOKUP = {
    "SQLiteVectorStore": ".stores.sqlite_store",
    "ChromaDBVectorStore": ".stores.chromadb_store",
    "OpenAIEmbedder": ".providers.embedders",
    "OllamaEmbedder": ".providers.embedders",
    "LocalEmbedder": ".providers.embedders",
    "MockEmbedder": ".providers.embedders"
}

def __getattr__(name):
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
    from .tools import SearchKnowledgeTool, IngestKnowledgeTool
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
