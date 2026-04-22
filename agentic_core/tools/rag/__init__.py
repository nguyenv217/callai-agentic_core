from ..manager import ToolManager
from .core import RAGConfig, IVectorStore, IEmbeddingProvider
from .tools import SearchKnowledgeTool, IngestKnowledgeTool

def register_rag_suite(
    tool_manager: ToolManager, 
    embedder: IEmbeddingProvider, 
    store: IVectorStore, 
    config: RAGConfig = None
):
    """
    Registers the RAG tools into the ToolManager and maps them to a toolset.
    """
    config = config or RAGConfig()

    # Instantiate tools
    search_tool = SearchKnowledgeTool(store, embedder, config)
    ingest_tool = IngestKnowledgeTool(store, embedder, config)

    # Register with the manager
    tool_manager.register_tool(search_tool)
    tool_manager.register_tool(ingest_tool)

    # Create a dedicated RAG toolset with its own instructional prompt
    tool_manager.add_toolset(
        name="rag_suite",
        tools=[search_tool.name, ingest_tool.name],
        prompt=config.suite_prompt
    )
