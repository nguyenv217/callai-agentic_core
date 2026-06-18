# RAG Tool Suite Guide

The RAG Suite allows you to give your agent a long-term memory or a domain-specific knowledge base using vector embeddings.

### Installation
```bash
git clone https://github.com/nguyenv217/callai-agentic_core 
cd callai-agentic_core
pip install ".[rag-sqlite]" 
# or simply
pip install "callai-agentic_core[rag-sqlite]"
```

| Option | Description |
| --- | --- |
| `[rag-chroma]` | Uses `ChromaDB` for vector storage. |
| `[rag-sqlite]` | Uses `sqlalchemy` for SQLite. **Note**: SQLite doesn't natively support vector indexing. |
| `[rag-openai]` | Enables OpenAI-compatible async embedder endpoints. |
| `[rag-transformers]` | Includes `sentence-transformers` for local embedder usage. |
| `[rag-ollama]` | For local `Ollama` embedders. |
| `[rag-all]` | All of the above. |

## Components

1. **Embedders**: Convert text to vectors.
   - `OpenAIEmbedder`: High-quality embeddings via OpenAI.
   - `LocalEmbedder`: Runs locally using `sentence-transformers`.
   - `OllamaEmbedder`: Uses local Ollama embedding models.
2. **Stores**: Store and retrieve vectors.
   - `SQLiteVectorStore`: Lightweight, file-based storage.
   - `ChromaDBVectorStore`: High-performance vector database. (Recommended).

## Quick Start Example

### 1. Setup your database
Prepare your specific backend-compatible vector database. See `examples/example_RAG_index_script.py` for a ready-to-use indexing CLI script.

### 2. Initialize the RAG Tools
```python
import asyncio
from agentic_core.agents import create_openai_agent, chat
from agentic_core.config import RunnerConfig
from agentic_core_rag import (
    RAGConfig,
    OpenAIEmbedder,
    SQLiteVectorStore,
    SearchKnowledgeTool, 
    IngestKnowledgeTool
)

async def main():
    # 1. Setup RAG components
    embedder = OpenAIEmbedder(api_key="sk-...")
    store = SQLiteVectorStore(db_path="my_knowledge.db")
    config = RAGConfig(
        chunk_size=1000,
        distance_metric="cosine",
        suite_prompt="You have access to an internal knowledge base. Retrieve facts and cite your answers."
    )
    
    # 2. Initialize the RAG Tools
    search_tool = SearchKnowledgeTool(store, embedder, config)
    ingest_tool = IngestKnowledgeTool(store, embedder, config)
    
    # 3. Manually ingest some documents
    await ingest_tool.execute(
        {"text": "The company's remote work policy allows 3 days at home.", "source": "HR_Handbook"},
        {}
    )
    
    # 4. Create an agent and register tools
    runner = create_openai_agent(api_key="sk-...")
    runner.tools.register_tool(search_tool)
    runner.tools.register_tool(ingest_tool)
    
    run_config = RunnerConfig(
        tools=[search_tool.schema, ingest_tool.schema],
        system_prompt=config.suite_prompt
    )

    # 5. Execute
    result = await chat(
        message="What is the remote work policy?",
        runner=runner,
        config=run_config
    )
    print(result.response.text)

asyncio.run(main())
```

## Key Configuration Options (`RAGConfig`)

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | 1000 | Maximum characters per document chunk. |
| `chunk_overlap` | 200 | Overlap between chunks to preserve context. |
| `default_top_k` | 3 | Number of documents to retrieve per search. |
| `suite_prompt` | (String) | Instructions injected to tell the agent how to use the RAG tool. |
