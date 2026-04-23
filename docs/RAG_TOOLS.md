# RAG Tool Suite Guide

The RAG Suite allows you to give your agent a long-term memory or a domain-specific knowledge base using vector embeddings.

### Installation
```bash
git clone https://github.com/nguyenv217/callai-agentic_core 
cd callai-agentic_core
pip install ".[rag-sqllite]" # or other built-in backend below
```

| Option | Description |
| --- | --- |
| `[rag-chromadb]` | Uses `ChromaDB` for vector storage. |
| `[rag-sqllite]` | Uses `sqlalchemy` for SQLite. **Note**: SQLite doesn't natively support vector indexing. |
| `[rag-openai]` | Enables OpenAI-compatible async embedder endpoints. |
| `[rag-transformer]` | Includes `SentenceTransformers` for local embedder usage. |
| `[rag-all]` | All of the above. |

## Components

1. **Embedders**: Convert text to vectors.
   - `OpenAIEmbedder`: High-quality embeddings via OpenAI.
   - `LocalEmbedder`: Runs locally using \`sentence-transformers\`.
   - `OllamaEmbedder`: Uses local Ollama embedding models.
2. **Stores**: Store and retrieve vectors.
   - `SQLiteVectorStore`: Lightweight, file-based storage (Recommended for most cases).
   - `ChromaDBStore`: High-performance vector database.

Or import your own backends by implementing `IEmbeddingProvider` and `IVectorStore`.

## Quick Start Example

```python
import asyncio
from agentic_core.agents import chat
from agentic_core.tools.rag import (
    RAGConfig,
    OpenAIEmbedder,
    SQLiteVectorStore,
    SearchKnowledgeTool, IngestKnowledgeTool
)

async def main():
    # 1. Setup RAG components
    embedder = OpenAIEmbedder(api_key="sk-...")
    store = SQLiteVectorStore(db_path="my_knowledge.db")
    config = RAGConfig(
        chunk_size=1000,
        distance_metric="cosine",
        suite_prompt=(
            "You have access to an internal knowledge base."
            "Retrieve facts with your search and ingest tool. Cite your answers."
            ) # This is injected into your base system prompt
    )
    
    # 2. Initialize the RAG Tools
    # These tools act as the bridge between the agent and the vector store
    search_tool = SearchKnowledgeTool(store, embedder, config)
    ingest_tool = IngestKnowledgeTool(store, embedder, config)
    
    # 3. Manually ingest some documents (or use the tool via the agent)
    # Use the ingest_tool directly to populate the DB before starting the agent
    await ingest_tool.execute(
        {"text": "The company's remote work policy allows 3 days at home.", "source": "HR_Handbook"},
        {}
    )
    
    # 4. Create an agent with RAG tools
    # Pass the tool schemas to the agent
    result = await chat(
        message="What is the remote work policy?",
        provider="openai",
        api_key="sk-...",
        tools=[search_tool.schema, ingest_tool.schema]
    )
    print(result)

asyncio.run(main())
```

**NOTE**: Each tool instance is stateful, i.e. if you populate a tool instance with data, this data is only available to that instance unless you provide a persistent database file.

## Key Configuration Options (`RAGConfig`)

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | 1000 | Maximum characters per document chunk. |
| `chunk_overlap` | 200 | Overlap between chunks to preserve context. |
| `default_top_k` | 3 | Number of documents to retrieve per search. |
| `suite_prompt` | (Default string) | The system prompt injected to tell the agent how to use the RAG tool. |
