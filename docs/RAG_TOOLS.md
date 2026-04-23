# RAG Tool Suite Guide

The RAG Suite allows you to give your agent a long-term memory or a domain-specific knowledge base using vector embeddings.

## Components

1. **Embedders**: Convert text to vectors.
   - `OpenAIEmbedder`: High-quality embeddings via OpenAI.
   - `LocalEmbedder`: Runs locally using \`sentence-transformers\`.
   - `OllamaEmbedder`: Uses local Ollama embedding models.
2. **Stores**: Store and retrieve vectors.
   - `SQLiteVectorStore`: Lightweight, file-based storage (Recommended for most cases).
   - `ChromaDBStore`: High-performance vector database.

## Quick Start Example

```python
import asyncio
from agentic_core.agents import chat
from agentic_core.tools.rag.core import RAGConfig
from agentic_core.tools.rag.providers.embedders import OpenAIEmbedder
from agentic_core.tools.rag.stores.sqlite_store import SQLiteVectorStore
from agentic_core.tools.rag.tools import SearchKnowledgeTool, IngestKnowledgeTool

async def main():
    # 1. Setup RAG components
    embedder = OpenAIEmbedder(api_key="sk-...")
    store = SQLiteVectorStore(db_path="my_knowledge.db")
    config = RAGConfig()
    
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

## Key Configuration Options (\`RAGConfig\`)

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | 1000 | Maximum characters per document chunk. |
| `chunk_overlap` | 200 | Overlap between chunks to preserve context. |
| `default_top_k` | 3 | Number of documents to retrieve per search. |
| `suite_prompt` | (Default string) | The system prompt injected to tell the agent how to use the RAG tool. |
