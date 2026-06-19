# RAG Module Specification

The `agentic_core_rag` package provides interfaces and concrete implementations for embedding generation and vector storage. 

## 1. Installation

The module requires specific optional dependencies based on the chosen backend.

```bash
pip install "callai-agentic_core[rag-chroma]"
pip install "callai-agentic_core[rag-sqlite]"
pip install "callai-agentic_core[rag-openai]"
pip install "callai-agentic_core[rag-transformers]"
pip install "callai-agentic_core[rag-ollama]"
pip install "callai-agentic_core[rag-all]"
```

## 2. Core Interfaces

*   `IEmbeddingProvider`: Protocol requiring `async def embed(self, texts: list[str]) -> list[list[float]]`.
*   `IVectorStore`: Protocol requiring `add()`, `search()`, `count()`, and `delete_all()`.

## 3. SQLite Vector Store (In-Memory Cache)

The `SQLiteVectorStore` provides persistent storage backed by a SQLite file, but maintains a synchronized in-memory NumPy array for search operations. Search operations (`np.dot` / `np.linalg.norm`) execute in `O(1)` database read complexity.

```python
from agentic_core_rag import SQLiteVectorStore

store = SQLiteVectorStore(db_path="vectors.db", distance_metric="cosine")
# Valid metrics: 'cosine', 'euclidean'
```

## 4. Integration via AgentBuilder

The module provides two standard tools: `SearchKnowledgeTool` and `IngestKnowledgeTool`. 

```python
import asyncio
from agentic_core.agents import AgentBuilder
from agentic_core.config import RunnerConfig
from agentic_core_rag import (
    RAGConfig,
    OpenAIEmbedder,
    SQLiteVectorStore,
    SearchKnowledgeTool, 
    IngestKnowledgeTool
)

async def main():
    embedder = OpenAIEmbedder(api_key="sk-...")
    store = SQLiteVectorStore(db_path="my_knowledge.db")
    
    config = RAGConfig(
        chunk_size=1000,
        chunk_overlap=200,
        default_top_k=3,
        suite_prompt="Search the database for internal context before answering."
    )
    
    search_tool = SearchKnowledgeTool(store, embedder, config)
    ingest_tool = IngestKnowledgeTool(store, embedder, config)
    
    agent = AgentBuilder() \
        .with_provider_openai(api_key="sk-...") \
        .with_tools([search_tool, ingest_tool]) \
        .build()
    
    run_config = RunnerConfig(system_prompt=config.suite_prompt)

    result = await agent.run_turn(
        user_input="What is the remote work policy?",
        config=run_config
    )
    print(result.text)

if __name__ == "__main__":
    asyncio.run(main())
```