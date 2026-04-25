
import pytest
from agentic_core.tools.manager import ToolManager
from agentic_core.tools.rag import RAGConfig, IEmbeddingProvider, IVectorStore, register_rag_suite

# --- Mock Implementations ---

class MockEmbeddingProvider(IEmbeddingProvider):
    async def embed(self, texts):
        embeddings = []
        for text in texts:
            vec = [sum(ord(c) for c in word) for word in text.split()]
            vec = (vec + [0]*10)[:10]
            embeddings.append(vec)
        return embeddings

class MockVectorStore(IVectorStore):
    def __init__(self):
        self.data = []

    async def add(self, texts, embeddings, metadata):
        for t, e, m in zip(texts, embeddings, metadata):
            self.data.append({"text": t, "embedding": e, "metadata": m})

    async def search(self, query_embedding, top_k=3):
        scores = []
        for item in self.data:
            score = sum(a*b for a, b in zip(query_embedding, item["embedding"]))
            scores.append({"text": item["text"], "metadata": item["metadata"], "score": score})
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:top_k]

# --- Test Functions ---
@pytest.mark.asyncio
async def test_rag_registration():
    print("Running test_rag_registration...")
    tm = ToolManager()
    embedder = MockEmbeddingProvider()
    store = MockVectorStore()

    register_rag_suite(tm, embedder, store)

    # Use _plugins because ToolManager stores tools there
    assert "knowledge_search" in tm._plugins
    assert "knowledge_ingest" in tm._plugins
    assert "rag_suite" in tm.toolsets
    assert "knowledge_search" in tm.toolsets["rag_suite"]
    assert "knowledge_ingest" in tm.toolsets["rag_suite"]
    

@pytest.mark.asyncio
async def test_ingest_and_search():
    print("Running test_ingest_and_search...")
    tm = ToolManager()
    embedder = MockEmbeddingProvider()
    store = MockVectorStore()
    config = RAGConfig(chunk_size=100, chunk_overlap=10)

    register_rag_suite(tm, embedder, store, config)

    ingest_tool = tm._plugins["knowledge_ingest"]
    search_tool = tm._plugins["knowledge_search"]

    text = "The secret code is 12345. The agent name is Bond."
    source = "secret_doc.txt"
    result_ingest = await ingest_tool.execute({"text": text, "source": source}, {})

    assert "Successfully ingested" in result_ingest
    assert len(store.data) > 0

    result_search = await search_tool.execute({"query": "secret code", "top_k": 1}, {})
    assert "secret code" in result_search.lower()
    assert "secret_doc.txt" in result_search
    

@pytest.mark.asyncio
async def test_search_no_results():
    print("Running test_search_no_results...")
    tm = ToolManager()
    embedder = MockEmbeddingProvider()
    store = MockVectorStore() 

    register_rag_suite(tm, embedder, store)
    search_tool = tm._plugins["knowledge_search"]

    result = await search_tool.execute({"query": "something non-existent"}, {})
    assert "No relevant information found" in result
    