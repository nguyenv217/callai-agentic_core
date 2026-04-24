import asyncio
import uuid
import os
import tempfile
import pytest
import pytest_asyncio
import shutil

from agentic_core.tools.manager import ToolManager
from agentic_core.tools.rag import (
    RAGConfig,
    SQLiteVectorStore,
    ChromaDBVectorStore,
    MockEmbedder,
    register_rag_suite
)

@pytest.mark.asyncio
async def test_sqlite_vector_store_basic():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, 'test_rag.db')
    
    try:
        store = SQLiteVectorStore(db_path=db_path, distance_metric='cosine')
        
        texts = ['Hello world', 'Python is great', 'Machine learning is cool']
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        metadata = [{'source': f'doc_{i}'} for i in range(len(texts))]
        
        await store.add(texts, embeddings, metadata)
        
        count = await store.count()
        assert count == 3, f'Expected 3 entries, got {count}'
        
        results = await store.search([0.15, 0.25, 0.35], top_k=2)
        assert len(results) <= 2
        assert all('text' in r for r in results)
        assert all('metadata' in r for r in results)
        
        await store.close()
        print('test_sqlite_vector_store_basic: PASSED')
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_sqlite_vector_store_euclidean():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, 'test_rag_euclidean.db')
    
    try:
        store = SQLiteVectorStore(db_path=db_path, distance_metric='euclidean')
        
        texts = ['Apple fruit', 'Orange fruit', 'Car vehicle']
        embeddings = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]
        metadata = [{'type': 'fruit'} if i < 2 else {'type': 'vehicle'} for i in range(3)]
        
        await store.add(texts, embeddings, metadata)
        
        results = await store.search([0.95, 0.05], top_k=2)
        assert len(results) <= 2
        assert 'Apple' in results[0]['text'] or 'Orange' in results[0]['text']
        
        await store.close()
        print('test_sqlite_vector_store_euclidean: PASSED')
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_sqlite_persistence():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, 'persist_rag.db')
    
    try:
        store1 = SQLiteVectorStore(db_path=db_path)
        texts = ['Persistent data', 'Survives restart']
        embeddings = [[0.5, 0.5], [0.6, 0.4]]
        await store1.add(texts, embeddings, [{'source': 'p1'}, {'source': 'p2'}])
        await store1.close()
        
        store2 = SQLiteVectorStore(db_path=db_path)
        count = await store2.count()
        assert count == 2, f'Expected 2 after restart, got {count}'
        
        results = await store2.search([0.55, 0.45], top_k=1)
        assert len(results) >= 1
        await store2.close()
        print('test_sqlite_persistence: PASSED')
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_chroma_vector_store_basic():
    temp_dir = tempfile.mkdtemp()
    
    try:
        store = ChromaDBVectorStore(
            persist_directory=temp_dir,
            collection_name='test_collection',
            distance_metric='cosine'
        )
        
        texts = ['Chroma test one', 'Chroma test two', 'Chroma test three']
        embeddings = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 0.1, 0.2, 0.3]]
        metadata = [{'id': i} for i in range(3)]
        
        await store.add(texts, embeddings, metadata)
        
        count = store.count()
        assert count == 3, f'Expected 3, got {count}'
        
        results = await store.search([0.15, 0.25, 0.35, 0.45], top_k=2)
        assert len(results) <= 2
        assert all('text' in r and 'score' in r for r in results)
        
        print('test_chroma_vector_store_basic: PASSED')
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_full_rag_suite_sqlite():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, 'full_rag.db')
    
    try:
        tm = ToolManager()
        store = SQLiteVectorStore(db_path=db_path)
        embedder = MockEmbedder()
        config = RAGConfig(chunk_size=50, chunk_overlap=10)
        
        register_rag_suite(tm, embedder, store, config)
        
        assert 'knowledge_search' in tm._plugins
        assert 'knowledge_ingest' in tm._plugins
        assert 'rag_suite' in tm.toolsets
        
        ingest_tool = tm._plugins['knowledge_ingest']
        search_tool = tm._plugins['knowledge_search']
        
        result = await ingest_tool.execute(
            {'text': 'The capital of France is Paris. It has the Eiffel Tower.', 'source': 'geo.txt'},
            {}
        )
        assert 'Successfully ingested' in result
        
        result = await ingest_tool.execute(
            {'text': 'Python was created by Guido van Rossum in 1991.', 'source': 'python.txt'},
            {}
        )
        assert 'Successfully ingested' in result
        
        search_result = await search_tool.execute({'query': 'Eiffel Tower France', 'top_k': 2}, {})
        assert 'Eiffel' in search_result or 'France' in search_result
        assert 'geo.txt' in search_result
        
        await store.close()
        print('test_full_rag_suite_sqlite: PASSED')
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_full_rag_suite_chroma():
    temp_dir = tempfile.mkdtemp()
    
    try:
        tm = ToolManager()
        store = ChromaDBVectorStore(persist_directory=temp_dir, collection_name='full_test')
        embedder = MockEmbedder()
        config = RAGConfig(chunk_size=50, chunk_overlap=10)
        
        register_rag_suite(tm, embedder, store, config)
        
        ingest_tool = tm._plugins['knowledge_ingest']
        search_tool = tm._plugins['knowledge_search']
        
        await ingest_tool.execute(
            {'text': 'Quantum computing uses qubits instead of classical bits.', 'source': 'quantum.txt'},
            {}
        )
        
        search_result = await search_tool.execute({'query': 'quantum computing qubits', 'top_k': 1}, {})
        assert 'quantum' in search_result.lower() or 'qubit' in search_result.lower()
        
        print('test_full_rag_suite_chroma: PASSED')
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_chroma_persistence():
    temp_dir = tempfile.mkdtemp()
    collection_name = f'persist_test_{uuid.uuid4().hex[:8]}'
    
    try:
        store1 = ChromaDBVectorStore(
            persist_directory=temp_dir,
            collection_name=collection_name
        )
        await store1.add(['Persistent document'], [[0.1, 0.2, 0.3]], [{'source': 'persist'}])
        del store1
        
        store2 = ChromaDBVectorStore(
            persist_directory=temp_dir,
            collection_name=collection_name
        )
        count = store2.count()
        assert count >= 1, f'Expected at least 1 after restart, got {count}'
        
        print('test_chroma_persistence: PASSED')
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_no_results_handling():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, 'empty.db')
    
    try:
        tm = ToolManager()
        store = SQLiteVectorStore(db_path=db_path)
        embedder = MockEmbedder()
        register_rag_suite(tm, embedder, store)
        
        search_tool = tm._plugins['knowledge_search']
        result = await search_tool.execute({'query': 'nonexistent query'}, {})
        assert 'No relevant information found' in result
        
        await store.close()
        print('test_no_results_handling: PASSED')
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_batch_ingestion():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, 'batch.db')
    
    try:
        tm = ToolManager()
        store = SQLiteVectorStore(db_path=db_path)
        embedder = MockEmbedder()
        register_rag_suite(tm, embedder, store)
        
        ingest_tool = tm._plugins['knowledge_ingest']
        
        for i in range(5):
            result = await ingest_tool.execute(
                {'text': f'Document number {i} with some content about topic {i % 3}.', 'source': f'doc_{i}.txt'},
                {}
            )
            assert 'Successfully ingested' in result
        
        count = await store.count()
        assert count >= 5, f'Expected at least 5 chunks, got {count}'
        
        search_result = await tm._plugins['knowledge_search'].execute({'query': 'topic 1', 'top_k': 3}, {})
        assert len(search_result) > 0
        
        await store.close()
        print('test_batch_ingestion: PASSED')
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    import uuid
    pytest.main([__file__, '-v', '-s'])
