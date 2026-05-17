from typing import List, Dict
from agentic_core.tools import BaseTool
from .core import IVectorStore, IEmbeddingProvider, RAGConfig

class SearchKnowledgeTool(BaseTool):
    def __init__(self, store: IVectorStore, embedder: IEmbeddingProvider, config: RAGConfig):
        super().__init__()
        self._name = 'knowledge_search'
        self.store = store
        self.embedder = embedder
        self.config = config

        self._schema = {
            'type': 'function',
            'function': {
                'name': 'knowledge_search',
                'description': 'Searches the internal knowledge base for relevant information.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'query': {
                            'type': 'string', 
                            'description': 'The search query. Make it specific and keyword-rich.'
                        },
                        'top_k': {
                            'type': 'integer',
                            'description': 'Number of chunks to retrieve. Default is 3.'
                        }
                    },
                    'required': ['query']
                }
            }
        }

    async def execute(self, args: dict, context: dict) -> str:
        query = args.get('query')
        top_k = args.get('top_k', self.config.default_top_k)

        query_vector = (await self.embedder.embed([query]))[0]

        results = await self.store.search(query_vector, top_k=top_k)

        if not results:
            return 'No relevant information found in the knowledge base.'

        formatted_results = []
        for i, res in enumerate(results):
            metadata = res.get('metadata') or {}
            if isinstance(metadata, str):
                metadata = {'source': metadata}
            source = metadata.get('source', 'Unknown') if isinstance(metadata, dict) else 'Unknown'
            text = res['text']
            formatted_results.append(f'--- Result {i+1} (Source: {source}) ---\n{text}')

        return '\n\n'.join(formatted_results)


class IngestKnowledgeTool(BaseTool):
    def __init__(self, store: IVectorStore, embedder: IEmbeddingProvider, config: RAGConfig):
        super().__init__()
        self._name = 'knowledge_ingest'
        self.store = store
        self.embedder = embedder
        self.config = config

        self._schema = {
            'type': 'function',
            'function': {
                'name': 'knowledge_ingest',
                'description': 'Saves new information into the vector database for future retrieval.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'text': {'type': 'string', 'description': 'The information to save.'},
                        'source': {'type': 'string', 'description': 'Where this info came from (e.g., URL, filename).'}
                    },
                    'required': ['text', 'source']
                }
            }
        }

    def _chunk_text(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        words_per_chunk = self.config.chunk_size // 5 
        overlap = self.config.chunk_overlap // 5

        for i in range(0, len(words), words_per_chunk - overlap):
            chunk = ' '.join(words[i:i + words_per_chunk])
            if chunk: chunks.append(chunk)
        return chunks

    async def execute(self, args: dict, context: dict) -> str:
        text = args.get('text')
        source = args.get('source')

        chunks = self._chunk_text(text)
        embeddings = await self.embedder.embed(chunks)
        metadata = [{'source': source, 'chunk_index': i} for i in range(len(chunks))]

        await self.store.add(chunks, embeddings, metadata)
        return f'Successfully ingested {len(chunks)} chunks from source: {source}.'
