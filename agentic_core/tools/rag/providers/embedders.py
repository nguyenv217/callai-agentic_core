from typing import List, Protocol, Optional
from ..core import IEmbeddingProvider

class OpenAIEmbedder(IEmbeddingProvider):
    def __init__(
        self,
        api_key: str = None,
        model: str = 'text-embedding-3-small',
        dimensions: int = 1536
    ):
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError('openai package required. Install with: `pip install openai`')

        self._api_key = api_key
        self._model = model
        self._dimensions = dimensions
        self._client = None
    
    @property
    def client(self):
        from openai import AsyncOpenAI
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        response = await self.client.embeddings.create(
            model=self._model,
            input=texts,
            dimensions=self._dimensions
        )
        return [item.embedding for item in response.data]

class OllamaEmbedder(IEmbeddingProvider):
    def __init__(
        self,
        base_url: str = 'http://localhost:11434',
        model: str = 'nomic-embed-text'
    ):
        try:
            import ollama
        except ImportError:
            raise ImportError('ollama package required. Install with: `pip install ollama`')
        
        self._base_url = base_url
        self._model = model
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        import ollama
        embeddings = []
        for text in texts:
            response = ollama.embeddings(model=self._model, prompt=text, host=self._base_url)
            embeddings.append(response['embedding'])
        return embeddings

class LocalEmbedder(IEmbeddingProvider):
    def __init__(
        self,
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        device: str = 'cpu'
    ):
        self._model_name = model_name
        self._device = device
        self._model = None
        self._tokenizer = None
    
    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError('sentence-transformers required. Install with: `pip install sentence-transformers`')
            
            self._model = SentenceTransformer(self._model_name, device=self._device)
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        self._load_model()
        embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()

class MockEmbedder(IEmbeddingProvider):
    async def embed(self, texts: List[str]) -> List[List[float]]:
        import hashlib
        embeddings = []
        for text in texts:
            hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
            vec = [(hash_val >> (i * 4)) & 0xFF for i in range(64)]
            norm = sum(x * x for x in vec) ** 0.5
            if norm > 0:
                vec = [x / norm for x in vec]
            embeddings.append(vec)
        return embeddings
