import faiss
import numpy as np
import openai
from typing import List, Tuple
from src.utils.rate_limiter import get_rate_limiter
from src.utils.config import get_openai_key


# Initialize OpenAI client
client = openai.AsyncOpenAI(api_key=get_openai_key())


class VectorStore:
    def __init__(self, embedding_dim: int = 1536, index_file: str = "vectorstore.index"):
        self.embedding_dim = embedding_dim
        self.index_file = index_file
        self.metadata = []

        # Load index if available, otherwise create a new one
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.load_index()

    async def _get_embedding(self, text: str):
        """Get embeddings using OpenAI's correct async API client."""
        async with get_rate_limiter():
            response = await client.embeddings.create(
                model="text-embedding-3-small",
                input=[text]
            )
            return response.data[0].embedding

    async def add_text(self, text: str, metadata: dict):
        """Convert text to an embedding and add it to the FAISS index."""
        embedding = await self._get_embedding(text)
        embedding = np.array(embedding).reshape(1, -1)
        self.index.add(embedding)
        self.metadata.append(metadata)

    async def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find the top_k most similar code snippets to the query."""
        query_embedding = np.array(await self._get_embedding(query)).reshape(1, -1)
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx < len(self.metadata):  # Ensure index is valid
                results.append((self.metadata[idx]["text"], distances[0][i]))
        return results

    def save_index(self, path: str = "vectorstore.index"):
        """Save FAISS index to a file."""
        faiss.write_index(self.index, path)

    def load_index(self):
        """Load FAISS index from a file if available."""
        try:
            self.index = faiss.read_index(self.index_file)
            print("Loaded existing FAISS index.")
        except:
            print("No existing FAISS index found, creating a new one.")
