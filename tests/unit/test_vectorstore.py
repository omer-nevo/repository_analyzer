import pytest
import numpy as np
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from src.core.vectorstore import VectorStore, client
from src.utils.async_utils import file_chunker


@pytest.fixture
def vector_store():
    """Fixture to provide a fresh VectorStore instance."""
    return VectorStore()


@pytest.mark.asyncio
async def test_file_chunker(tmp_path):
    """Test async file chunking for large files."""
    test_file = tmp_path / "test_file.txt"
    test_file.write_text("print('Hello')\n" * 100)  # Simulated large file

    chunks = []
    async for chunk in file_chunker(test_file, chunk_size=500):
        chunks.append(chunk)

    assert len(chunks) > 1  # Ensure multiple chunks are created
    assert all(len(chunk) <= 500 for chunk in chunks)  # Ensure chunk size constraint


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["print('Hello')", "import numpy"])
@patch("core.vectorstore.faiss.IndexFlatL2")
async def test_add_and_search(mock_faiss, vector_store, query):
    """Test adding text and searching with FAISS mock."""
    mock_index = MagicMock()
    mock_faiss.return_value = mock_index

    vector_store.index = mock_index  # ✅ Ensure vectorstore uses the mocked index
    mock_index.search.return_value = (np.array([[0.1]]), np.array([[0]]))  # ✅ Fake search results

    await vector_store.add_text("print('Hello World')", {"text": "print('Hello World')"})
    results = await vector_store.search(query, top_k=1)

    assert len(results) > 0  # ✅ Ensure at least one match
    assert isinstance(results[0], tuple) and isinstance(results[0][1], float)

    results = await vector_store.search(query, top_k=1)
    assert len(results) > 0  # Ensure at least one match
    assert isinstance(results[0], tuple) and isinstance(results[0][1], float)  # Should return text-distance pairs


@patch("openai.Embedding.acreate")
@pytest.mark.asyncio
async def test_get_embedding(mock_embedding, vector_store):
    """Test OpenAI API embedding generation with a mock."""
    mock_embedding.return_value = asyncio.Future()
    mock_embedding.return_value.set_result({"data": [{"embedding": [0.1] * 1536}]})

    result = await vector_store._get_embedding("test")
    assert len(result) == 1536  # Ensure embedding size matches


@pytest.mark.asyncio
@patch.object(client.embeddings, "create", new_callable=AsyncMock)  # ✅ Correct async patch
async def test_rate_limiting(mock_create, vector_store):
    """Test rate limiting by making multiple calls to OpenAI API."""

    # ✅ Correctly mock OpenAI API response
    mock_response = AsyncMock()
    mock_response.data = [{"embedding": [0.1] * 1536}]
    mock_create.return_value = mock_response

    async def make_requests():
        return await vector_store._get_embedding("rate_limit_test")

    tasks = [make_requests() for _ in range(10)]
    results = await asyncio.gather(*tasks)

    assert len(results) == 10  # ✅ Ensure all requests processed
    mock_create.assert_called()  # ✅ Ensure OpenAI API was called


@patch("core.vectorstore.faiss.write_index")
@patch("core.vectorstore.faiss.read_index")
def test_save_and_load_index(mock_read, mock_write, vector_store):
    """Test saving and loading FAISS index."""
    vector_store.save_index()
    mock_write.assert_called_once()

    vector_store.load_index()
    mock_read.assert_called_once()
