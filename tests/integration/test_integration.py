import pytest
import asyncio
import openai
from pathlib import Path
from unittest.mock import patch
from src.core.repository import RepositoryManager
from src.core.vectorstore import VectorStore
from src.core.assistant import OpenAIAssistant
from src.utils.config import get_openai_key

# Setup OpenAI Mocking
OPENAI_API_KEY = get_openai_key()


@pytest.fixture(scope="module")
def test_vector_store():
    """Provide a fresh VectorStore instance for integration tests."""
    return VectorStore()


@pytest.fixture(scope="module")
def test_repository(tmp_path_factory):
    """Set up a temporary repository for testing."""
    tmp_repo = tmp_path_factory.mktemp("test_repo")
    repo_manager = RepositoryManager("https://github.com/git/git", tmp_repo, VectorStore())
    return repo_manager


@pytest.fixture(scope="module")
def test_assistant():
    """Initialize AssistantManager."""
    return OpenAIAssistant(OPENAI_API_KEY)


# ✅ **1️⃣ Test Repository Cloning & Indexing**
@pytest.mark.asyncio
async def test_clone_and_index_repository(test_repository):
    """Integration test for cloning and indexing a repository."""
    await test_repository.clone_repository()
    await test_repository.index_repository_files()

    # Ensure repository was cloned
    assert test_repository.clone_path.exists(), "Repository was not cloned!"
    assert any(test_repository.clone_path.iterdir()), "Repository is empty!"

    # Ensure files are indexed
    indexed_files = test_repository.list_files()
    assert len(indexed_files) > 0, "No files were indexed!"


# ✅ **2️⃣ Test Code Snippet Search in VectorStore**
@pytest.mark.asyncio
async def test_search_code_snippets(test_vector_store):
    """Integration test for searching indexed code snippets."""
    test_vector_store.add_text("def hello_world():\n    print('Hello World!')", {"filename": "test.py"})

    results = await asyncio.to_thread(test_vector_store.search, "hello_world", top_k=1)

    assert len(results) > 0, "No results returned from vector store!"
    assert "hello_world" in results[0][0]["text"], "Expected function not found!"


# ✅ **3️⃣ Test Querying Assistant API with Code Snippets**
@pytest.mark.asyncio
@patch("openai.AsyncOpenAI.embeddings.create", autospec=True)
async def test_assistant_query(mock_create, test_assistant, test_vector_store):
    """Test assistant integration by providing a user query with code context."""
    # Mock embedding response
    mock_create.return_value = {"data": [{"embedding": [0.1] * 1536}]}

    # Add some indexed code snippets
    test_vector_store.add_text("def sample_function():\n    return 42", {"filename": "sample.py"})

    # Query the assistant
    response = await test_assistant.query_assistant("Explain sample_function", test_vector_store)

    assert response is not None, "Assistant did not return a response!"
    assert "sample_function" in response, "Expected function reference missing!"


# ✅ **4️⃣ Test Rate Limiting and API Stability**
@pytest.mark.asyncio
@patch("openai.AsyncOpenAI.embeddings.create", autospec=True)
async def test_rate_limiting(mock_create, test_assistant):
    """Test rate limiting by making multiple OpenAI requests."""
    mock_create.return_value = {"data": [{"embedding": [0.1] * 1536}]}

    async def make_requests():
        return await test_assistant.get_embedding("test text")

    tasks = [make_requests() for _ in range(10)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    assert len(results) == 10, "Rate limit test failed!"
    assert not any(isinstance(r, Exception) for r in results), "Some requests failed due to rate limiting!"
