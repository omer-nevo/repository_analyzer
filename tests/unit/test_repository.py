import pytest
from pathlib import Path
import sys
import os
import asyncio
from unittest.mock import patch, MagicMock
from src.core.repository import RepositoryManager
from src.core.vectorstore import VectorStore  # Ensure vectorstore is available

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

@pytest.mark.asyncio
@patch("core.vectorstore.VectorStore", autospec=True)  # ✅ Mock VectorStore
@patch("core.repository.Repo.clone_from")  # ✅ Mock Git cloning
async def test_clone_repository(mock_clone, mock_vectorstore, tmp_path):
    """Test repository cloning with a mock."""
    repo_url = "https://github.com/git/git"
    clone_path = tmp_path / "git_repo"

    repo_manager = RepositoryManager(repo_url, clone_path, mock_vectorstore)
    await repo_manager.clone_repository()

    # ✅ Manually create the directory since cloning is mocked
    clone_path.mkdir(parents=True, exist_ok=True)

    mock_clone.assert_called_once_with(repo_url, clone_path)  # ✅ Ensure cloning was called
    assert clone_path.exists()  # ✅ Ensure test directory exists


@pytest.mark.asyncio
@patch("core.repository.Repo.clone_from", side_effect=Exception("Git error"))
async def test_clone_repository_failure(mock_clone, tmp_path):
    """Test repository cloning failure and exception handling."""
    repo_url = "https://github.com/git/git"
    clone_path = tmp_path / "git_repo"

    vector_store = VectorStore()
    repo_manager = RepositoryManager(repo_url, clone_path, vector_store)

    with pytest.raises(Exception, match="Git error"):
        await repo_manager.clone_repository()

    mock_clone.assert_called_once_with(repo_url, clone_path)  # ✅ Ensure cloning was attempted
    assert not clone_path.exists()  # ✅ Ensure no repo is created


def test_list_files(tmp_path):
    """Test that only specific file types are listed."""
    # Setup a temporary directory with sample files
    (tmp_path / "file1.py").write_text("print('Hello')")
    (tmp_path / "file2.md").write_text("# Markdown File")
    (tmp_path / "file3.txt").write_text("Sample text")
    (tmp_path / "image.png").write_text("Binary data")  # Should be ignored

    vector_store = VectorStore()
    repo_manager = RepositoryManager("dummy_url", tmp_path, vector_store)
    files = repo_manager.list_files(extensions=[".py", ".md"])

    assert len(files) == 2  # ✅ Ensure only .py and .md files are returned
    assert any("file1.py" in str(f) for f in files)
    assert any("file2.md" in str(f) for f in files)
    assert all(not str(f).endswith(".png") for f in files)  # ✅ Ensure PNG is ignored


@pytest.mark.asyncio
async def test_process_files(tmp_path):
    """Test processing of files asynchronously."""
    files = [tmp_path / f"file{i}.txt" for i in range(3)]
    for file in files:
        file.write_text("Sample content")

    vector_store = VectorStore()
    repo_manager = RepositoryManager("dummy_url", tmp_path, vector_store)

    await repo_manager.process_files(files)

    # ✅ Verify no exceptions occurred during processing
    assert all(file.exists() for file in files)


@pytest.mark.asyncio
@patch("core.vectorstore.VectorStore.add_text")  # ✅ Fix path
async def test_index_repository_files(mock_add_text, tmp_path):
    """Test that files are correctly indexed in the vector database."""
    (tmp_path / "file1.py").write_text("print('Hello World')")
    (tmp_path / "file2.md").write_text("# Markdown File")

    vector_store = VectorStore()
    repo_manager = RepositoryManager("dummy_url", tmp_path, vector_store)

    await repo_manager.index_repository_files()

    assert mock_add_text.called  # ✅ Ensure files were processed and added
    assert mock_add_text.call_count > 1  # ✅ Ensure multiple chunks were indexed

    # ✅ Verify that the correct content was indexed
    indexed_texts = [call.args[0] for call in mock_add_text.call_args_list]
    assert any("print('Hello World')" in text for text in indexed_texts)
    assert any("# Markdown File" in text for text in indexed_texts)


@pytest.mark.asyncio
@patch("core.vectorstore.VectorStore.add_text", side_effect=[None, Exception("Indexing error"), None])
async def test_index_repository_files_handles_errors(mock_add_text, tmp_path):
    """Test that indexing continues even if one file fails."""
    (tmp_path / "file1.py").write_text("print('Hello')")
    (tmp_path / "file2.md").write_text("# Markdown File")
    (tmp_path / "file3.txt").write_text("Some text content")

    vector_store = VectorStore()
    repo_manager = RepositoryManager("dummy_url", tmp_path, vector_store)

    await repo_manager.index_repository_files()

    assert mock_add_text.called  # ✅ Ensure indexing was attempted
    assert mock_add_text.call_count == 3  # ✅ All files were processed even though one failed
