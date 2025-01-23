import os
import sys
import signal
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from git import Repo, GitCommandError
from src.core.vectorstore import VectorStore
from src.utils.async_utils import file_chunker


async def shutdown(signal, loop):
    """Handles shutdown for graceful exit."""
    print(f"Received exit signal {signal.name}... Shutting down.")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    # Log Errors
    for task, result in zip(tasks, results):
        if isinstance(result, Exception):
            print(f"Error in task {task.get_name()}: {result}")
    loop.stop()

if sys.platform != "win32":
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda sig=sig: asyncio.create_task(shutdown(sig, loop)))
else:
    print("Skipping signal handlers on Windows (not supported).")


class RepositoryManager:
    def __init__(self, repo_url: str, clone_path: Path, vector_store: VectorStore):
        self.repo_url = repo_url
        self.clone_path = clone_path
        self.vector_store = vector_store  # Add FAISS storage

    async def clone_repository(self):
        """Clones a Git repository asynchronously and indexes it in FAISS."""
        async with async_repo_manager(self.repo_url, self.clone_path, self.vector_store) as repo:
            if repo.clone_path.exists():
                print(f"Repository already exists at {repo.clone_path}")
                return

            try:
                print(f"Cloning repository from {repo.repo_url}...")
                await asyncio.to_thread(Repo.clone_from, repo.repo_url, repo.clone_path)
                print("Repository cloned successfully.")

                await repo.index_repository_files()  # Indexing after clone

            except GitCommandError as e:
                print(f"Failed to clone repository: {e}")
                raise

    def list_files(self, extensions=None):
        """
        List all files in the repository directory.
        :param extensions: List of file extensions to filter by (e.g., ['.py', '.md']).
        :return: List of file paths.
        """
        extensions = extensions or []
        all_files = []
        for root, _, files in os.walk(self.clone_path):
            #TODO: use root to better understand file purpose (maybe code into metadata)
            for file in files:
                if not extensions or file.endswith(tuple(extensions)):
                    all_files.append(Path(root) / file)
        return all_files

    async def index_repository_files(self):
        """Reads, chunks, and stores repository code into the FAISS vector database asynchronously."""
        print("Indexing repository files...")
        files = self.list_files(extensions=[".py", ".md", ".txt"])  # Index only relevant files
        tasks = [self.process_file(file) for file in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log Errors
        for file, result in zip(files, results):
            if isinstance(result, Exception):
                print(f"Error processing {file}: {result}")
        await asyncio.to_thread(self.vector_store.save_index)

    async def process_file(self, file):
        """Process file asynchronously using async for."""
        try:
            file_extension = file.suffix
            chunk_number = 0
            async for chunk in file_chunker(file, chunk_size=512):
                metadata = {
                    "text": chunk,
                    "filename": file.name,
                    "chunk_number": chunk_number,
                    "file_extension": file_extension,
                }
                await asyncio.to_thread(self.vector_store.add_text, chunk, metadata)
                chunk_number += 1
        except Exception as e:
            print(f"Skipping {file}: {e}")


@asynccontextmanager
async def async_repo_manager(repo_url, clone_path, vector_store):
    """Manages repository resources asynchronously."""
    repo_manager = RepositoryManager(repo_url, clone_path, vector_store)
    try:
        yield repo_manager
    finally:
        print(f"Closing repository: {repo_url}")

