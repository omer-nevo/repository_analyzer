import aiofiles


async def file_chunker(file_path : str, chunk_size: int = 512):
    """Asynchronous generator that yields file chunks """
    async with aiofiles.open(file_path, "r", errors="ignore") as f:
        chunk = []
        async for line in f:
            chunk.append(line)
            if sum(len(l) for l in chunk) >= chunk_size:
                yield "".join(chunk)
                chunk = []  # Reset chunk buffer

        if chunk:
            yield "".join(chunk)