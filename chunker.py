from typing import List
import config


class Chunker:

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None) -> None:
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

    def fixed_size_chunk(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def recursive_chunk(self, text: str, separators: List[str] = None) -> List[str]:
        if separators is None:
            separators = ["\n\n", "\n", ". ", " "]

        chunks = []
        self._recursive_split(text, separators, 0, chunks)
        return chunks

    def _recursive_split(
        self, text: str, separators: List[str], depth: int, chunks: List[str]
    ) -> None:
        if len(text) <= self.chunk_size:
            if text.strip():
                chunks.append(text.strip())
            return

        if depth >= len(separators):
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunk = text[i : i + self.chunk_size].strip()
                if chunk:
                    chunks.append(chunk)
            return

        separator = separators[depth]
        parts = text.split(separator)

        current = ""
        for part in parts:
            candidate = (current + separator + part).strip() if current else part.strip()

            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())
                if len(part) > self.chunk_size:
                    self._recursive_split(part, separators, depth + 1, chunks)
                    current = ""
                else:
                    current = part.strip()

        if current.strip():
            chunks.append(current.strip())

    def chunk_documents(self, documents: List[str], method: str = "recursive") -> List[str]:
        all_chunks = []
        chunker_fn = self.recursive_chunk if method == "recursive" else self.fixed_size_chunk

        for doc in documents:
            all_chunks.extend(chunker_fn(doc))

        return all_chunks
