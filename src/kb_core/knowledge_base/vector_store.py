"""ChromaDB-backed vector store for semantic retrieval."""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from ..schema_engine import KnowledgeChunk


class VectorStore:
    """Stores and retrieves KnowledgeChunks via semantic similarity."""

    COLLECTION_NAME = "pichia_chunks"

    def __init__(self, db_path: Path) -> None:
        self._client = chromadb.PersistentClient(path=str(db_path))
        self._ef = embedding_functions.DefaultEmbeddingFunction()
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[KnowledgeChunk]) -> int:
        """Add chunks to the collection. Returns number added."""
        if not chunks:
            return 0

        ids = [c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = [
            {
                "source_file": c.source_file,
                "section": c.section or "",
                "keywords": ",".join(c.keywords),
            }
            for c in chunks
        ]

        # Skip duplicates
        existing = set(self._collection.get(ids=ids)["ids"])
        new = [(i, d, m) for i, d, m in zip(ids, documents, metadatas) if i not in existing]
        if not new:
            return 0

        new_ids, new_docs, new_metas = zip(*new)
        self._collection.add(
            ids=list(new_ids),
            documents=list(new_docs),
            metadatas=list(new_metas),
        )
        return len(new_ids)

    def query(self, text: str, n_results: int = 6) -> list[dict]:
        """Return top-n most relevant chunks with metadata."""
        results = self._collection.query(
            query_texts=[text],
            n_results=min(n_results, self._collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )
        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                "content": doc,
                "source_file": meta.get("source_file", ""),
                "section": meta.get("section", ""),
                "keywords": meta.get("keywords", "").split(","),
                "relevance": round(1 - dist, 4),
            })
        return output

    def count(self) -> int:
        return self._collection.count()

    def clear(self) -> None:
        self._client.delete_collection(self.COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )
