import glob
import os
import re

import chromadb
from google.genai import types

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DOCS_DIR,
    EMBEDDING_MODEL,
    gemini_client,
)

# Fallback window for sections that are too long to embed as one chunk.
MAX_CHUNK_CHARS = 800
CHUNK_OVERLAP = 100
MIN_CHUNK_CHARS = 50
EMBED_BATCH_SIZE = 50


def load_documents():
    documents = []
    for filepath in sorted(glob.glob(os.path.join(DOCS_DIR, "*.txt"))):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        documents.append({"text": text, "source": os.path.basename(filepath)})
        print(f"Loaded: {os.path.basename(filepath)}")
    return documents


def _split_long(text, max_chars=MAX_CHUNK_CHARS, overlap=CHUNK_OVERLAP):
    """Sliding window used only when a single section exceeds max_chars."""
    if len(text) <= max_chars:
        return [text]
    pieces, start = [], 0
    while start < len(text):
        pieces.append(text[start:start + max_chars])
        start += max_chars - overlap
    return pieces


def chunk_documents(documents):
    """Section-aware chunking.

    The knowledge base is written as self-contained sections separated by
    '---' ("Issue: VPN keeps disconnecting" + its steps). Splitting on those
    boundaries keeps a problem and its fix in the same chunk. A fixed 500-char
    window cut procedures mid-step and mixed unrelated issues together.
    Only oversized sections fall back to a sliding window with overlap.
    """
    chunks = []
    for doc in documents:
        # Match only a line that is exactly '---'. The longer '-----' lines under
        # each "Issue:" title are underlines, not separators.
        sections = re.split(r"(?m)^---[ \t]*$", doc["text"])
        chunk_index = 0
        for section in sections:
            for piece in _split_long(section.strip()):
                piece = piece.strip()
                if len(piece) < MIN_CHUNK_CHARS:
                    continue
                chunks.append({
                    "text": piece,
                    "id": f"{doc['source']}_chunk_{chunk_index}",
                    "source": doc["source"],
                })
                chunk_index += 1
        print(f"Created {chunk_index} chunks from {doc['source']}")
    return chunks


class GeminiEmbeddingFunction:
    """Chroma-compatible embedding function.

    Gemini embeddings are asymmetric: documents and queries should be embedded
    with different task types (RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY) so a short
    question lands near the passage that answers it.
    """

    def name(self):
        return "GeminiEmbeddingFunction"

    def _embed(self, texts, task_type):
        embeddings = []
        # One API call per batch instead of one per chunk.
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = list(texts[i:i + EMBED_BATCH_SIZE])
            result = gemini_client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            embeddings.extend(e.values for e in result.embeddings)
        return embeddings

    # Chroma calls this when it has to embed on its own; treat as documents.
    def __call__(self, input):
        return self._embed(input, "RETRIEVAL_DOCUMENT")

    def embed_documents(self, input):
        return self._embed(input, "RETRIEVAL_DOCUMENT")

    def embed_query(self, input):
        return self._embed(input, "RETRIEVAL_QUERY")


def store_in_chromadb(chunks):
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Full rebuild: simplest way to guarantee the index matches data/ exactly.
    existing = [c.name for c in chroma_client.list_collections()]
    if COLLECTION_NAME in existing:
        chroma_client.delete_collection(COLLECTION_NAME)

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=GeminiEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        documents=[c["text"] for c in chunks],
        ids=[c["id"] for c in chunks],
        metadatas=[{"source": c["source"]} for c in chunks],
    )
    print(f"Stored {collection.count()} chunks in ChromaDB")


if __name__ == "__main__":
    print("=== Phase 1: Ingestion ===")

    print("\n[1/3] Loading documents...")
    documents = load_documents()
    if not documents:
        raise SystemExit(f"No .txt files found in {DOCS_DIR}")

    print("\n[2/3] Chunking...")
    chunks = chunk_documents(documents)

    print("\n[3/3] Storing in ChromaDB...")
    store_in_chromadb(chunks)

    print("\nDone! Run phase1_rag.py to query.")
