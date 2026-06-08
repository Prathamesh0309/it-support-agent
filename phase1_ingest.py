import os
import glob
from dotenv import load_dotenv
import chromadb
from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

DOCS_DIR = "./data"
CHROMA_DIR = "./db"
COLLECTION_NAME = "it_knowledge_base"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

def load_documents():
    documents = []
    txt_files = glob.glob(os.path.join(DOCS_DIR, "*.txt"))

    for filepath in txt_files:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        documents.append({
            "text": text,
            "source": os.path.basename(filepath)
        })
        print(f"Loaded: {os.path.basename(filepath)}")

    return documents

def chunk_documents(documents):
    chunks = []

    for doc in documents:
        text = doc["text"]
        source = doc["source"]
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + CHUNK_SIZE
            chunk_text = text[start:end].strip()

            if len(chunk_text) > 50:
                chunks.append({
                    "text": chunk_text,
                    "id": f"{source}_chunk_{chunk_index}",
                    "source": source
                })
                chunk_index += 1

            start += CHUNK_SIZE - CHUNK_OVERLAP

        print(f"Created {chunk_index} chunks from {source}")

    return chunks

class GeminiEmbeddingFunction:
    def __init__(self, gemini_client):
        self.gemini_client = gemini_client

    def name(self):
        return "GeminiEmbeddingFunction"

    def _embed(self, texts, task_type):
        embeddings = []
        for text in texts:
            result = self.gemini_client.models.embed_content(
                model="gemini-embedding-001",
                contents=text,
            )
            embeddings.append(result.embeddings[0].values)
        return embeddings

    def __call__(self, input):
        return self._embed(input, "retrieval_document")

    def embed_documents(self, input):
        return self._embed(input, "retrieval_document")

    def embed_query(self, input):
        return self._embed(input, "retrieval_query")
    
    
    
def store_in_chromadb(chunks, gemini_client):
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=GeminiEmbeddingFunction(gemini_client),
        metadata={"hnsw:space": "cosine"}
    )

    collection.add(
        documents=[c["text"] for c in chunks],
        ids=[c["id"] for c in chunks],
        metadatas=[{"source": c["source"]} for c in chunks]
    )

    print(f"Stored {collection.count()} chunks in ChromaDB")


if __name__ == "__main__":
    print("=== Phase 1: Ingestion ===")

    print("\n[1/3] Loading documents...")
    documents = load_documents()

    print("\n[2/3] Chunking...")
    chunks = chunk_documents(documents)

    print("\n[3/3] Storing in ChromaDB...")
    store_in_chromadb(chunks, gemini_client)

    print("\nDone! Run phase1_rag.py to query.")