"""Shared settings and the single Gemini client used by every module."""
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors

load_dotenv()

_api_key = os.getenv("GEMINI_API_KEY")
if not _api_key:
    raise RuntimeError("GEMINI_API_KEY is not set. Add it to a .env file (see README).")

gemini_client = genai.Client(api_key=_api_key)

# Models
EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-3.1-flash-lite"
# The judge is deliberately a different model from the generator, so the system
# is not grading its own homework.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-3.8-flash")

# Vector store
DOCS_DIR = "./data"
CHROMA_DIR = "./db"
COLLECTION_NAME = "it_knowledge_base"

# Retrieval
TOP_K = 3
# Cosine similarity below this means "nothing relevant found"; we skip the LLM
# and escalate instead of letting it improvise.
# Calibrated empirically on this KB: in-scope questions scored 0.66-0.74,
# out-of-scope ones 0.51-0.60, so 0.62 sits in the gap. Small sample - re-check
# with evaluate_rag.py whenever the knowledge base or embedding model changes.
MIN_RELEVANCE = 0.62


def generate(model, contents, config=None, retries=6):
    """generate_content with exponential backoff on transient server errors (5xx)."""
    for attempt in range(retries + 1):
        try:
            return gemini_client.models.generate_content(
                model=model, contents=contents, config=config
            )
        except errors.ServerError:
            # 5xx = transient overload. 4xx (bad model, quota exhausted) is not
            # fixed by waiting, so those propagate immediately.
            if attempt == retries:
                raise
            time.sleep(min(2 ** (attempt + 1), 30))
