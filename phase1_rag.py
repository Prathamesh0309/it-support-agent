import os
from dotenv import load_dotenv
import chromadb
from google import genai
from phase1_ingest import GeminiEmbeddingFunction

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

CHROMA_DIR = "./db"
COLLECTION_NAME = "it_knowledge_base"
TOP_K = 3

def load_collection():
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=GeminiEmbeddingFunction(gemini_client)
    )
    return collection

def retrieve(query):
    collection = load_collection()
    
    results = collection.query(
        query_texts=[query],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "score": round(1 - results["distances"][0][i], 3)
        })

    return chunks

def generate_answer(query: str, chunks: list, history: list = []) -> str:
    
    # Format history for the prompt
    history_text = ""
    if history:
        for turn in history:
            role = "Employee" if turn["role"] == "user" else "Copilot"
            history_text += f"{role}: {turn['content']}\n"
    else:
        history_text = "No previous conversation."

    context = ""
    for i, chunk in enumerate(chunks, 1):
        context += f"\nSource {i} ({chunk['source']}, relevance: {chunk['score']}):\n"
        context += chunk["text"] + "\n"

    prompt = f"""You are a helpful IT Support Copilot for a tech company.
Answer the employee's question using ONLY the context provided below.
Be conversational and empathetic. Explain WHY each step helps, don't just list them.

CONVERSATION HISTORY:
{history_text}

KNOWLEDGE BASE CONTEXT:
{context}

INSTRUCTIONS:
- Take conversation history into account when answering
- Don't repeat suggestions already made in the conversation history
- If the employee said they already tried something, acknowledge it and move on
- If the answer is not in the context say: "I don't have information about that. Please contact it-support@company.com"

EMPLOYEE LATEST QUESTION: {query}

ANSWER:"""

    response = gemini_client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt
    )

    return response.text

def rag_query(query):
    print(f"\n{'─'*50}")
    print(f"Query: {query}")
    print(f"{'─'*50}")

    chunks = retrieve(query)

    print(f"\nTop {TOP_K} retrieved chunks:")
    for i, c in enumerate(chunks, 1):
        print(f"  {i}. [{c['source']}] score={c['score']}")
        print(f"     \"{c['text'][:80]}...\"")

    answer = generate_answer(query, chunks)
    return answer


if __name__ == "__main__":
    print("=== IT Support Copilot — RAG Query ===")
    print("Type 'quit' to exit\n")

    while True:
        user_input = input("Your question: ").strip()

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if not user_input:
            continue

        answer = rag_query(user_input)
        print(f"\nIT Copilot:\n{answer}")