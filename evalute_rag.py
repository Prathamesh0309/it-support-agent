from google import genai
from dotenv import load_dotenv
import os
from phase1_rag import retrieve, generate_answer

load_dotenv()
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

test_cases = [
    {
    "question": "Can I reset my password by calling the IT hotline?",
    "ground_truth": "No. Password reset is done via self-service at accounts.company.com/reset using mobile OTP. The IT hotline at ext 4357 is only for urgent account unlocks, not password resets."
},
{
    "question": "Does Jira auto approve for everyone?",
    "ground_truth": "No. Jira is auto-approved only for Engineering team members. For all other teams it takes 1 business day and requires approval."
}
]


def evaluate_with_llm(question, answer, context, ground_truth):
    prompt = f"""You are an expert evaluator for AI systems. 
Evaluate the following RAG system response on two metrics.
Score each metric from 0.0 to 1.0.

QUESTION: {question}

RETRIEVED CONTEXT: {context}

GENERATED ANSWER: {answer}

GROUND TRUTH: {ground_truth}

Evaluate:
1. FAITHFULNESS (0.0-1.0): Is the answer grounded in the retrieved context? 
   Does it avoid hallucinating information not present in the context?
   1.0 = completely faithful, 0.0 = completely hallucinated

2. ANSWER RELEVANCY (0.0-1.0): Does the answer actually address the question?
   Is it helpful and on-topic?
   1.0 = perfectly relevant, 0.0 = completely irrelevant

Respond in this EXACT format:
FAITHFULNESS: <score>
ANSWER_RELEVANCY: <score>
REASONING: <one sentence explaining your scores>"""

    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    
    text = response.text
    lines = text.strip().split("\n")
    
    scores = {}
    for line in lines:
        if line.startswith("FAITHFULNESS:"):
            scores["faithfulness"] = float(line.split(":")[1].strip())
        elif line.startswith("ANSWER_RELEVANCY:"):
            scores["answer_relevancy"] = float(line.split(":")[1].strip())
        elif line.startswith("REASONING:"):
            scores["reasoning"] = line.split(":", 1)[1].strip()
    
    return scores

def evaluate_with_llm(question, answer, context, ground_truth):
    prompt = f"""You are an expert evaluator for AI systems. 
Evaluate the following RAG system response on two metrics.
Score each metric from 0.0 to 1.0.

QUESTION: {question}

RETRIEVED CONTEXT: {context}

GENERATED ANSWER: {answer}

GROUND TRUTH: {ground_truth}

Evaluate:
1. FAITHFULNESS (0.0-1.0): Is the answer grounded in the retrieved context? 
   Does it avoid hallucinating information not present in the context?
   1.0 = completely faithful, 0.0 = completely hallucinated

2. ANSWER RELEVANCY (0.0-1.0): Does the answer actually address the question?
   Is it helpful and on-topic?
   1.0 = perfectly relevant, 0.0 = completely irrelevant

Respond in this EXACT format:
FAITHFULNESS: <score>
ANSWER_RELEVANCY: <score>
REASONING: <one sentence explaining your scores>"""

    response = gemini_client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt
    )
    
    text = response.text
    lines = text.strip().split("\n")
    
    scores = {}
    for line in lines:
        if line.startswith("FAITHFULNESS:"):
            scores["faithfulness"] = float(line.split(":")[1].strip())
        elif line.startswith("ANSWER_RELEVANCY:"):
            scores["answer_relevancy"] = float(line.split(":")[1].strip())
        elif line.startswith("REASONING:"):
            scores["reasoning"] = line.split(":", 1)[1].strip()
    
    return scores

def run_evaluation():
    print("=== RAG Evaluation — IT Support Copilot ===\n")
    
    all_faithfulness = []
    all_relevancy = []
    
    for i, test in enumerate(test_cases):
        print(f"Evaluating question {i+1}/{len(test_cases)}...")
        print(f"  Q: {test['question']}")
        
        # Run actual RAG pipeline
        chunks = retrieve(test["question"])
        answer = generate_answer(test["question"], chunks)
        context = " ".join([c["text"] for c in chunks])
        
        # Evaluate with LLM as judge
        scores = evaluate_with_llm(
            test["question"],
            answer,
            context,
            test["ground_truth"]
        )
        
        all_faithfulness.append(scores.get("faithfulness", 0))
        all_relevancy.append(scores.get("answer_relevancy", 0))
        
        print(f"  Faithfulness:     {scores.get('faithfulness', 0):.3f}")
        print(f"  Answer Relevancy: {scores.get('answer_relevancy', 0):.3f}")
        print(f"  Reasoning:        {scores.get('reasoning', 'N/A')}")
        print()
    
    avg_faith = sum(all_faithfulness) / len(all_faithfulness)
    avg_rel = sum(all_relevancy) / len(all_relevancy)
    overall = (avg_faith + avg_rel) / 2
    
    print("="*55)
    print("  Final Scores")
    print("="*55)
    print(f"  Avg Faithfulness:     {avg_faith:.3f}")
    print(f"  Avg Answer Relevancy: {avg_rel:.3f}")
    print(f"  Overall Score:        {overall:.3f}")
    print("="*55)

if __name__ == "__main__":
    run_evaluation()