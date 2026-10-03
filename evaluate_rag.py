"""Evaluate the RAG pipeline on two separate questions:

1. RETRIEVAL  - did we fetch the right document?  (hit rate, MRR; no LLM involved)
2. GENERATION - is the answer faithful to the context and relevant? (LLM-as-judge)

Keeping them apart tells you *which* half is broken when a score drops.
"""
import re

from google.genai import types

from config import JUDGE_MODEL, generate
from phase1_rag import FALLBACK_ANSWER, generate_answer, is_relevant, retrieve

# expected_source: file that contains the answer (None = not in the KB at all,
# the correct behaviour is to refuse rather than invent something).
TEST_CASES = [
    {
        "question": "Can I reset my password by calling the IT hotline?",
        "ground_truth": "No. Password reset is self-service at accounts.company.com/reset using mobile OTP. The hotline (ext. 4357) is only for urgent account unlocks.",
        "expected_source": "vpn_and_accounts.txt",
    },
    {
        "question": "Does Jira auto approve for everyone?",
        "ground_truth": "No. Jira is auto-approved only for Engineering. Other teams wait 1 business day for approval.",
        "expected_source": "software_and_hardware.txt",
    },
    {
        "question": "My VPN keeps dropping every 20 minutes, what should I change?",
        "ground_truth": "Set the MTU to 1300 in VPN advanced settings, disable adapter power saving, use 5GHz Wi-Fi or wired, and reinstall the client if it persists.",
        "expected_source": "vpn_and_accounts.txt",
    },
    {
        "question": "How long until I can get AWS console access?",
        "ground_truth": "AWS Console needs Security team approval and takes 5-7 days.",
        "expected_source": "software_and_hardware.txt",
    },
    {
        "question": "My second monitor isn't showing anything",
        "ground_truth": "Check the cable at both ends, try another cable/port, run Detect Displays in OS settings, and reboot with the monitor connected.",
        "expected_source": "software_and_hardware.txt",
    },
    {
        "question": "I got locked out after typing my password wrong too many times",
        "ground_truth": "Accounts lock after 5 failed attempts. Wait 15 minutes for auto-unlock, or contact IT / call ext. 4357 for an urgent unlock.",
        "expected_source": "vpn_and_accounts.txt",
    },
    # Out-of-scope: the system should refuse, not hallucinate.
    {
        "question": "What is the company's parental leave policy?",
        "ground_truth": FALLBACK_ANSWER,
        "expected_source": None,
    },
]

JUDGE_PROMPT = """You are a strict evaluator for a RAG system. Score two metrics from 0.0 to 1.0.

QUESTION: {question}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}

REFERENCE ANSWER (for judging relevancy/correctness only):
{ground_truth}

1. FAITHFULNESS: judge ONLY against the RETRIEVED CONTEXT. Every claim in the answer must be
   supported by the context. 1.0 = fully supported, 0.0 = made up.
   (Politely declining because the context lacks the answer counts as faithful.)
2. ANSWER_RELEVANCY: does the answer correctly address the question, consistent with the reference?

Respond in this EXACT format:
FAITHFULNESS: <score>
ANSWER_RELEVANCY: <score>
REASONING: <one sentence>"""


def _parse_score(text, label):
    m = re.search(rf"{label}\s*:\s*([01](?:\.\d+)?)", text)
    return float(m.group(1)) if m else None


def judge(question, answer, context, ground_truth):
    response = generate(
        JUDGE_MODEL,
        JUDGE_PROMPT.format(
            question=question, context=context, answer=answer, ground_truth=ground_truth
        ),
        config=types.GenerateContentConfig(temperature=0),
    )
    text = response.text or ""
    reasoning = re.search(r"REASONING\s*:\s*(.+)", text)
    return {
        "faithfulness": _parse_score(text, "FAITHFULNESS"),
        "answer_relevancy": _parse_score(text, "ANSWER_RELEVANCY"),
        "reasoning": reasoning.group(1).strip() if reasoning else "N/A",
    }


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else float("nan")


def run_evaluation():
    print("=== RAG Evaluation — IT Support Copilot ===\n")

    hits, reciprocal_ranks, faith, rel, refusals_ok = [], [], [], [], []

    for i, test in enumerate(TEST_CASES, 1):
        q = test["question"]
        print(f"[{i}/{len(TEST_CASES)}] {q}")

        chunks = retrieve(q)
        answer = generate_answer(q, chunks)
        context = "\n---\n".join(c["text"] for c in chunks)

        expected = test["expected_source"]
        if expected:
            sources = [c["source"] for c in chunks]
            rank = sources.index(expected) + 1 if expected in sources else None
            hits.append(1.0 if rank else 0.0)
            reciprocal_ranks.append(1.0 / rank if rank else 0.0)
            print(f"  Retrieval:  {'HIT at rank ' + str(rank) if rank else 'MISS'} "
                  f"(top score {chunks[0]['score']})")
        else:
            # Should be refused: low similarity => guardrail short-circuits.
            ok = not is_relevant(chunks)
            refusals_ok.append(1.0 if ok else 0.0)
            print(f"  Refusal:    {'correct' if ok else 'WRONG - answered an out-of-scope question'} "
                  f"(top score {chunks[0]['score']})")

        print(f"  Answer: {answer[:300].strip()}...")
        scores = judge(q, answer, context, test["ground_truth"])
        faith.append(scores["faithfulness"])
        rel.append(scores["answer_relevancy"])
        print(f"  Faithfulness: {scores['faithfulness']}   Relevancy: {scores['answer_relevancy']}")
        print(f"  Judge: {scores['reasoning']}\n")

    print("=" * 55)
    print("  Results")
    print("=" * 55)
    print(f"  Retrieval hit rate@{len(chunks)}:   {mean(hits):.3f}")
    print(f"  Retrieval MRR:           {mean(reciprocal_ranks):.3f}")
    print(f"  Out-of-scope refusals:   {mean(refusals_ok):.3f}")
    print(f"  Avg faithfulness:        {mean(faith):.3f}")
    print(f"  Avg answer relevancy:    {mean(rel):.3f}")
    print("=" * 55)
    print("  Note: a handful of cases and an LLM judge give a rough signal, not a benchmark.")


if __name__ == "__main__":
    run_evaluation()
