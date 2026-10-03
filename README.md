# IT Support Agent

A completed proof-of-concept IT support assistant built with retrieval-augmented generation (RAG), local vector search, and an MCP orchestration layer.

## What this project does

- Ingests IT support text documents into a local Chroma vector database.
- Retrieves relevant content for user questions using embeddings.
- Generates contextual IT support answers with Gemini LLMs.
- Supports a lightweight MCP server to simulate tool-enabled workflows such as ticket creation and Slack notifications.
- Includes evaluation tooling for assessing retrieval and answer quality.

## Architecture

```
question ─► rewrite follow-ups (history) ─► embed as RETRIEVAL_QUERY ─► Chroma top-3
        ─► similarity < MIN_RELEVANCE? ─► yes: refuse + escalate (no LLM guessing)
        ─► no: Gemini answers from context only ─► Gemini picks tools (schema-constrained JSON)
        ─► MCP client ─► create_ticket / check_ticket_status / notify_slack
```

## Repository structure

- `config.py` — models, thresholds, the shared Gemini client, and a retry wrapper for transient API errors.
- `phase1_ingest.py` — loads `data/*.txt`, chunks by section, embeds (batched, `RETRIEVAL_DOCUMENT`), stores in `db/`.
- `phase1_rag.py` — retrieval (query rewriting, cosine scores, relevance guardrail) and answer generation.
- `mcp_server.py` — MCP server with `create_ticket`, `check_ticket_status`, `notify_slack`; tickets persist to `tickets.json`.
- `orchestrator.py` — RAG, then an LLM routing decision, then MCP tool calls over one long-lived session.
- `evaluate_rag.py` — retrieval metrics (hit rate, MRR), refusal check, and LLM-as-judge faithfulness/relevancy.
- `test_mcp.py` — smoke test for the MCP server, including ticket persistence across processes.
- `data/` — the sample knowledge base (committed so the project is reproducible).

## Design decisions

- **Section-aware chunking.** The KB is written as self-contained "Issue + steps" sections, so we split on `---` rather than fixed character windows, which cut procedures mid-step. Only oversized sections fall back to an overlapping window.
- **Asymmetric embeddings.** Documents use `RETRIEVAL_DOCUMENT`, queries use `RETRIEVAL_QUERY`.
- **Refusal guardrail.** If the best chunk's cosine similarity is below `MIN_RELEVANCE` (0.62, calibrated on this KB: in-scope 0.66–0.74, out-of-scope 0.51–0.60), the LLM is skipped and the request is escalated to a ticket.
- **Structured routing.** The tool decision uses a response schema with `temperature=0`, instead of parsing free text with a regex.
- **MCP stdio.** stdout is the protocol channel, so the server logs to stderr only.
- **Separate judge.** Evaluation uses a different model from the generator (`JUDGE_MODEL` env var overrides it).

## Known limitations

- Tiny knowledge base (6 chunks) and 7 eval cases: metrics are a rough signal, not a benchmark.
- The relevance threshold is tuned on very few examples and must be re-checked when the KB or embedding model changes.
- Ticketing and Slack are mocks (JSON file / log line).
- Retrieval is dense-only; there is no keyword (BM25) or reranking stage.

## Prerequisites

- Python 3.13+ (recommended)
- `pip` or `venv` support
- A Gemini API key stored in a `.env` file as `GEMINI_API_KEY`

## Setup

1. Create and activate a virtual environment, then install dependencies:

```bash
python -m venv it-agent && source it-agent/bin/activate
pip install -r requirements.txt
```

2. Create a `.env` file with your Gemini API key:

```text
GEMINI_API_KEY=your_api_key_here
```

## Usage

### 1. Ingest knowledge into ChromaDB

```bash
python phase1_ingest.py
```

This reads the text files in `data/`, chunks them, embeds them via Gemini, and stores them in `db/`.

### 2. Run a RAG query session

```bash
python phase1_rag.py
```

This starts a prompt loop where you can ask IT support questions and receive answers grounded in your knowledge base.

### 3. Run the MCP orchestrator

```bash
python orchestrator.py
```

This launches an interactive session that performs retrieval, reasoning, and tool actions such as ticket creation and Slack notifications.

### 4. Evaluate RAG performance

```bash
python evaluate_rag.py
```

Reports retrieval hit rate/MRR, out-of-scope refusal accuracy, and LLM-judged faithfulness and relevancy.

## Quick command summary

```bash
python -m venv it-agent && source it-agent/bin/activate
pip install -r requirements.txt
python phase1_ingest.py
python phase1_rag.py
python orchestrator.py
python evaluate_rag.py
python test_mcp.py   # MCP smoke test
```
