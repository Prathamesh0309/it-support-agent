# IT Support Agent

A completed proof-of-concept IT support assistant built with retrieval-augmented generation (RAG), local vector search, and an MCP orchestration layer.

## What this project does

- Ingests IT support text documents into a local Chroma vector database.
- Retrieves relevant content for user questions using embeddings.
- Generates contextual IT support answers with Gemini LLMs.
- Supports a lightweight MCP server to simulate tool-enabled workflows such as ticket creation and Slack notifications.
- Includes evaluation tooling for assessing retrieval and answer quality.

## Repository structure

- `phase1_ingest.py` — loads `data/*.txt`, chunks content, generates embeddings, and stores them in `db/`.
- `phase1_rag.py` — query interface that retrieves top-K context and generates answers.
- `mcp_server.py` — MCP tool server with `create_ticket`, `check_ticket_status`, and `notify_slack` endpoints.
- `orchestrator.py` — orchestrates RAG retrieval, reasoning, and tool action execution via MCP.
- `evalute_rag.py` — evaluation script for RAG response quality using ground-truth test cases.
- `test.py` — a quick experiment script for ad hoc checks.
- `data/` — source knowledge documents for the IT support agent.
- `db/` — local Chroma database storage.
- `it-agent/` — bundled Python virtual environment for this project.

## Prerequisites

- Python 3.13+ (recommended)
- `pip` or `venv` support
- A Gemini API key stored in a `.env` file as `GEMINI_API_KEY`

## Setup

1. Activate the local virtual environment:

```bash
source it-agent/bin/activate
```

2. Install any missing dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your Gemini API key:

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
python evalute_rag.py
```

This runs predefined test cases and evaluates generated answers against ground truth.

## Notes

- The current system uses Gemini for both embeddings and generation.
- `phase1_rag.py` and `orchestrator.py` both rely on the local `db/` vector database.
- `mcp_server.py` provides mock tooling and can be extended to integrate with real ticketing or Slack APIs.

## Recommended next steps

- Add more IT knowledge sources to `data/`.
- Improve chunking and metadata handling for better retrieval accuracy.
- Replace mock MCP tools with real backend integrations.
- Add tests for conversation history, tool invocation, and end-to-end behavior.

## Quick command summary

```bash
source it-agent/bin/activate
pip install -r requirements.txt
python phase1_ingest.py
python phase1_rag.py
python orchestrator.py
python evalute_rag.py
```


