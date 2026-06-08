# IT Support Agent — Phase 1

Short README for the code present in this workspace as of now.

## Overview

This repository contains a small proof-of-concept RAG (retrieval-augmented generation) pipeline and ingestion helpers intended for an IT support assistant. It includes data files, a local Chroma DB, and example scripts to ingest and query the data.

## Repository structure

- `phase1_ingest.py` — scripts to ingest text files into the vector DB.
- `phase1_rag.py` — example retrieval / RAG runtime that queries the DB.
- `test.py` — small script used for quick checks or experiments.
- `requirements.txt` — Python dependencies for the project.
- `data/` — source documents used for ingestion (e.g., `software_and_hardware.txt`, `vpn_and_accounts.txt`).
- `db/` — Chroma DB files (e.g., `chroma.sqlite3`).
- `it-agent/` — local Python virtual environment used for development.

## Setup

1. Activate the provided virtual environment (if you want to use it):

```bash
source it-agent/bin/activate
```

2. (Optional) Install dependencies into your active environment:

```bash
pip install -r requirements.txt
```

## Running

- Ingest data into the vector DB:

```bash
python phase1_ingest.py
# or, when using the included venv:
it-agent/bin/python phase1_ingest.py
```

- Run the RAG example / query flow:

```bash
python phase1_rag.py
# or:
it-agent/bin/python phase1_rag.py
```

- Quick checks / experiments:

```bash
python test.py
```

## Data and DB

- Source text: `data/` contains the plain-text documents used by the ingestion step.
- Local DB: `db/chroma.sqlite3` stores the Chroma vector DB used at runtime.

## Notes & Next steps

- The repo currently assumes local-only models / embeddings or configured remote services. Update configuration in the scripts to point to your embedding / LLM provider if necessary.
- We're planning to add an MCP (Model Context Protocol) server to host models and manage conversation context; this will provide a lightweight API for retrieval and model serving.


