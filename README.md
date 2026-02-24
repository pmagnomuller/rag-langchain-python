# LangChain RAG with Ollama and Chroma

This project sets up a minimal Retrieval-Augmented Generation (RAG) stack using:

- **LangChain** for orchestration
- **Ollama** for the local LLM (and optionally embeddings)
- **Chroma** as the vector store
- **Local files** (text/markdown/PDF) as the primary knowledge source
- A **Jupyter notebook** for interactive experimentation

## Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com) installed and running locally
- At least one Ollama model pulled, for example:

```bash
ollama pull llama3
```

## Setup

Install Python dependencies (preferably in a virtual environment):

```bash
pip install -r requirements.txt
```

Create a `data/` directory and place your documents there (e.g. `.txt`, `.md`, `.pdf`).

Optionally, create a `.env` file in the project root to override defaults (see `rag/config.py` once created), for example:

```bash
RAG_OLLAMA_MODEL=llama3
RAG_EMBEDDING_MODEL=nomic-embed-text
RAG_DATA_DIR=data
RAG_CHROMA_DIR=chroma
```

## Usage

1. Start Ollama (if not already running).
2. Launch Jupyter and open `rag_notebook.ipynb`:

```bash
jupyter notebook
```

3. Run the notebook cells in order:
   - Environment & configuration
   - Ingestion / indexing
   - RAG chain construction
   - Ask questions against your documents

The `rag/` Python package provides reusable helpers for configuration, ingestion, and chain creation.

