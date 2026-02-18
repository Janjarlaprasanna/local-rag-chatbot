# Local RAG Chatbot

A fully local Retrieval-Augmented Generation (RAG) chatbot. Upload documents (PDF, TXT, CSV), ask questions, and get accurate answers with source citations. **No API keys needed — everything runs on your machine.**

## Architecture

```
User uploads doc ─→ Document Loader (chunk) ─→ Embeddings (encode) ─→ ChromaDB (store)

User asks question ─→ Embeddings (encode query) ─→ ChromaDB (top-k search) ─→ Build Prompt ─→ Ollama LLM (generate) ─→ Stream to UI
```

### Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Embedding Model | `all-MiniLM-L6-v2` (sentence-transformers) | Convert text to 384-dim vectors |
| Vector Store | ChromaDB (embedded, persistent) | Store and search document chunks |
| LLM | Ollama + `llama3.2:3b` | Generate answers from context |
| Document Loader | LangChain loaders + splitters | Parse PDF/TXT/CSV and chunk |
| UI | Streamlit | Chat interface with file upload |

## Tech Stack

- **Python 3.11**
- **sentence-transformers** — CPU-friendly embeddings (no GPU required)
- **ChromaDB** — Embedded vector database (no Docker needed)
- **Ollama** — Local LLM inference (OpenAI-compatible API)
- **LangChain** — Document loaders and text splitters
- **Streamlit** — Interactive chat UI

## Setup

### Prerequisites

1. **Python 3.11+** installed
2. **Ollama** installed from [ollama.com](https://ollama.com)
3. Pull the LLM model:
   ```bash
   ollama pull llama3.2:3b
   ```

### Installation

```bash
git clone https://github.com/Janjarlaprasanna/local-rag-chatbot.git
cd local-rag-chatbot
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration (Optional)

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

Available settings:
- `OLLAMA_BASE_URL` — Ollama API endpoint (default: `http://localhost:11434/v1`)
- `OLLAMA_MODEL` — LLM model name (default: `llama3.2:3b`)
- `CHUNK_SIZE` — Document chunk size in characters (default: `500`)
- `CHUNK_OVERLAP` — Overlap between chunks (default: `50`)
- `TOP_K` — Number of chunks to retrieve (default: `5`)

## Usage

### Start the Chatbot

```bash
streamlit run app.py
```

### Quick Start

1. Launch the app
2. Upload a document (PDF, TXT, or CSV) via the sidebar
3. Click **Ingest Documents**
4. Ask questions in the chat input

### Try with Sample Data

The `data/` directory includes sample documents:
- `sample.txt` — Acme Corp Engineering Handbook
- `sample.csv` — Employee directory (20 rows)
- `sample.pdf` — Q3 2025 Quarterly Report

Upload these through the UI to test the system.

## Project Structure

```
local-rag-chatbot/
├── app.py                        # Streamlit UI
├── src/rag/
│   ├── config.py                 # Configuration constants
│   ├── document_loader.py        # PDF/TXT/CSV loading & chunking
│   ├── embeddings.py             # Sentence-transformers wrapper
│   ├── vector_store.py           # ChromaDB operations
│   ├── llm.py                    # Ollama client (OpenAI-compatible)
│   └── chain.py                  # RAG pipeline: retrieve → prompt → generate
├── data/                         # Sample documents
├── evaluation/
│   ├── eval_dataset.json         # Test Q&A pairs
│   └── evaluate.py               # Automated evaluation script
└── tests/                        # Unit tests (no Ollama needed)
```

## Testing

Tests are designed to run **without Ollama** (LLM calls are mocked):

```bash
pytest tests/ -v
```

### Evaluation (requires Ollama)

Run the full RAG evaluation against sample documents:

```bash
python -m evaluation.evaluate
```

## Key Design Decisions

- **OpenAI-compatible client** — The Ollama integration uses the OpenAI Python library, making it trivial to switch to OpenAI/Azure/any compatible API by changing one config value
- **Singleton pattern** — Embedding model and DB client are loaded once and reused, avoiding reloading the 80MB model per request
- **Deterministic chunk IDs** — Format `{filename}__chunk_{i}` ensures re-uploads overwrite existing chunks rather than creating duplicates
- **Anti-hallucination prompt** — The RAG prompt explicitly instructs the LLM to only use provided context and admit when information is insufficient
- **Streaming responses** — Answers stream token-by-token for a responsive chat experience

## License

MIT
