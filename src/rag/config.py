"""Configuration constants with environment variable overrides."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", str(PROJECT_ROOT / "chroma_db"))
UPLOAD_DIR = PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Ollama / LLM
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# Embedding
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Batch size for embedding and upserting
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "256"))

# RAG
TOP_K = int(os.getenv("TOP_K", "5"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "documents")

# Prompt template
RAG_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the user's question using ONLY the context provided below. If the context does not contain enough information to answer the question, say "I don't have enough information in the provided documents to answer that question."

Do not make up information. Do not use any knowledge outside of the provided context.

Context:
{context}

Question: {question}

Answer:"""
