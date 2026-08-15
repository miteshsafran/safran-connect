from pathlib import Path


# backend/
BASE_DIR = Path(__file__).resolve().parents[1]

# Company documents
DOCUMENTS_DIR = BASE_DIR / "documents"


# Ollama
OLLAMA_BASE_URL = "http://localhost:11434"

# Embedding model
EMBEDDING_MODEL = "company-embedding:latest"

# Generation model
LLM_MODEL = "company-gemma:latest"


# Qdrant
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

QDRANT_COLLECTION = "company_documents"