# ConnectAI

I want to create an application for my company where employees can ask questions about company documents, such as Leave Policies, HR Policies, and other internal guidelines.

The application should use a RAG (Retrieval-Augmented Generation) architecture with a local LLM and a Vector Database. Users can enter prompts/questions, and the system will search the relevant documents, retrieve the most relevant information, and generate accurate answers based on the document content.

The application should also display the source documents and relevant sections used to generate the response.

Technology Stack:

Frontend: React
Backend: Python (FastAPI)
RAG Framework: LangChain or LlamaIndex
Vector Database: ChromaDB or Qdrant
Local LLM: Llama 3.1, Mistral, or Qwen via Ollama

Key Features:

Upload and manage company documents (PDF, DOCX, etc.)
Generate embeddings and store them in a vector database
Semantic document search using RAG
Chat interface for employees
Source citations and document references
Local LLM support for data privacy and security
Role-based access control (Admin, HR, Employee)
Conversation history and logging