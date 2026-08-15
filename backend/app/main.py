from fastapi import FastAPI

from app.api.chat import router as chat_router


app = FastAPI(
    title="Company AI Assistant",
    description="Internal company RAG application",
    version="1.0.0",
)


app.include_router(
    chat_router
)


@app.get("/")
def root():

    return {
        "application": "Company AI Assistant",
        "status": "running",
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
    }