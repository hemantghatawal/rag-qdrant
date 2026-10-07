from fastapi import FastAPI
from routes.query import router as query_router

app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation (RAG) API for indexing PDF documents and performing semantic search over their content. "
    "Uses Qdrant as the vector database, LangChain for document processing pipelines, and Google Gemini for embeddings and LLM capabilities. "
    "Features include PDF loading, recursive character text splitting, vector similarity search, and context-aware question answering.",
    version="0.1.0",
)

app.include_router(query_router)


