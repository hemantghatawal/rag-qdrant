from fastapi import APIRouter
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from google import genai

load_dotenv()

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

router = APIRouter(prefix="/query", tags=["query"])


# Embedding the chunks using OpenAI Embeddings
embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

vector_db = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    collection_name="sample_collection",
    url="http://localhost:6333",
)


@router.post("/")
async def query(question: str):
    search_results = vector_db.similarity_search(query=question)
    print(search_results)
    context = " ".join(
        [
            f"Page {result.metadata.get('page', 'N/A')}: {result.page_content}"
            for result in search_results
        ]
    )
    SYSTEM_PROMPT = f"""
    You are a helpful assistant that answers questions based on the provided context. If the context does not contain the answer, respond with "I don't know." {context} 
    """

    response = gemini_client.interactions.create(
        model="gemini-2.5-flash",
        contents=[
            {"role": "user", "parts": [{"text": SYSTEM_PROMPT}, {"text": question}]}
        ],
    )

    return {"question": question, "answer": response.outputs[-1].text}
