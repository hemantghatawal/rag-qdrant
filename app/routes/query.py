from fastapi import APIRouter
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from google import genai
from google.genai import types

load_dotenv()

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

router = APIRouter(prefix="/query", tags=["query"])


embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

vector_db = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    collection_name="sample_collection",
    url="http://localhost:6333",
)


@router.post("/")
async def query(question: str):
    search_results = vector_db.similarity_search(query=question, k=4)
    print(search_results)
    context = " ".join(
        [
            f"Page {result.metadata.get('page', 'N/A')}: {result.page_content}"
            for result in search_results
        ]
    )

    response = gemini_client.models.generate_content(
        model="gemini-3.8-flash",
        config=types.GenerateContentConfig(
            system_instruction='You are a helpful assistant that answers questions based on the provided context. If the context does not contain the answer, respond with "I don\'t know."'
        ),
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(text=f"Context:\n{context}\n\nQuestion: {question}")],
            )
        ],
    )

    try:
        answer = response.candidates[0].content.parts[0].text
    except (IndexError, AttributeError):
        answer = "I don't know."

    return {"question": question, "answer": answer}
