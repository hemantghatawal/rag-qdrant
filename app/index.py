# this file is for the indexing of the files
from dotenv import load_dotenv

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()


pdf_path = Path(__file__).parent / "data" / "sample.pdf"

# PDF Loader
loader = PyPDFLoader(str(pdf_path))
docs = loader.load()

# split the document in the chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
chunks = text_splitter.split_documents(documents=docs)

# Embedding the chunks using OpenAI Embeddings
embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
# gemini-embedding-001


# Create a Qdrant Vector Store
vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name = "sample_collection",
    url="http://localhost:6333"
)

print("Vector Store created and documents embedded successfully ✅")