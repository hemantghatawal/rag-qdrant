# RAG API

Retrieval-Augmented Generation (RAG) API for indexing PDF documents and answering questions about their content using semantic search and a large language model.

Built with **FastAPI**, **LangChain**, **Qdrant** vector database, and **Google Gemini** for embeddings + LLM responses.

---

## Architecture

```
PDF Documents → PDF Loader → Text Splitter → Gemini Embeddings → Qdrant Vector Store

User Question → Gemini Embeddings → Qdrant Similarity Search → Context
                                                              ↓
User Question + Context → Gemini LLM → Natural Language Answer
```

### Key Components

| File | Purpose |
|------|---------|
| [app/main.py](file:///Users/kanishkkaushik/Documents/RAG/app/main.py) | FastAPI app entrypoint, router registration |
| [app/index.py](file:///Users/kanishkkaushik/Documents/RAG/app/index.py) | One-off indexing script: loads PDF, chunks, embeds, and stores in Qdrant |
| [app/routes/query.py](file:///Users/kanishkkaushik/Documents/RAG/app/routes/query.py) | `/query` endpoint — searches Qdrant and asks Gemini |
| [app/data/sample.pdf](file:///Users/kanishkkaushik/Documents/RAG/app/data/sample.pdf) | Sample PDF bundled with the project (Panchatantra) |
| [docker-compose.yml](file:///Users/kanishkkaushik/Documents/RAG/docker-compose.yml) | Runs a local Qdrant instance on port 6333 |

### Pipeline Details

- **PDF loading**: `PyPDFLoader` from `langchain-community`
- **Chunking**: `RecursiveCharacterTextSplitter` — 1000-character chunks with 400-character overlap
- **Embeddings**: `gemini-embedding-2` model from Google Generative AI
- **Vector DB**: Qdrant collection `sample_collection` at `http://localhost:6333`
- **Retrieval**: Top-4 (`k=4`) cosine-similarity chunks
- **LLM**: `gemini-2.5-flash` with a system instruction to answer only from context (or say "I don't know")

---

## Prerequisites

- Python 3.12+
- Docker (for running Qdrant locally)
- A Google AI Studio API key → https://aistudio.google.com/apikey

---

## Setup

### 1. Clone & install dependencies

```bash
cd RAG
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and paste your Google API key:
# GEMINI_API_KEY=your_key_here
```

### 3. Start Qdrant

```bash
docker compose up -d
```

Qdrant Web UI: http://localhost:6333/dashboard

### 4. Index the sample PDF

This step loads the PDF, chunks it, embeds it, and stores vectors in Qdrant.

```bash
python app/index.py
```

You should see:

```
Vector Store created and documents embedded successfully ✅
```

### 5. Start the API server

```bash
uvicorn app.main:app --reload
```

Or from within the `app` directory:

```bash
cd app
fastapi dev
```

- API: http://127.0.0.1:8000
- Docs (Swagger): http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

---

## Usage

### Ask a question

```bash
curl -X POST "http://127.0.0.1:8000/query/" \
  --get --data-urlencode "question=Who is the author of the book?"
```

Or visit the [/docs](http://127.0.0.1:8000/docs) UI and use the **POST /query/** endpoint.

**Response**:

```json
{
  "question": "Who is the author of the book?",
  "answer": "The author of the book is Pandit Vishnu Sharma."
}
```

### Using a different PDF

1. Copy your PDF into `app/data/`.
2. Edit [app/index.py](file:///Users/kanishkkaushik/Documents/RAG/app/index.py#L13) and update `pdf_path` to point to your file.
3. Optionally change the `collection_name` in both [index.py](file:///Users/kanishkkaushik/Documents/RAG/app/index.py#L32) and [query.py](file:///Users/kanishkkaushik/Documents/RAG/app/routes/query.py#L20) if you don't want to overwrite `sample_collection`.
4. Re-run `python app/index.py`.

---

## Project Structure

```
RAG/
├── app/
│   ├── __pycache__/
│   ├── data/
│   │   └── sample.pdf
│   ├── routes/
│   │   ├── __pycache__/
│   │   └── query.py
│   ├── index.py
│   └── main.py
├── .env.example
├── .gitignore
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Common Issues

### `Connection refused` when indexing or querying

Qdrant isn't running. Make sure `docker compose up -d` ran successfully and port 6333 is reachable.

### `TypeError: create() got unexpected keyword argument(s): contents`

Make sure you're using `gemini_client.models.generate_content(...)` (not `interactions.create(...)`).

### API says "I don't know." for everything

- Re-run the indexing script (`python app/index.py`) to ensure the collection is populated.
- Verify chunks exist via the Qdrant dashboard at http://localhost:6333/dashboard → Collections → `sample_collection`.
- Try a question closely worded to text you know is in the PDF.

### Import errors when starting the server

Make sure to start the server from the project **root** using the module path:

```bash
uvicorn app.main:app --reload
```

---

## Tech Stack

| Layer | Library / Service |
|-------|-------------------|
| Web Framework | FastAPI |
| LLM Framework | LangChain |
| PDF Loader | PyPDF via langchain-community |
| Embeddings | Google Generative AI — `gemini-embedding-2` |
| LLM | Google Generative AI — `gemini-2.5-flash` |
| Vector DB | Qdrant (via Docker) |
| Configuration | python-dotenv |
| ASGI Server | Uvicorn |
