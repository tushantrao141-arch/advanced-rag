# Advanced RAG Pipeline

A production-ready, two-stage Retrieval-Augmented Generation (RAG) pipeline featuring semantic search with FAISS, cross-encoder reranking, and OpenAI LLM response generation — with both CLI and Streamlit UI interfaces.

## Architecture

```
Query → Embedding (all-mpnet-base-v2) → FAISS Retrieval (Top-K)
                                              ↓
                                    Reranking (msmarco-distilbert-base-v3)
                                              ↓
                                    Context Assembly (Top-N)
                                              ↓
                                    LLM Generation (GPT-3.5-turbo)
                                              ↓
                                          Answer
```

## Features

- **Semantic Search**: FAISS index with SentenceBERT (`all-mpnet-base-v2`) embeddings for fast similarity search.
- **Reranking**: Cross-encoder model (`msmarco-distilbert-base-v3`) to improve retrieval precision.
- **Response Generation**: OpenAI GPT-3.5-turbo via LangChain for context-grounded answers.
- **Evaluation Metrics**: Built-in Recall@K, MRR, and NDCG@K for retrieval quality assessment.
- **Chunking Strategies**: Fixed-size and recursive text chunking with configurable overlap.
- **Web Interface**: Interactive Streamlit application with adjustable retriever/reranker settings.
- **Containerization**: Docker & Docker Compose for one-command deployment.
- **Tests**: Pytest test suite covering retrieval, reranking, and evaluation modules.

## Project Structure

```
advanced-rag/
├── main.py              # CLI entry point
├── app.py               # Streamlit web UI
├── config.py            # Centralized configuration
├── dataset.py           # HuggingFace dataset loader
├── embeddings.py        # SentenceTransformer embedding wrapper
├── retriever.py         # FAISS-based semantic retriever
├── reranker.py          # Cross-encoder reranker
├── chunker.py           # Text chunking strategies
├── llm.py               # LangChain OpenAI LLM wrapper
├── evaluate.py          # Retrieval evaluation metrics
├── utils.py             # Utility helpers
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container image definition
├── docker-compose.yml   # Multi-service orchestration
├── .env.example         # Environment variable template
├── .gitignore           # Git ignore rules
└── tests/
    ├── __init__.py
    ├── test_evaluate.py
    ├── test_reranker.py
    └── test_retriever.py
```

## Prerequisites

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/api-keys)
- (Optional) Docker & Docker Compose

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/tushantrao141-arch/advanced-rag.git
cd advanced-rag
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit the `.env` file and add your OpenAI API key:

```
OPENAI_API_KEY="sk-..."
```

## Usage

### Run CLI Pipeline

```bash
python main.py
```

This will:
1. Download the `jamescalam/ai-arxiv-chunked` dataset from HuggingFace
2. Build (or load) a FAISS index of document embeddings
3. Retrieve and rerank relevant documents for a sample query
4. Generate an answer using GPT-3.5-turbo

### Run Streamlit Web UI

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. The UI lets you:
- Enter custom queries
- Adjust Retriever Top-K and Reranker Top-N
- View source documents with similarity scores
- See generated answers with timing breakdowns

### Run with Docker Compose

```bash
docker-compose up --build
```

### Run Tests

```bash
pytest tests/ -v
```

> **Note**: The reranker and retriever tests download model weights on first run (~250 MB). Subsequent runs use cached models.

## Configuration

All configuration is centralized in [`config.py`](config.py):

| Parameter | Default | Description |
|---|---|---|
| `DATASET_NAME` | `jamescalam/ai-arxiv-chunked` | HuggingFace dataset |
| `EMBEDDING_MODEL_NAME` | `all-mpnet-base-v2` | SentenceTransformer model |
| `EMBEDDING_DIMENSION` | `768` | Embedding vector size |
| `RETRIEVER_TOP_K` | `20` | Number of documents retrieved |
| `RERANKER_MODEL_NAME` | `sentence-transformers/msmarco-distilbert-base-v3` | Reranker model |
| `RERANKER_TOP_N` | `5` | Documents after reranking |
| `LLM_MODEL_NAME` | `gpt-3.5-turbo` | OpenAI model (via env) |
| `LLM_TEMPERATURE` | `0.0` | Generation temperature |
| `BATCH_SIZE` | `32` | Embedding batch size |
| `CHUNK_SIZE` | `512` | Text chunk size (chars) |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |

## License
This project is open source and available under the [MIT License](LICENSE).