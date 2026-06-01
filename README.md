# Advanced RAG Pipeline

A two-stage Retrieval-Augmented Generation (RAG) pipeline featuring semantic search with FAISS, cross-encoder reranking, and OpenAI LLM response generation.

## Features
- **Semantic Search**: FAISS index utilizing SentenceBERT (`all-mpnet-base-v2`) embeddings.
- **Reranking**: Cross-encoder model (`msmarco-distilbert-base-v3`) to rerank retrieved documents.
- **Response Generation**: OpenAI GPT-3.5 via LangChain.
- **Evaluation**: Built-in retrieval metrics (Recall@K, MRR, NDCG@K).
- **Interface**: Streamlit web application.
- **Containerization**: Docker configuration.

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Kdotseth7/advanced-rag.git
   cd advanced-rag
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   Create a `.env` file from the example:
   ```bash
   cp .env.example .env
   ```
   Add your `OPENAI_API_KEY` to the `.env` file.

## Usage

### Run CLI Pipeline
```bash
python main.py
```

### Run Streamlit UI
```bash
streamlit run app.py
```

### Run with Docker Compose
```bash
docker-compose up --build
```

### Run Tests
```bash
pytest tests/ -v
```