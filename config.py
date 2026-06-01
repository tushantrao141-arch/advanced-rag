import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

DATASET_NAME = "jamescalam/ai-arxiv-chunked"
DATASET_SPLIT = "train"

EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
EMBEDDING_DIMENSION = 768

INDEX_DIR = ".indices"
INDEX_PATH = os.path.join(INDEX_DIR, "index_latest.idx")

RETRIEVER_TOP_K = 20

RERANKER_MODEL_NAME = "sentence-transformers/msmarco-distilbert-base-v3"
RERANKER_TOP_N = 5

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
LLM_TEMPERATURE = 0.0

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
