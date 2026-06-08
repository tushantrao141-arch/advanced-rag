import os
import faiss
import numpy as np
from tqdm import tqdm

import config
from utils import Utils
from dataset import Dataset
from embeddings import Embeddings
from retriever import Retriever
from reranker import Reranker
from llm import LLM


def build_or_load_index(documents, embed_model: Embeddings) -> faiss.Index:
    if os.path.exists(config.INDEX_PATH):
        return faiss.read_index(config.INDEX_PATH)

    index = faiss.IndexFlatL2(config.EMBEDDING_DIMENSION)

    for i in tqdm(range(0, len(documents), config.BATCH_SIZE), desc="Embedding Documents"):
        batch = documents[i:i + config.BATCH_SIZE]
        embeds = embed_model.get_embedding(batch["text"])
        index.add(np.array(embeds))

    faiss.write_index(index, config.INDEX_PATH)
    return index


def main():
    Utils.check_dir(config.INDEX_DIR)

    dataset = Dataset(config.DATASET_NAME, config.DATASET_SPLIT)
    documents = dataset.get_dataset()

    embed_model = Embeddings(config.EMBEDDING_MODEL_NAME)
    index = build_or_load_index(documents, embed_model)

    query = "Can you explain why we would want to do RLHF?"
    docs = Retriever.search(
        documents=documents,
        embed_model=embed_model,
        index=index,
        query=query,
        top_k=config.RETRIEVER_TOP_K,
    )

    reranker = Reranker(config.RERANKER_MODEL_NAME)
    reranked_docs = reranker.rerank(docs, query, top_n=config.RERANKER_TOP_N)
    context = "\n".join([doc[0] for doc in reranked_docs])

    llm = LLM(model=config.LLM_MODEL_NAME, temperature=config.LLM_TEMPERATURE)
    answer = llm.generate(query=query, context=context)

    print("\n" + "=" * 60)
    print(f"Query: {query}")
    print("=" * 60)
    print(f"\n{answer}\n")


if __name__ == "__main__":
    main()