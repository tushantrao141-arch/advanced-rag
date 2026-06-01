import numpy as np
from embeddings import Embeddings


class Retriever:

    @staticmethod
    def search(documents, embed_model: Embeddings, index, query: str, top_k: int) -> list:
        query_embedding = embed_model.get_embedding([query])
        distances, indices = index.search(np.array(query_embedding), top_k)
        return [documents[int(idx)]['text'] for idx in indices[0] if idx < len(documents)]
