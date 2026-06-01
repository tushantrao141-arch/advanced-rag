from sentence_transformers import SentenceTransformer, util


class Reranker:

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def rerank(self, documents: list, query: str, top_n: int = 10) -> list:
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        document_embeddings = self.model.encode(documents, convert_to_tensor=True)

        similarities = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]
        doc_scores = list(zip(documents, similarities))
        reranked_documents = sorted(doc_scores, key=lambda x: x[1], reverse=True)

        return reranked_documents[:top_n]