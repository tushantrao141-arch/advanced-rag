from sentence_transformers import SentenceTransformer


class Embeddings:

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, text) -> list:
        return self.model.encode(text)
