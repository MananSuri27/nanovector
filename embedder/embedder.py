from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, sentences):
        return self.model.encode(sentences)
