import numpy as np

from index.abstract_index import AbstractIndex
from utils.utils import normalise_embeddings


class Index(AbstractIndex):
    def __init__(self, embeddings: np.array, dimension: int, normalise=False):
        super().__init__(len(embeddings), dimension)
        if embeddings.shape[1] != dimension:
            raise ValueError(
                f"Expected embeddings of dimension {dimension} but got {embeddings.shape[1]}"
            )

        self.embeddings = (
            embeddings if not normalise else normalise_embeddings(embeddings)
        )
        self.dimension = dimension
        self.normalise = normalise

    def add_vector(self, vector: np.array):
        if vector.shape[1] != self.dimension:
            raise ValueError(
                f"Expected vector of dimension {self.dimension} but got {len(vector)}"
            )
        vector = vector if not self.normalise else normalise_embeddings(vector)
        self.embeddings = np.vstack([self.embeddings, vector])
        self.num_vectors = self.num_vectors + vector.shape[0]

    def get_similarity(self, query: np.array, k: int):
        k = k if k <= self.num_vectors else self.num_vectors

        query = query if not self.normalise else normalise_embeddings(query)
        scores = np.dot(query, self.embeddings.T)
        top_k_indices = np.argpartition(-scores, kth=k)[:k]
        top_k_indices_sorted = top_k_indices[np.argsort(top_k_indices)]
        top_k_embeddings = self.embeddings[top_k_indices_sorted]

        return top_k_indices_sorted, top_k_embeddings
