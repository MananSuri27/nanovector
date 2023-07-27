import numpy as np


def normalise_embeddings(embeddings: np.array) -> np.array:
    EPS = 1e-6
    return embeddings / (np.linalg.norm(embeddings, keepdims=True) + EPS)
