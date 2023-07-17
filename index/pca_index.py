from index.abstract_index import AbstractIndex
from utils.utils import normalise_embeddings
import numpy as np
from sklearn.decomposition import PCA


class PCAIndex(AbstractIndex):
    def __init__(self, embeddings: np.array, dimension_input: int, dimension_final: int ,normalise=False):
        super().__init__(len(embeddings), dimension_input)
        if(embeddings.shape[1]!=dimension_input):
            raise ValueError(f"Expected embeddings of dimension {dimension_input} but got {embeddings.shape[1]}")
        
        self.dimension = dimension_input
        self.dimension_final = dimension_final
        self.normalise = normalise

        self.PCA = PCA(self.dimension_final)
        
        self.embeddings = self.PCA.fit_transform(embeddings if not self.normalise else normalise_embeddings(embeddings))
        
        

    def add_vector(self, vector: np.array):
        if(vector.shape[1]!=self.dimension):
            raise ValueError(f"Expected vector of dimension {self.dimension} but got {len(vector)}")
        vector = vector if not self.normalise else normalise_embeddings(vector)
        vector = self.PCA.transform(vector)
        self.embeddings = np.vstack([self.embeddings, vector])
        self.num_vectors = self.num_vectors+1
    
    def get_similarity(self, query: np.array, k:int):
        k = k if k<=self.num_vectors else self.num_vectors

        query = query if not self.normalise else normalise_embeddings(query)
        query = self.PCA.transform(query)
        scores = np.dot(query, self.embeddings.T)
        top_k_indices = np.argpartition(-scores, kth=k)[:k]
        top_k_indices_sorted = top_k_indices[np.argsort(top_k_indices)]
        top_k_embeddings = self.embeddings[top_k_indices_sorted]

        return top_k_indices_sorted, top_k_embeddings
