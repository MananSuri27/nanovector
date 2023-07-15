from abc import ABC, abstractmethod

class AbstractIndex(ABC):
    def __init__(self, num_vectors, dimension):
        self.num_vectors = num_vectors
        self.dimension = dimension
    
    def __str__(self):
        return f"NanoVector Table of [{self.num_vectors},{self.dimension}]."
    
    def __repr__(self):
        return self.__str__

    def __len__(self):
        return self.num_vectors

    @abstractmethod
    def add_vector(self, id, embedding):
        pass

    @abstractmethod
    def get_similarity(self, query, k):
        pass