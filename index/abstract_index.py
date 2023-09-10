from abc import ABC, abstractmethod


class AbstractIndex(ABC):
    """
    An abstract base class representing an index for vector search.

    Attributes:
        num_vectors (int): The number of vectors indexed in the table.
        dimension (int): The dimensionality of the vectors.

    Methods:
        __str__(): Get a string representation of the index.
        __repr__(): Get a string representation of the index.
        __len__(): Get the number of vectors in the index.
        add_vector(id, embedding): Add a vector to the index.
        get_similarity(query, k): Retrieve the top-k similar vectors to a query vector.

    Example:
        class MyIndex(AbstractIndex):
            def add_vector(self, id, embedding):
                # Implement the method to add a vector to the index.

            def get_similarity(self, query, k):
                # Implement the method to retrieve similar vectors.
    """

    def __init__(self, num_vectors, dimension):
        """
        Initialize an AbstractIndex instance.

        Args:
            num_vectors (int): The number of vectors indexed in the table.
            dimension (int): The dimensionality of the vectors.
        """
        self.num_vectors = num_vectors
        self.dimension = dimension

    def __str__(self):
        """
        Get a string representation of the index.

        Returns:
            str: A string describing the index.
        """
        return f"NanoVector Table of [{self.num_vectors},{self.dimension}]."

    def __repr__(self):
        """
        Get a string representation of the index.

        Returns:
            str: A string describing the index.
        """
        return self.__str__

    def __len__(self):
        """
        Get the number of vectors in the index.

        Returns:
            int: The number of vectors.
        """
        return self.num_vectors

    @abstractmethod
    def add_vector(self, vector):
        """
        Add a vector to the index.

        Args:
            vector: The vector to be added to the index.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def get_similarity(self, query, k):
        """
        Retrieve the top-k similar vectors to a query vector.

        Args:
            query: The query vector for similarity search.
            k (int): The number of similar vectors to retrieve.

        Returns:
            tuple: A tuple containing two arrays: top-k indices and top-k embeddings.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        pass
