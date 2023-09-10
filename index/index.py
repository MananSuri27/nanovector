import numpy as np

from index.abstract_index import AbstractIndex
from utils.utils import normalise_embeddings


class Index(AbstractIndex):
    """
    A class representing an index for vector search.

    Attributes:
        embeddings (np.array): The array of embeddings indexed in the table.
        dimension (int): The dimensionality of the embeddings.
        normalise (bool): Whether the embeddings are to be normalized.

    Methods:
        add_vector(vector): Add a vector to the index.
        get_similarity(query_vector, k): Retrieve the top-k similar vectors to a query vector.

    Example:
        embeddings = np.random.rand(100, 256)
        index = Index(embeddings, dimension=256, normalise=False)
        query = np.random.rand(1, 256)
        indices, top_k_vectors = index.get_similarity(query, k=10)
    """

    def __init__(self, embeddings: np.array, dimension: int, normalise=False):
        """
        Initialize an Index instance.

        Args:
            embeddings (np.array): The array of embeddings indexed in the table.
            dimension (int): The dimensionality of the embeddings.
            normalise (bool, optional): Whether the embeddings are to be normalized (default is False).

        Raises:
            ValueError: If the shape of embeddings is not compatible with the specified dimension.
        """
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
        """
        Add a vector to the index.

        Args:
            vector (np.array): The vector to be added to the index.

        Raises:
            ValueError: If the shape of the provided vector is not compatible with the index dimension.
        """
        if len(vector.shape) == 1 and len(vector) == self.dimension:
            vector = vector.reshape(1, self.dimension)
        elif len(vector.shape) == 1:
            raise ValueError(
                f"Expected vector of dimension {self.dimension} but got {len(vector)}"
            )

        if vector.shape[1] != self.dimension:
            raise ValueError(
                f"Expected vector of dimension {self.dimension} but got {vector.shape[1]}"
            )
        vector = vector if not self.normalise else normalise_embeddings(vector)
        self.embeddings = np.vstack([self.embeddings, vector])
        self.num_vectors = self.num_vectors + vector.shape[0]

    def get_similarity(self, query_vector: np.array, k: int):
        """
        Retrieve the top-k similar vectors to a query vector.

        Args:
            query_vector (np.array): The query vector for similarity search.
            k (int): The number of similar vectors to retrieve.

        Returns:
            tuple: A tuple containing two arrays: top-k indices and top-k embeddings.

        Raises:
            ValueError: If k is less than zero or the shape of the query vector is not compatible with the index dimension.
            NotImplementedError: If multi-vector queries are not supported.
        """
        if k < 0:
            raise ValueError(f"Expected k>0 got k={k}")
        if (
            len(query_vector.shape) == 2
            and query_vector.shape[1] == self.dimension
            and len(query_vector) == 1
        ):
            query_vector = query_vector.reshape(
                self.dimension,
            )
        elif len(query_vector.shape) > 1:
            raise NotImplementedError("Multi-vector query not supported yet.")
        elif query_vector.shape[0] != self.dimension:
            raise ValueError(
                f"Expected vector of dimension {self.dimension} but got {query_vector.shape[0]}"
            )

        # Determine the actual number of neighbors based on the available vectors
        num_neighbors = min(k, self.num_vectors)

        # Normalize the query vector if required
        normalized_query = (
            query_vector if not self.normalise else normalise_embeddings(query_vector)
        )

        # Compute the dot product (similarity scores) between the normalized query and all embeddings
        similarity_scores = np.dot(normalized_query, self.embeddings.T)

        if num_neighbors != self.num_vectors:
            # Get the indices of the top k similarity scores using argpartition
            top_k_indices = np.argpartition(-similarity_scores, kth=num_neighbors)[
                :num_neighbors
            ]
        else:
            top_k_indices = np.argsort(-similarity_scores)

        # Sort the indices in ascending order (to preserve the original order)
        top_k_indices_sorted = top_k_indices[np.argsort(top_k_indices)]

        # Get the top k embeddings based on the sorted indices
        top_k_embeddings = self.embeddings[top_k_indices_sorted]

        return top_k_indices_sorted, top_k_embeddings
