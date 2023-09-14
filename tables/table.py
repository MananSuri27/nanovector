import uuid
from datetime import datetime
from typing import Union

import numpy as np

from index.abstract_index import AbstractIndex
from utils.config import IndexConfig
from utils.initialise_index import initialise_index


class VectorTable:
    """
    A class representing a vector table.

    Attributes:
        table_name (str): The name of the table.
        config (IndexConfig): The configuration for the table.
        embeddings (np.array): The embeddings stored in the table.
        description (str, optional): A description of the table (default is None).
        use_embedder (bool, optional): Whether to use an embedder (default is False).
        model_name (str, optional): Model string of a sentence transformer to use for embedding (default is None).
        has_texts (bool, optional): Whether the table has associated texts (default is False).
        texts (list, optional): A list of associated texts (default is None).
    """

    def __init__(
        self,
        table_name: str,
        config: IndexConfig,
        embeddings: np.array,
        description: str = None,
        use_embedder: bool = False,
        model_name: str = None,
        has_texts: bool = False,
        texts: list = None,
    ):
        """
        Initialize a VectorTable instance.

        Args:
            table_name (str): The name of the table.
            config (IndexConfig): The configuration for the table.
            embeddings (np.array): The embeddings stored in the table.
            description (str, optional): A description of the table (default is None).
            use_embedder (bool, optional): Whether to use an embedder (default is False).
            model_name (str, optional): Model string of a sentence transformer to use for embedding (default is None).
            has_texts (bool, optional): Whether the table has associated texts (default is False).
            texts (list, optional): A list of associated texts (default is None).
        """
        self._uuid = uuid.uuid4()
        self._created_at = datetime.utcnow()
        self._last_queried_at = None
        self._table_name = table_name
        self._index = initialise_index(config, embeddings)
        self._config = config
        self.description = description
        self._use_embedder = use_embedder
        self._model_name = model_name
        self._has_texts = has_texts or texts != None
        self._texts = texts

    @property
    def uuid(self) -> uuid.UUID:
        """Get the UUID of the table."""
        return self._uuid

    @property
    def created_at(self) -> datetime:
        """Get the creation timestamp of the table."""
        return self._created_at

    @property
    def last_queried_at(self) -> datetime:
        """Get the timestamp of the last query to the table."""
        return self._last_queried_at

    @last_queried_at.setter
    def last_queried_at(self, value: datetime):
        """
        Set the timestamp of the last query to the table.

        Args:
            value (datetime): The timestamp of the last query.

        Raises:
            ValueError: If the provided timestamp is not greater than the current last queried timestamp.
        """
        if self._last_queried_at is not None and value <= self._last_queried_at:
            raise ValueError(
                "Timestamp must be greater than the current last queried timestamp."
            )
        self._last_queried_at = value

    @property
    def table_name(self) -> str:
        """Get the name of the table."""
        return self._table_name

    @property
    def index(self) -> AbstractIndex:
        """Get the index associated with the table."""
        return self._index

    @property
    def config(self) -> IndexConfig:
        """Get the configuration of the table."""
        return self._config

    @property
    def model_name(self):
        return self._model_name

    @property
    def use_embedder(self):
        return self._use_embedder

    @property
    def has_texts(self):
        return self._has_texts

    @property
    def texts(self):
        return self._texts

    def __repr__(self) -> str:
        return f"VectorTable(uuid={self.uuid}, created_at={self.created_at}, last_queried_at={self.last_queried_at}, table_name={self.table_name}, table_description={self.description}, config={self.config}, num_rows={len(self.index)}, use_embedder={self.use_embedder}, self.has_texts={self.has_texts} )"

    def __str__(self) -> str:
        return f"VectorTable(uuid={self.uuid}, created_at={self.created_at}, last_queried_at={self.last_queried_at}, table_name={self.table_name}, table_description={self.description}, config={self.config}, num_rows={len(self.index)}, use_embedder={self.use_embedder}, self.has_texts={self.has_texts} )"

    def add_vector(self, vector: np.array, texts: Union[str, list] = None):
        """
        Add a vector to the vector table.

        Args:
            vector (np.array): The vector to be added to the table.
            texts (Union[str, list], optional): An optional text or list of texts associated with the vector (default is None).
        """
        if self.has_texts:
            if isinstance(texts, list):
                self._texts.extend(texts)
            else:
                self._texts.append(texts)

        self._index.add_vector(vector)

    def query(self, query_vector: np.array, k: int = 1):
        """
        Perform a similarity query on the vector table.

        Args:
            query_vector (np.array): The query vector for similarity search.
            k (int, optional): The number of similar vectors to retrieve (default is 1).

        Returns:
            tuple: A tuple containing two arrays: top-k indices and top-k embeddings.

        Example:
            table = VectorTable(table_name="my_table", config=config, embeddings=embeddings)
            query_vector = np.random.rand(1, config.dim_input)
            top_k_indices, top_k_embeddings = table.query(query_vector, k=10)
        """
        top_k_indices_sorted, top_k_embeddings = self._index.get_similarity(
            query_vector, k
        )

        texts = (
            [self.texts[i] for i in top_k_indices_sorted.tolist()]
            if self.has_texts
            else None
        )
        return top_k_indices_sorted, top_k_embeddings, texts
