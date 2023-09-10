import uuid
from datetime import datetime

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
        description (str): A description of the table (optional).
    """

    def __init__(
        self,
        table_name: str,
        config: IndexConfig,
        embeddings: np.array,
        description: str = None,
    ):
        """
        Initialize a VectorTable instance.

        Args:
            table_name (str): The name of the table.
            config (IndexConfig): The configuration for the table.
            embeddings (np.array): The embeddings stored in the table.
            description (str, optional): A description of the table (default is None).
        """
        self._uuid = uuid.uuid4()
        self._created_at = datetime.utcnow()
        self._last_queried_at = None
        self._table_name = table_name
        self._index = initialise_index(config, embeddings)
        self._config = config
        self.description = description

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

    def __repr__(self) -> str:
        return f"VectorTable(uuid={self.uuid}, created_at={self.created_at}, last_queried_at={self.last_queried_at}, table_name={self.table_name}, table_description={self.description}, config={self.config})"
