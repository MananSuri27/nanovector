import uuid
from datetime import datetime
import numpy as np

from index.abstract_index import AbstractIndex
from utils.initialise_index import initialise_index
from utils.config import IndexConfig

class VectorTable:
    def __init__(self, table_name: str, config: IndexConfig, embeddings: np.array):
        self._uuid = uuid.uuid4()
        self._created_at = datetime.utcnow()
        self._last_queried_at = None
        self._table_name = table_name
        self._index = initialise_index(config, embeddings)
        self._config = config

    @property
    def uuid(self):
        return self._uuid

    @property
    def created_at(self):
        return self._created_at

    @property
    def last_queried_at(self):
        return self._last_queried_at

    @last_queried_at.setter
    def last_queried_at(self, value):
        # You can customize this setter to restrict or handle updates to last_queried_at
        # For example, you can add validation logic or simply ignore updates
        self._last_queried_at = value

    @property
    def table_name(self):
        return self._table_name

    @property
    def index(self):
        return self._index
    
    @property
    def config(self):
        return self._config

    def __repr__(self):
        return f"CustomTable(uuid={self.uuid}, created_at={self.created_at}, last_queried_at={self.last_queried_at}, table_name={self.table_name}, table_object={self.table_object}, config={self.config})"