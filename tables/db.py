from datetime import datetime
from typing import Optional, Union

import numpy as np

from tables.table import VectorTable


class VectorDB:
    """
    A class representing a database for storing vector tables.

    Attributes:
        tables (dict): A dictionary of vector tables, where keys are table names and values are VectorTable instances.
        created_at (datetime): The timestamp when the database was created.

    Methods:
        add_table(table): Add a new vector table to the database.
        delete_table(table_name): Delete a vector table from the database.
        __len__(): Get the number of vector tables in the database.
        list_tables(): List all vector tables in the database with their creation timestamps.
        __repr__(): Get a string representation of the database.

    Example:
        db = VectorDB()
        db.add_table(table1)
        db.add_table(table2)
        print(len(db))  # Prints the number of tables
    """

    def __init__(self):
        """
        Initialize a VectorDB instance.
        """
        self.created_at = datetime.utcnow()
        self._tables = {}

    def get_table(self, table_name: str):
        """
        Get a table with desired table name

        Args:
            table_name (str): Name of table to get

        Raises:
            ValueError: if table doesn't exist
        """
        if table_name not in self._tables.keys():
            raise ValueError(f"Table {table_name} doesn't exist in the database.")

        return self._tables[table_name]

    @property
    def tables(self):
        return self._tables

    def add_table(self, table: VectorTable):
        """
        Add a vector table to the database.

        Args:
            table (VectorTable): The vector table to add to the database.

        Raises:
            ValueError: If a table with the same name already exists in the database.

        Example:
            db = VectorDB()
            table_name = "my_table"
            config = IndexConfig(dim_input=512, dim_final=512)
            embeddings = np.random.rand(100, 512)
            description = "My sample table"
            vector_table = VectorTable(table_name, config, embeddings, description)
            db.add_table(vector_table)
        """
        if table.table_name in self._tables:
            raise ValueError(
                f"Table with name {table.table_name} already exists, re-initialize table."
            )
        self.tables[table.table_name] = table

    def delete_table(self, table_name: str):
        """
        Delete a vector table from the database.

        Args:
            table_name (str): The name of the vector table to delete from the database.

        Raises:
            ValueError: If the specified table does not exist in the database.

        Example:
            db = VectorDB()
            table_name = "my_table"
            config = IndexConfig(dim_input=512, dim_final=512)
            embeddings = np.random.rand(100, 512)
            description = "My sample table"
            vector_table = VectorTable(table_name, config, embeddings, description)
            db.add_table(vector_table)

            # Deleting the table
            db.delete_table(table_name)
        """
        self.check_table(table_name)

        del self.tables[table_name]

    def add_vector(
        self,
        table_name: str,
        vector: np.array,
        texts: Union[str, list[str], None] = None,
    ):
        """
        Add a vector to a specified table.

        Args:
            table_name (str): The name of the table to which the vector will be added.
            vector (np.array): The vector to be added to the table.
            texts ( Union[str, list[str], None]): corresponding texts to be added, defaults to None

        Raises:
            ValueError: If the specified table does not exist.

        Example:
            db = VectorDB()
            table_name = "my_table"
            vector = np.random.rand(1, config.dim_input)
            db.add_vector(table_name, vector)
        """
        self.check_table(table_name)
        self._tables[table_name].add_vector(vector, texts)

    def query(self, table_name: str, query_vector: np.array, k: int = 1):
        """
        Perform a similarity query on a specified table.

        Args:
            table_name (str): The name of the table to query.
            query_vector (np.array): The query vector for similarity search.
            k (int, optional): The number of similar vectors to retrieve (default is 1).

        Returns:
            tuple: A tuple containing two arrays: top-k indices and top-k embeddings.

        Raises:
            ValueError: If the specified table does not exist in the database.

        Example:
            db = VectorDB()
            table_name = "my_table"
            query_vector = np.random.rand(1, config.dim_input)
            top_k_indices, top_k_embeddings = db.query(table_name, query_vector, k=10)
        """
        self.check_table(table_name)
        return self._tables[table_name].query(query_vector, k)

    def update_time(self, table_name: str):
        """
        Update the last queried timestamp for a table.

        Args:
            table_name (str): The name of the table to update the timestamp for.

        Raises:
            ValueError: If the specified table does not exist in the database.
        """
        self.check_table(table_name)
        self._tables[table_name].last_queried_at = datetime.utcnow()

    def check_table(self, table_name: str):
        """
        Check if a specified table exists in the database.

        Args:
            table_name (str): The name of the table to check.

        Returns:
            bool: True if the table exists, False otherwise.

        Raises:
            ValueError: If the specified table does not exist.

        Example:
            db = VectorDB()
            table_name = "my_table"
            if db.check_table(table_name):
                print(f"Table '{table_name}' exists.")
        """
        if table_name not in self._tables.keys():
            raise ValueError(f"Table '{table_name}' doesn't exist in the database.")
        return True

    def __len__(self):
        """
        Get the number of vector tables in the database.

        Returns:
            int: The number of vector tables.
        """
        return len(self.tables)

    def list_tables(self):
        """
        List all vector tables in the database with their creation timestamps.

        Returns:
            list: A list of tuples containing table names and their creation timestamps.
        """
        return [
            (table_name, self.tables[table_name].created_at)
            for table_name in self.tables
        ]

    def __repr__(self):
        """
        Get a string representation of the database.

        Returns:
            str: A string representation of the database including the number of tables and creation timestamp.
        """
        return (
            f"VectorDB - Number of Tables: {len(self)}, Created At: {self.created_at}"
        )
