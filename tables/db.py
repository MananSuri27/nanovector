from table import VectorTable
from datetime import datetime


from datetime import datetime


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
        self.tables = {}

    def add_table(self, table):
        """
        Add a vector table to the database.

        Args:
            table (VectorTable): The vector table to add to the database.

        Raises:
            ValueError: If a table with the same name already exists in the database.
        """
        if table.table_name in self.tables:
            raise ValueError(
                f"Table with name {table.table_name} already exists, re-initialize table."
            )
        self.tables[table.table_name] = table

    def delete_table(self, table_name):
        """
        Delete a vector table from the database.

        Args:
            table_name (str): The name of the vector table to delete from the database.

        Raises:
            ValueError: If the specified table does not exist in the database.
        """
        if table_name not in self.tables:
            raise ValueError(
                f"Can't delete table {table_name}, it does not exist in the database."
            )
        del self.tables[table_name]

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
