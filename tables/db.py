from table import VectorTable
from datetime import datetime


class VectorDB:
    def __init__(self):
        self.created_at = datetime.utcnow()
        self.tables = {}
        self.num_tables = 0
    
    def add_table(self, table: VectorTable):
        if table.table_name in self.tables.keys():
            raise ValueError(f"Table with name {table.table_name} already exists, re-initialise table.")
        self.tables[table.table_name] = table
        self.num_tables += 1
    
    def delete_table(self, table_name: str):
        if table_name not in self.tables.keys():
            raise ValueError(f"Can't delete table {table_name}, it does not exist in the database.")
        del self.tables[table_name]
    