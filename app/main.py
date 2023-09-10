from flask import Flask, jsonify, request
import numpy as np

from tables.db import VectorDB
from tables.table import VectorTable

from utils.config import IndexConfig
app = Flask(__name__)

tables = VectorDB()

def check_table_exists(route_function):
    def wrapper(table, *args, **kwargs):
        if table not in tables:
            return jsonify(message="Table not found"), 404
        return route_function(table, *args, **kwargs)
    return wrapper

def load_data_from_json(data, variable):
    """
    Load data from the 'data' dictionary based on the 'variable' name and its corresponding '_path' key.
    If '{variable}_path' is provided and exists in 'data', load data from the path.
    If 'variable' is provided and exists in 'data', load data from '{variable}'.
    """
    if f"{variable}_path" is not None and f"{variable}_path" in data:
        return np.load(f"{variable}_path")
    elif variable is not None and variable in data:
        if not isinstance(data[variable], np.array):
            raise ValueError(variable)
        return data[variable]
    else:
        return None

@app.route("/create", methods=["POST"])
def create_table():
    data = request.get_json()

    # Extract table creation parameters from JSON data
    table_name = data.get("table_name")
    description = data.get("description", None)  # Use default value None if "description" is not provided
    embeddings_path = data.get("path", None)  # Use default value None if "path" is not provided
    embeddings = data.get("embeddings", None)  # Use default value None if "embeddings" is not provided


    if embeddings_path is not None and embeddings is not None:
        app.logger.warning("Both 'embeddings_path' and 'embeddings' provided; 'embeddings_path' will be used.")

    embeddings = load_data_from_json(data, "embeddings")

    # Check if embeddings has exactly 2 dimensions
    if len(embeddings.shape) != 2:
        return jsonify(message="Embeddings must have exactly 2 dimensions"), 400

    # Extract other configuration parameters
    pca = data.get("pca", False)  # Use default value False if "pca" is not provided
    normalise = data.get("normalise", True)  # Use default value True if "normalise" is not provided
    dim_input = embeddings.shape[1]
    dim_final = data.get("dim_final", dim_input)  # Use dim_input as default if "dim_final" is not provided


    # Create an IndexConfig object with specified configuration
    config = IndexConfig(dim_input, dim_final, pca, normalise)

    # Create a VectorTable and add it to the database
    table = VectorTable(table_name, config, embeddings, description)
    tables.add_table(table)

    return jsonify(message=f"Table '{table_name}' created successfully"), 201


@app.route("/<table>/details", methods=["GET"])
@check_table_exists
def table_details(table):
    table= tables.get_table(table)
    return jsonify(str(table)), 200


@app.route("/<table>/delete", methods=["DELETE"])
@check_table_exists
def delete_table(table):
    del tables.delete_table(table)
    return jsonify(message=f"Table {table} deleted successfully"), 200


@app.route("/<table>/add", methods=["POST"])
@check_table_exists
def add_to_table(table):

    data = request.get_json()

    vector = data.get("vector", None)
    vector_path = data.get("vector_path", None)

    if vector_path is not None and vector is not None:
        app.logger.warning("Both 'vector_path' and 'vector' provided; 'vector_path' will be used.")

    vector = load_data_from_json(data, "vector")

    tables.add_vector(table, vector)

    return jsonify(message="Row added successfully"), 201


@app.route("/<table>/query", methods=["POST"])
@check_table_exists
def query_table(table):
    if table not in tables:
        return jsonify(message="Table not found"), 404

    data = request.get_json()
    
    k = data.get("k", 1)

    query_vector = data.get("query_vector", None)
    query_vector_path = data.get("query_vector_path", None)

    if query_vector_path is not None and query_vector is not None:
        app.logger.warning("Both 'query_vector_path' and 'query_vector' provided; 'query_vector_path' will be used.")

    query_vector = load_data_from_json(data, "query_vector")

    top_k_indices_sorted, top_k_embeddings = tables.query(table, query_vector, k)

    results = {
        "top_k_indices_sorted": top_k_indices_sorted.tolist(),
        "top_k_embeddings": top_k_embeddings,
    }

    return jsonify(results), 200


@app.route("/list_tables", methods=["GET"])
def list_tables():
    # works: do we also want to save timestamp?
    return jsonify(tables.list_tables()), 200


@app.errorhandler(400)
@app.errorhandler(404)
@app.errorhandler(500)
def handle_invalid_request(error):
    response = jsonify(message="Invalid request", error=str(error))
    response.status_code = error.code
    return response


if __name__ == "__main__":
    app.run(debug=True)
