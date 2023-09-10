from flask import Flask, jsonify, request
import numpy as np

from tables.db import VectorDB
from tables.table import VectorTable

from utils.config import IndexConfig
app = Flask(__name__)

tables = VectorDB()


@app.route("/create", methods=["POST"])
def create_table():
    data = request.get_json()

    # Extract table creation parameters from JSON data
    table_name = data.get("table_name")
    description = data.get("description", None)  # Use default value None if "description" is not provided
    embeddings_path = data.get("path", None)  # Use default value None if "path" is not provided
    embeddings = data.get("embeddings", None)  # Use default value None if "embeddings" is not provided

    # Check if both embeddings_path and embeddings are None
    if embeddings_path is None and embeddings is None:
        return jsonify(message="Both 'embeddings_path' and 'embeddings' are missing"), 400

    # If both embeddings_path and embeddings are provided, log a warning
    if embeddings_path is not None and embeddings is not None:
        app.logger.warning("Both 'embeddings_path' and 'embeddings' provided; 'embeddings_path' will be used.")

    # Load embeddings from the specified path if embeddings is not provided
    if embeddings is None:
        if embeddings_path is None:
            return jsonify(message="'embeddings' or 'embeddings_path' must be provided"), 400
        embeddings = np.load(embeddings_path)

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
def table_details(table):
    if table not in tables:
        return jsonify(message="Table not found"), 404


    table_details = tables.get_table(table)
    return jsonify(table_data), 200


@app.route("/<table>/delete", methods=["DELETE"])
def delete_table(table):
    if table not in tables:
        return jsonify(message="Table not found"), 404

    # Looks good

    del tables[table]
    return jsonify(message=f"Table {table} deleted successfully"), 200


@app.route("/<table>/add", methods=["POST"])
def add_to_table(table):
    if table not in tables:
        return jsonify(message="Table not found"), 404

    # check if row can be added

    data = request.get_json()
    tables[table].append(data)
    return jsonify(message="Row added successfully"), 201


@app.route("/<table>/query", methods=["POST"])
def query_table(table):
    if table not in tables:
        return jsonify(message="Table not found"), 404

    query = request.get_json()

    #  rework whole thing
    filtered_data = [
        row
        for row in tables[table]
        if all(row.get(key) == value for key, value in query.items())
    ]
    return jsonify(filtered_data), 200


@app.route("/list_tables", methods=["GET"])
def list_tables():
    # works: do we also want to save timestamp?
    return jsonify(list(tables.keys())), 200


@app.errorhandler(400)
@app.errorhandler(404)
@app.errorhandler(500)
def handle_invalid_request(error):
    response = jsonify(message="Invalid request", error=str(error))
    response.status_code = error.code
    return response


if __name__ == "__main__":
    app.run(debug=True)
