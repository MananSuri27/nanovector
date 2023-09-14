import urllib.parse

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

from embedder.embedder import Embedder
from tables.db import VectorDB
from tables.table import VectorTable
from utils.config import IndexConfig

app = Flask(__name__)
CORS(app)

tables = VectorDB()
models = {}


def check_table_exists(route_function):
    def wrapper(table, *args, **kwargs):
        if not tables.check_table(table):
            return jsonify(message="Table not found"), 404
        return route_function(table, *args, **kwargs)

    wrapper.__name__ = route_function.__name__
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
        return np.array(data[variable])
    else:
        raise ValueError("Either path or  embeddings should be provided.")
        return None


@app.route("/", methods=["GET"])
def trying():
    return "hello"


@app.route("/create", methods=["POST"])
def create_table():
    data = request.get_json()

    table_name = data.get("table_name")
    table_name = table_name.replace("/", "_")

    if table_name != urllib.parse.quote(table_name, safe="/"):
        raise ValueError(
            "table_name contains characters that are not suitable for a URL. Please use only letters, numbers, hyphens, or underscores."
        )

    description = data.get("description", None)
    use_embedder = data.get("use_embedder", False)
    model_name = data.get("model_name", None)

    if use_embedder:
        texts = data.get("texts", None)
        print(texts)
        if texts == None or model_name == None:
            raise AssertionError(
                "use_embedder not possible, either texts are missing or model_name is missing."
            )
        if model_name not in models:
            models[model_name] = Embedder(model_name)

        embeddings = models[model_name].generate_embeddings(texts)
    else:
        embeddings_path = data.get("embeddings_path", None)
        embeddings = data.get("embeddings", None)
        if embeddings_path is not None and embeddings is not None:
            app.logger.warning(
                "Both 'embeddings_path' and 'embeddings' provided; 'embeddings_path' will be used."
            )
        embeddings = load_data_from_json(data, "embeddings")

    # Check if embeddings has exactly 2 dimensions
    if len(embeddings.shape) != 2:
        return jsonify(message="Embeddings must have exactly 2 dimensions"), 400

    # Extract other configuration parameters
    pca = data.get("pca", False)
    normalise = data.get("normalise", True)
    dim_input = embeddings.shape[1]
    dim_final = data.get("dim_final", dim_input)

    # Create an IndexConfig object with specified configuration
    config = IndexConfig(dim_input, dim_final, pca, normalise)

    # Create a VectorTable and add it to the database
    table = VectorTable(
        table_name,
        config,
        embeddings,
        description,
        use_embedder,
        model_name,
        texts=texts,
    )
    tables.add_table(table)

    return jsonify(message=f"Table '{table_name}' created successfully"), 201


@app.route("/<table>/details", methods=["GET"])
@check_table_exists
def table_details(table):
    table = tables.get_table(table)
    return jsonify(str(table)), 200


@app.route("/<table>/delete", methods=["DELETE"])
@check_table_exists
def delete_table(table):
    tables.delete_table(table)
    return jsonify(message=f"Table {table} deleted successfully"), 200


@app.route("/<table>/add", methods=["POST"])
@check_table_exists
def add_to_table(table):
    data = request.get_json()

    texts = data.get("texts", None)

    if tables.get_table(table).use_embedder:
        if texts == None:
            raise ValueError(
                "Table is configured to work with texts, 'texts' field empty in request."
            )

        vector = models[tables.get_table(table).model_name].generate_embeddings(texts)
    else:
        vector = data.get("vector", None)
        vector_path = data.get("vector_path", None)

        if vector_path is not None and vector is not None:
            app.logger.warning(
                "Both 'vector_path' and 'vector' provided; 'vector_path' will be used."
            )

        vector = load_data_from_json(data, "vector")

    tables.add_vector(table, vector, texts)

    return jsonify(message="Row added successfully"), 201


@app.route("/<table>/query", methods=["POST"])
@check_table_exists
def query_table(table):
    data = request.get_json()

    k = data.get("k", 1)
    k = int(k)

    if tables.get_table(table).use_embedder:
        texts = data.get("texts", None)
        if texts == None:
            raise ValueError(
                "Table is configured to work with texts, 'texts' field empty in request."
            )

        query_vector = models[tables.get_table(table).model_name].generate_embeddings(
            texts
        )
    else:
        query_vector = data.get("query_vector", None)
        query_vector_path = data.get("query_vector_path", None)

        if query_vector_path is not None and query_vector is not None:
            app.logger.warning(
                "Both 'query_vector_path' and 'query_vector' provided; 'query_vector_path' will be used."
            )

        query_vector = load_data_from_json(data, "query_vector")

    top_k_indices_sorted, top_k_embeddings, texts = tables.query(table, query_vector, k)

    results = {
        "top_k_indices_sorted": top_k_indices_sorted.tolist(),
        "top_k_embeddings": top_k_embeddings.tolist(),
        "texts": texts,
    }

    tables.update_time(table)

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
