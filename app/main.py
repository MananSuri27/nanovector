from flask import Flask, jsonify, request

app = Flask(__name__)

# Dictionary to store tables
tables = {}


@app.route("/create", methods=["POST"])
def create_table():
    data = request.get_json()
    table_name = data.get("table_name")

    if not table_name or table_name in tables:
        return jsonify(message="Invalid table name or table already exists"), 400

    # create table here
    tables[table_name] = []
    return jsonify(message=f"Table {table_name} created successfully"), 201


@app.route("/<table>/details", methods=["GET"])
def table_details(table):
    if table not in tables:
        return jsonify(message="Table not found"), 404

    # Get details such as number of rows, type etc

    table_data = tables[table]
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
