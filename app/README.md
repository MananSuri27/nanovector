# API Documentation

The Vector Database API is a Flask-based RESTful API that facilitates the creation, management, and querying of vector-based databases. This documentation provides a comprehensive guide to the available endpoints and their functionalities.

## Base URL
All API endpoints use the following base URL: `/`

## Endpoints

### 1. Create a Table

- **Endpoint**: `/create`
- **Method**: `POST`
- **Description**: Create a new vector table in the database.
  You can provide:
  - Either the embeddings (`embeddings`) key
  - the path to the embeddings on the local machine where the server is hosted (`embeddings_path`)
  - or just `texts`: a list of strings you want to embed. 
- **Request Body**:
  - `table_name` (string, required): The desired name for the new table.
  - `description` (string, optional): A brief description of the table.
  - `use_embedder` (boolean, optional): If set to `true`, the table is configured for text data.
  - `model_name` (string, optional): The name of the embedding model (required if using text data). This model should be a valid sentence_transformers model.
  - `texts` (list of strings, required if `use_embedder` is `true`): A list of text data for initializing the table.
  - `embeddings` (2D array, optional): Initial embeddings for the table (if not using `texts`).
  - `embeddings_path` (string, optional): Path to a file containing initial embeddings (if not using `texts`).
  - `pca` (boolean, optional): Enable Principal Component Analysis (PCA) on the embeddings.
  - `normalise` (boolean, optional): Normalize the embeddings.
  - `dim_final` (integer, optional): The final dimensionality of the embeddings.

- **Response**:
  - Status Code: 201 (Created)
  - Body: `{"message": "Table '<table_name>' created successfully"}`

### 2. Get Table Details

- **Endpoint**: `/<table>/details`
- **Method**: `GET`
- **Description**: Retrieve details about a specific table.
- **Response**:
  - Status Code: 200 (OK)
  - Body: JSON representation of the table details.

### 3. Delete a Table

- **Endpoint**: `/<table>/delete`
- **Method**: `DELETE`
- **Description**: Delete a table from the database.
- **Response**:
  - Status Code: 200 (OK)
  - Body: `{"message": "Table '<table_name>' deleted successfully"}`

### 4. Add Data to a Table

- **Endpoint**: `/<table>/add`
- **Method**: `POST`
- **Description**: Add data (text or vector) to an existing table.
  - Either the embeddings (`vector`) key
  - the path to the vector on the local machine where the server is hosted (`vector_path`)
  - or just `texts`: a list of strings/ single string you want to add. 
- **Request Body**:
  - `texts` (list of strings, required if the table uses an embedder): A list of text data to add to the table.
  - `vector` (2D array, optional): The vector data to add (if not using `texts`).
  - `vector_path` (string, optional): Path to a file containing vector data (if not using `texts`).
- **Response**:
  - Status Code: 201 (Created)
  - Body: `{"message": "Row added successfully"}`

### 5. Query a Table

- **Endpoint**: `/<table>/query`
- **Method**: `POST`
- **Description**: Query a table to retrieve top-k results based on a query vector or text.
  - Either the embeddings (`query_vector`) key
  - the path to the vector on the local machine where the server is hosted (`query_vector_path`)
  - or just `texts`: a single string you want to query.
- **Request Body**:
  - `k` (integer, optional): The number of top results to retrieve (default is 1).
  - `texts` (list of strings, required if the table uses an embedder): A list of text queries.
  - `query_vector` (2D array, optional): The query vector (if not using `texts`).
  - `query_vector_path` (string, optional): Path to a file containing the query vector (if not using `texts`).
- **Response**:
  - Status Code: 200 (OK)
  - Body: JSON containing query results, including top-k indices, top-k embeddings, and corresponding texts.

### 6. List Tables

- **Endpoint**: `/list_tables`
- **Method**: `GET`
- **Description**: List all the tables in the database.
- **Response**:
  - Status Code: 200 (OK)
  - Body: JSON array of table names.

### Error Handling

The API handles common errors with appropriate status codes and error messages. Possible error codes include:
- 400 (Bad Request): Invalid request parameters or missing required fields.
- 404 (Not Found): The requested table does not exist.
- 500 (Internal Server Error): An internal server error occurred.

## Running the API

To run nanovector API locally, execute the script as follows:

### Docker
Assuming you have docker installed, you can easily use docker to setup the vector server.

1. Pull the Docker image from Docker Hub:
   ```bash
   docker pull manansuri27/nanovector
   ```
   
2. Run the image now,
   ```bash
   docker run manansuri27/nanovector
   ```
The server will be running on `localhost:5000` now.

### GitHub
Follow the steps below to setup the repository and run the server.

1. Clone the repo
   ```bash
   git clone https://github.com/MananSuri27/nanovector.git
   cd nanovector
   ```
3. Create a conda environment, and activate it
   ```bash
   conda create -n nanovector
   conda activate nanovector
   ```
5. Install dependencies
   ```bash
   pip3 install -r requirements.txt
   ```
7. Run the server
   ```bash
   python3 -m app.app
   ```
The server will be running on `localhost:5000` now.
