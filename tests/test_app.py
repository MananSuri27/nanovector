import json

import numpy as np
import pytest

from app.app import app


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config["TESTING"] = True
    client = app.test_client()
    yield client


def test_create_table(client):
    """Test the /create route."""
    # Define test data
    test_data = {
        "table_name": "test_table",
        "embeddings": np.random.rand(20, 256).tolist(),
    }

    # Send a POST request to create a table
    response = client.post("/create", json=test_data)

    assert response.status_code == 201


def test_query(client):
    """Test the /query route."""
    # Define test data
    test_data = {
        "k": "3",
        "query_vector": np.random.rand(256).tolist(),
    }

    # Send a POST request to create a table
    response = client.post("/test_table/query", json=test_data)

    print(response)

    assert response.status_code == 200


def test_details(client):
    """Test the /details route."""
    # Define test data
    test_data = {
        "k": "3",
        "query_vector": np.random.rand(256).tolist(),
    }

    # Send a POST request to create a table
    response = client.get("/test_table/details")

    # print(response)

    assert response.status_code == 200


def test_add(client):
    """Test the /add route."""
    # Define test data
    test_data = {
        "vector": np.random.rand(256).tolist(),
    }

    # Send a POST request to create a table
    response = client.get("/test_table/details")

    # print(response)

    assert response.status_code == 200
