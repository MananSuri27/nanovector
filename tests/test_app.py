import pytest
import numpy as np
from app.app import app

@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    client = app.test_client()
    yield client

def test_create_table(client):
    """Test the /create route."""
    # Define test data
    test_data = {
        "table_name": "test_table",
        "embeddings": np.random.rand(20,256).tolist(),
    }

    # Send a POST request to create a table
    response = client.post("/create", json=test_data)

    assert response.status_code == 201
