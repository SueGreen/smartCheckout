from fastapi.testclient import TestClient
from api import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_main_page():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"content": {"Hello": "World"}}

def test_predict_item():
    filename = "tmp/tmp.jpg"
    response = client.post(
        "/predict/", files={"file": (filename, open(filename, "rb"))}
    )
    print(f'\n\nResponse: {response.json()}')
    assert response.status_code == 200
