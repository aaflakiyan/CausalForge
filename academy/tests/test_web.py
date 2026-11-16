from fastapi.testclient import TestClient
from physical_ai_lab.app import app

client = TestClient(app)


def test_homepage():
    res = client.get("/")
    assert res.status_code == 200
    assert "Physical AI" in res.text


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["engine"] == "mujoco"


def test_validation():
    assert client.post("/api/step", json={"steps": 0}).status_code == 422
    assert client.post("/api/config", json={"x": 100}).status_code == 422
