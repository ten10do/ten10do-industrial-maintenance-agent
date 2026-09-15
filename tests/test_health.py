"""Smoke tests for the service entry point and package wiring."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_schemas_importable() -> None:
    from app.schemas import MaintenanceQuery, MaintenanceResponse

    query = MaintenanceQuery(query="Pump vibration is abnormal")
    response = MaintenanceResponse(answer="")
    assert query.query
    assert response.answer == ""
