from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def test_health() -> None:
    app = create_app(settings=Settings(_env_file=None, openai_api_key="test"))

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model": "deepseek-flash"}
