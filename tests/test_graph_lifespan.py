from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.graph import runtime as graph_runtime
from app.main import create_app


def test_lifespan_initializes_and_closes_graph(monkeypatch):
    calls: list[tuple[str, object]] = []

    async def fake_init(settings):
        calls.append(("init", settings))

    async def fake_close():
        calls.append(("close", None))

    async def aclose():
        calls.append(("aclose", None))

    monkeypatch.setattr(graph_runtime, "init_graph", fake_init)
    monkeypatch.setattr(graph_runtime, "close_graph", fake_close)

    runtime = SimpleNamespace(
        settings=SimpleNamespace(log_level="INFO"),
        aclose=aclose,
    )
    app = create_app(runtime=runtime)

    with TestClient(app):
        pass

    assert [name for name, _ in calls] == ["init", "close", "aclose"]
