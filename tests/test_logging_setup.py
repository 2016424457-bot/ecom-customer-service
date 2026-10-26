import logging

from app.config import Settings
from app.main import _configure_logging, create_app


def test_log_file_handler_writes(tmp_path):
    path = tmp_path / "app.log"

    _configure_logging(
        Settings(
            _env_file=None,
            openai_api_key="test",
            log_file=path,
        )
    )
    logging.getLogger("app.test").info("hello ch07")
    for handler in logging.getLogger().handlers:
        handler.flush()

    assert "hello ch07" in path.read_text(encoding="utf-8")


def test_create_app_logs_context_budget(tmp_path):
    path = tmp_path / "app.log"

    create_app(
        settings=Settings(
            _env_file=None,
            openai_api_key="test",
            log_file=path,
        )
    )
    for handler in logging.getLogger().handlers:
        handler.flush()

    text = path.read_text(encoding="utf-8")
    assert "context_budget" in text
    assert "history=" in text