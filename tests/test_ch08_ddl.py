from pathlib import Path

from app.db.models import ToolAuditLog


def test_ch08_ddl_has_tool_audit_logs():
    sql = Path("sql/ch08-ddl.sql").read_text(encoding="utf-8")
    assert "SET NAMES utf8mb4" in sql
    assert "CREATE TABLE tool_audit_logs" in sql
    assert "ENUM('builtin','mcp')" in sql
    assert "ENUM('成功','失败','超时','校验拦下','权限拒绝')" in sql
    assert "arguments       JSON" in sql
    assert "KEY idx_conversation_id (conversation_id)" in sql
    assert "KEY idx_tool_name (tool_name)" in sql
    assert "KEY idx_status (status)" in sql
    assert "FOREIGN KEY" not in sql
    assert "ENGINE=InnoDB DEFAULT CHARSET=utf8mb4" in sql


def test_tool_audit_model_matches_ddl():
    columns = ToolAuditLog.__table__.columns
    assert set(columns.keys()) == {
        "id",
        "conversation_id",
        "tool_call_id",
        "tool_name",
        "tool_source",
        "mcp_server",
        "arguments",
        "result_summary",
        "status",
        "error_message",
        "retry_count",
        "duration_ms",
        "created_at",
    }
    assert columns["conversation_id"].nullable is True
    assert columns["mcp_server"].nullable is True
    assert not any(column.foreign_keys for column in columns)