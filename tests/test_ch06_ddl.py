from pathlib import Path


def test_ch06_ddl_has_refund_requests():
    sql = Path("sql/ch06-ddl.sql").read_text(encoding="utf-8")
    assert "SET NAMES utf8mb4" in sql
    assert "CREATE TABLE IF NOT EXISTS refund_requests" in sql
    assert "ENUM('退款','退货')" in sql
    assert "ENUM('待处理','已处理')" in sql
    assert "PRIMARY KEY (refund_no)" in sql
    assert "KEY idx_conversation_id (conversation_id)" in sql
    assert "FOREIGN KEY (conversation_id)" in sql
