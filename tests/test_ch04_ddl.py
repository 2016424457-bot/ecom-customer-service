from pathlib import Path


def test_ch04_ddl_HAS_both_tables_and_enums():
    sql = Path("sql/ch04-ddl.sql").read_text(encoding="utf-8")
    assert "SET NAMES utf8mb4" in sql
    assert "CREATE TABLE IF NOT EXISTS low_confidence_questions" in sql
    assert "CREATE TABLE IF NOT EXISTS faith_cases" in sql
    assert "ENUM('retrieval_low_conf','self_check','user_feedback')" in sql
    assert "ENUM('未解决','已解决','无需解决')" in sql
    assert "UNIQUE KEY uk_eval_id (eval_id)" in sql
