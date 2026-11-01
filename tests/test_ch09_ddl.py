from pathlib import Path

from app.db.models import EvalRun, LowConfidenceQuestion, ReviewQueue


def test_review_queue_model_matches_ddl():
    columns = ReviewQueue.__table__.columns
    assert set(columns.keys()) == {
        "id", "normalized_question", "ai_suggested_answer",
        "occurrence_count", "review_status", "approved_answer",
        "created_at", "updated_at",
    }
    assert columns["occurrence_count"].nullable is False
    assert columns["review_status"].nullable is False


def test_eval_run_model_matches_ddl():
    columns = EvalRun.__table__.columns
    assert set(columns.keys()) == {
        "id", "triggered_by", "dataset_size", "metrics", "created_at",
    }
    assert columns["metrics"].type.__class__.__name__ == "JSON"


def test_low_confidence_question_has_ch09_columns():
    columns = LowConfidenceQuestion.__table__.columns
    assert columns["retrieved_chunks"].type.__class__.__name__ == "JSON"
    fk = next(iter(columns["matched_review_id"].foreign_keys))
    assert fk.target_fullname == "review_queue.id"
    assert fk.ondelete == "SET NULL"


def test_ch09_ddl_has_review_queue_and_eval_runs():
    sql = Path("sql/ch09-ddl.sql").read_text(encoding="utf-8")
    assert "SET NAMES utf8mb4" in sql
    assert "CREATE TABLE review_queue" in sql
    assert "ENUM('待审','通过','驳回')" in sql
    assert "CREATE TABLE eval_runs" in sql
    assert "ENUM('定时','手动')" in sql
    assert "metrics      JSON" in sql
    assert "ENGINE=InnoDB DEFAULT CHARSET=utf8mb4" in sql


def test_ch09_ddl_alters_low_confidence_questions():
    sql = Path("sql/ch09-ddl.sql").read_text(encoding="utf-8")
    assert "ALTER TABLE low_confidence_questions" in sql
    assert "ADD COLUMN retrieved_chunks  JSON" in sql
    assert "ADD COLUMN matched_review_id BIGINT UNSIGNED" in sql
    assert "ADD KEY idx_matched_review_id (matched_review_id)" in sql
    assert "CONSTRAINT fk_lcq_review FOREIGN KEY" in sql
    assert "REFERENCES review_queue (id) ON DELETE SET NULL" in sql


def test_ch09_ddl_creates_review_queue_before_alter():
    sql = Path("sql/ch09-ddl.sql").read_text(encoding="utf-8")
    assert sql.index("CREATE TABLE review_queue") < sql.index(
        "ALTER TABLE low_confidence_questions"
    )