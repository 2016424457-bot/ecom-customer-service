import pytest

from app.config import Settings


def test_ch09_settings_defaults(clean_unit_env):
    settings = Settings(_env_file=None, openai_api_key="test")
    assert settings.langfuse_enabled is False
    assert settings.langfuse_base_url == "https://cloud.langfuse.com"
    assert settings.evidence_confidence_threshold == pytest.approx(0.50)
    assert settings.evidence_effective_floor == pytest.approx(0.2)
    assert settings.flywheel_enabled is True
    assert settings.flywheel_dedup_candidate_limit == 100
    assert settings.eval_schedule_enabled is False
    assert settings.eval_interval_hours == 24
    assert settings.review_queue_page_size == 20