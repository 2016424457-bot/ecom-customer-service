from dataclasses import dataclass
from datetime import datetime

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.prompt_builder import PromptBuilder
from app.runtime import Runtime
from app.sessions import ConversationLocks
from tests.fakes import FakeRepository


@dataclass
class FakeChunk:
    id: int
    category: str = "运费"
    questions: str = "退货运费谁出？"
    answer: str = "退货运费由买家承担。"
    section_path: str = "退货与退款政策 / 退货运费与邮费"
    content_type: str = "policy"
    source_doc: str = "returns-policy.md"


@dataclass
class FakeCase:
    id: int
    eval_id: str
    bucket: str
    query: str
    strategy: str
    answer: str
    reason: str
    citations: list | None
    judge_model: str | None
    status: str
    seen_count: int
    first_seen_at: datetime
    last_seen_at: datetime
    resolution: str | None
    resolved_at: datetime | None


class FakeKnowledgeRepository:
    def __init__(self):
        self.chunks = {7: FakeChunk(id=7)}

    async def get_chunk_by_id(self, chunk_id):
        return self.chunks.get(chunk_id)


def make_case(**overrides):
    now = datetime(2026, 10, 22, 12, 0, 0)
    values = dict(
        id=1,
        eval_id="A01",
        bucket="A_policy",
        query="q",
        strategy="hybrid_rerank",
        answer="a",
        reason="r",
        citations=None,
        judge_model="m",
        status="未解决",
        seen_count=1,
        first_seen_at=now,
        last_seen_at=now,
        resolution=None,
        resolved_at=None,
    )
    values.update(overrides)
    return FakeCase(**values)


class FakeFaithRepository:
    def __init__(self):
        self.cases = {1: make_case()}

    async def list_cases(
        self, status=None, bucket=None, limit=50, offset=0
    ):
        items = [
            case
            for case in self.cases.values()
            if (status is None or case.status == status)
            and (bucket is None or case.bucket == bucket)
        ]
        return items[offset : offset + limit], len(items)

    async def get_case(self, case_id):
        return self.cases.get(case_id)

    async def update_status(self, case_id, status, resolution):
        case = self.cases.get(case_id)
        if case is None:
            return None
        if status != "未解决" and not (resolution or "").strip():
            raise ValueError("resolution is required")
        case.status = status
        case.resolution = (
            resolution if status != "未解决" else None
        )
        if status != "未解决":
            case.resolved_at = datetime.now()
        return case


def make_client(tmp_path, clean_unit_env):
    system = tmp_path / "system.txt"
    system.write_text("你是客服。", encoding="utf-8")
    extract = tmp_path / "extract.txt"
    extract.write_text("抽取", encoding="utf-8")
    runtime = Runtime(
        settings=Settings(_env_file=None, openai_api_key="test"),
        prompt_builder=PromptBuilder(
            system_prompt_path=system,
            extract_prompt_path=extract,
            brand_name="示例商城",
            service_hours="9:00-21:00",
        ),
        chat_model=None,
        extraction_model=None,
        locks=ConversationLocks(),
        repository=FakeRepository(),
        knowledge_repository=FakeKnowledgeRepository(),
        faith_repository=FakeFaithRepository(),
    )
    return TestClient(create_app(runtime=runtime))


def test_chunk_detail_returns_source(tmp_path, clean_unit_env):
    client = make_client(tmp_path, clean_unit_env)

    response = client.get("/api/knowledge/chunks/7")

    assert response.status_code == 200
    assert response.json()["section_path"] == (
        "退货与退款政策 / 退货运费与邮费"
    )
    assert response.json()["answer"] == "退货运费由买家承担。"


def test_chunk_detail_404(tmp_path, clean_unit_env):
    client = make_client(tmp_path, clean_unit_env)

    assert client.get("/api/knowledge/chunks/999").status_code == 404


def test_list_faith_cases_filters(tmp_path, clean_unit_env):
    client = make_client(tmp_path, clean_unit_env)

    response = client.get(
        "/api/faith-cases", params={"status": "未解决"}
    )

    assert response.status_code == 200
    assert [
        item["eval_id"] for item in response.json()["items"]
    ] == ["A01"]


def test_patch_faith_case_requires_resolution(tmp_path, clean_unit_env):
    client = make_client(tmp_path, clean_unit_env)

    response = client.patch(
        "/api/faith-cases/1",
        json={"status": "已解决", "resolution": ""},
    )

    assert response.status_code == 422


def test_patch_faith_case_rejects_invalid_status(
    tmp_path, clean_unit_env
):
    client = make_client(tmp_path, clean_unit_env)

    response = client.patch(
        "/api/faith-cases/1", json={"status": "已驳回"}
    )

    assert response.status_code == 422


def test_patch_faith_case_resolves(tmp_path, clean_unit_env):
    client = make_client(tmp_path, clean_unit_env)

    response = client.patch(
        "/api/faith-cases/1",
        json={"status": "已解决", "resolution": "补了知识库"},
    )

    assert response.status_code == 200
    assert response.json()["resolution"] == "补了知识库"
