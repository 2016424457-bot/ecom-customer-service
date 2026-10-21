"""Route-level tests for structured after-sales extraction (no network)."""

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from app.config import Settings
from app.main import create_app
from app.prompt_builder import PromptBuilder
from app.runtime import Runtime
from app.schemas import AfterSalesExtraction, RequestType
from app.sessions import ConversationLocks


class FakeStatusError(Exception):
    """Upstream error carrying the HTTP status used by ``map_upstream_error``."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"upstream status {status_code}")
        self.status_code = status_code


class FakeExtractionModel:
    """Return a scripted result or raise a scripted exception."""

    def __init__(
        self,
        *,
        result: Any = None,
        error: BaseException | None = None,
    ) -> None:
        self._result = result
        self._error = error
        self.calls: list[list[Any]] = []

    async def ainvoke(self, messages: list[Any]) -> Any:
        self.calls.append(messages)
        if self._error is not None:
            raise self._error
        return self._result




@pytest.fixture
def settings(clean_unit_env: pytest.MonkeyPatch) -> Settings:
    return Settings(_env_file=None, openai_api_key="test")


@pytest.fixture
def prompt_builder(tmp_path: Path) -> PromptBuilder:
    system_prompt_path = tmp_path / "system_prompt.txt"
    system_prompt_path.write_text("你是电商客服。", encoding="utf-8")
    extract_prompt_path = tmp_path / "extract_prompt.txt"
    extract_prompt_path.write_text("抽取原始文本", encoding="utf-8")
    return PromptBuilder(
        system_prompt_path=system_prompt_path,
        extract_prompt_path=extract_prompt_path,
        brand_name="示例商城",
        service_hours="9:00-21:00",
    )


def make_client(
    *,
    settings: Settings,
    prompt_builder: PromptBuilder,
    extraction_model: FakeExtractionModel,
) -> TestClient:
    runtime = Runtime(
        settings=settings,
        prompt_builder=prompt_builder,
        chat_model=object(),
        extraction_model=extraction_model,
        locks=ConversationLocks(),
    )
    return TestClient(create_app(runtime=runtime))


def test_extract_returns_three_fields(
    settings: Settings, prompt_builder: PromptBuilder
) -> None:
    extraction = AfterSalesExtraction(
        order_id="订单号：A12345",
        request_type=RequestType.RETURN,
        desired_solution="退货退款",
    )
    model = FakeExtractionModel(result=extraction)
    client = make_client(
        settings=settings,
        prompt_builder=prompt_builder,
        extraction_model=model,
    )
    user_text = "订单号：A12345 我要退货退款"

    with client:
        response = client.post("/api/chat/extract", json={"text": user_text})

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"order_id", "request_type", "desired_solution"}
    assert body == {
        "order_id": "订单号：A12345",
        "request_type": "return",
        "desired_solution": "退货退款",
    }

    assert model.calls
    messages = model.calls[-1]
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == prompt_builder.extract_prompt
    assert isinstance(messages[-1], HumanMessage)
    assert messages[-1].content == user_text


def test_extract_rate_limit_maps_to_503(
    settings: Settings, prompt_builder: PromptBuilder
) -> None:
    model = FakeExtractionModel(error=FakeStatusError(429))
    client = make_client(
        settings=settings,
        prompt_builder=prompt_builder,
        extraction_model=model,
    )

    with client:
        response = client.post(
            "/api/chat/extract", json={"text": "我要退货"}
        )

    assert response.status_code == 503
    body = response.json()
    assert body["code"] == "upstream_rate_limit"
    assert body["message"]


def test_extract_parse_error_maps_to_502(
    settings: Settings, prompt_builder: PromptBuilder
) -> None:
    model = FakeExtractionModel(
        error=OutputParserException("invalid structured output")
    )
    client = make_client(
        settings=settings,
        prompt_builder=prompt_builder,
        extraction_model=model,
    )

    with client:
        response = client.post(
            "/api/chat/extract", json={"text": "我要退货"}
        )

    assert response.status_code == 502
    assert response.json()["code"] == "parse_error"


def test_extract_validation_error_from_model_maps_to_502(
    settings: Settings, prompt_builder: PromptBuilder
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        AfterSalesExtraction.model_validate({"request_type": "bogus"})
    # with_structured_output raises this during ainvoke when the model's
    # tool-call arguments do not satisfy the schema.
    model = FakeExtractionModel(error=exc_info.value)
    client = make_client(
        settings=settings,
        prompt_builder=prompt_builder,
        extraction_model=model,
    )

    with client:
        response = client.post(
            "/api/chat/extract", json={"text": "我要退货"}
        )

    assert response.status_code == 502
    assert response.json()["code"] == "parse_error"


def test_extract_invalid_model_result_maps_to_502(
    settings: Settings, prompt_builder: PromptBuilder
) -> None:
    model = FakeExtractionModel(result=None)
    client = make_client(
        settings=settings,
        prompt_builder=prompt_builder,
        extraction_model=model,
    )

    with client:
        response = client.post(
            "/api/chat/extract", json={"text": "我要退货"}
        )

    assert response.status_code == 502
    assert response.json()["code"] == "parse_error"


def test_extract_rejects_blank_text(
    settings: Settings, prompt_builder: PromptBuilder
) -> None:
    model = FakeExtractionModel()
    client = make_client(
        settings=settings,
        prompt_builder=prompt_builder,
        extraction_model=model,
    )

    with client:
        response = client.post("/api/chat/extract", json={"text": "  "})

    assert response.status_code == 422
    assert model.calls == []
