import pytest
from pydantic import ValidationError

from app.schemas import AfterSalesExtraction, RequestType


def test_accepts_raw_order_id_and_enum_string() -> None:
    extraction = AfterSalesExtraction(
        order_id="订单号：A12345",
        request_type="refund",
        desired_solution="退款",
    )

    assert extraction.order_id == "订单号：A12345"
    assert extraction.request_type is RequestType.REFUND


def test_null_optional_fields_are_allowed() -> None:
    extraction = AfterSalesExtraction(
        order_id=None,
        request_type="other",
        desired_solution=None,
    )

    assert extraction.order_id is None
    assert extraction.desired_solution is None


def test_invalid_request_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        AfterSalesExtraction(request_type="bogus")


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        AfterSalesExtraction(request_type="other", unexpected="x")
