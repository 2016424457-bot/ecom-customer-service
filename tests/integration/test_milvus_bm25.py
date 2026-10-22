import pytest

from app.config import Settings
from app.knowledge import milvus_client
from app.knowledge.milvus_client import VectorRecord

pytestmark = pytest.mark.integration
pytest.importorskip("jieba")


def test_bm25_and_hybrid_hit_model_number(tmp_path, clean_unit_env):
    settings = Settings(
        _env_file=None,
        openai_api_key="test",
        milvus_uri=str(tmp_path / "bm25.db"),
    )
    milvus_client.reset_client_cache()
    client = milvus_client.get_client(settings)
    milvus_client.ensure_collection(
        client, "knowledge", analyzer_type="jieba"
    )
    milvus_client.upsert_vectors(
        client,
        [
            VectorRecord(
                id=1,
                text="商品规格 ANC-06 曜石黑 降噪深度 48 dB 价格 599 元",
                vector=[0.1] * 1024,
                category="商品规格",
                content_type="spec",
                section_path="商品规格",
                source_doc="product-specs.md",
            ),
            VectorRecord(
                id=2,
                text="退货运费由买家承担",
                vector=[0.2] * 1024,
                category="退货与退款政策 / 退货运费与邮费",
                content_type="policy",
                section_path="退货与退款政策 / 退货运费与邮费",
                source_doc="returns-policy.md",
            ),
        ],
        "knowledge",
    )

    bm25_hits = milvus_client.bm25_search(
        client, "ANC-06 降噪深度", 5, "knowledge"
    )
    assert bm25_hits[0][0] == 1

    hybrid_hits = milvus_client.hybrid_search(
        client, [0.1] * 1024, "ANC-06 降噪深度", 5, "knowledge"
    )
    assert hybrid_hits[0][0] == 1

    filtered = milvus_client.bm25_search(
        client,
        "退货运费",
        5,
        "knowledge",
        filter_expr='content_type == "policy"',
    )
    assert [hit[0] for hit in filtered] == [2]
    milvus_client.reset_client_cache()
