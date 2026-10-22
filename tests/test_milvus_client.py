import pytest

from app.knowledge import milvus_client


@pytest.fixture(autouse=True)
def clear_milvus_cache():
    milvus_client.reset_client_cache()
    yield
    milvus_client.reset_client_cache()


class FakeSchema:
    def __init__(self) -> None:
        self.fields: list[tuple[str, object, dict]] = []
        self.functions: list[dict] = []

    def add_field(self, name, dtype, **kwargs) -> None:
        self.fields.append((name, dtype, kwargs))

    def add_function(self, function) -> None:
        self.functions.append(
            {
                "name": function.name,
                "input_field_names": function.input_field_names,
                "output_field_names": function.output_field_names,
            }
        )


class FakeIndexParams:
    def __init__(self) -> None:
        self.indexes: list[dict] = []

    def add_index(self, **kwargs) -> None:
        self.indexes.append(kwargs)


class FakeClient:
    def __init__(self) -> None:
        self.collections: set[str] = set()
        self.descriptions: dict[str, list[dict]] = {}
        self.schema = FakeSchema()
        self.index_params = FakeIndexParams()
        self.auto_id: bool | None = None
        self.created: tuple | None = None
        self.loaded: str | None = None
        self.upserted: tuple | None = None
        self.search_kwargs: dict | None = None
        self.search_results: list[dict] = []
        self.deleted: tuple | None = None
        self.dropped: list[str] = []

    def has_collection(self, name: str) -> bool:
        return name in self.collections

    def create_schema(self, auto_id: bool = True):
        self.auto_id = auto_id
        return self.schema

    def prepare_index_params(self):
        return self.index_params

    def create_collection(
        self, collection_name, schema, index_params
    ) -> None:
        self.created = (collection_name, schema, index_params)
        self.collections.add(collection_name)

    def describe_collection(self, name: str) -> dict:
        return {"fields": self.descriptions.get(name, [])}

    def drop_collection(self, name: str) -> None:
        self.dropped.append(name)
        self.collections.discard(name)

    def load_collection(self, name: str) -> None:
        self.loaded = name

    def upsert(self, collection_name, data) -> None:
        self.upserted = (collection_name, data)

    def search(self, **kwargs):
        self.search_kwargs = kwargs
        return [self.search_results]

    def delete(self, collection_name, ids):
        self.deleted = (collection_name, ids)
        return {"delete_count": len(ids)}

    def get(self, collection_name, ids, output_fields=None):
        return [{"id": value} for value in ids]


def test_ensure_collection_creates_bm25_schema():
    client = FakeClient()

    milvus_client.ensure_collection(
        client, "knowledge", dim=1024, analyzer_type="jieba"
    )

    fields = {field[0]: field for field in client.schema.fields}
    assert list(fields) == [
        "id",
        "text",
        "sparse",
        "dense",
        "category",
        "content_type",
        "section_path",
        "source_doc",
    ]
    assert client.auto_id is False
    assert fields["text"][2]["enable_analyzer"] is True
    assert fields["text"][2]["analyzer_params"] == {"type": "jieba"}
    assert client.schema.functions[0]["name"] == "text_bm25"
    assert client.schema.functions[0]["input_field_names"] == ["text"]
    assert client.schema.functions[0]["output_field_names"] == ["sparse"]
    assert client.index_params.indexes == [
        {
            "field_name": "dense",
            "index_type": "AUTOINDEX",
            "metric_type": "COSINE",
        },
        {
            "field_name": "sparse",
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "BM25",
        },
    ]
    assert client.created[0] == "knowledge"
    assert client.loaded == "knowledge"


def test_ensure_collection_is_idempotent():
    client = FakeClient()
    client.collections.add("knowledge")
    client.descriptions["knowledge"] = [
        {"name": name}
        for name in (
            "id",
            "text",
            "sparse",
            "dense",
            "category",
            "content_type",
            "section_path",
            "source_doc",
        )
    ]

    milvus_client.ensure_collection(client, "knowledge")

    assert client.created is None
    assert client.loaded == "knowledge"


def test_ensure_collection_rejects_old_schema():
    client = FakeClient()
    client.collections.add("knowledge")
    client.descriptions["knowledge"] = [{"name": "id"}, {"name": "dense"}]

    with pytest.raises(RuntimeError, match="--recreate"):
        milvus_client.ensure_collection(client, "knowledge")


def test_recreate_collection_drops_and_creates():
    client = FakeClient()
    client.collections.add("knowledge")

    milvus_client.recreate_collection(client, "knowledge")

    assert client.dropped == ["knowledge"]
    assert client.created[0] == "knowledge"


def test_upsert_vectors_maps_text_and_metadata():
    client = FakeClient()
    record = milvus_client.VectorRecord(
        id=1,
        text="分类\n问题\n答案",
        vector=[0.1, 0.2],
        category="分类",
        content_type="policy",
        section_path="分类 / 问题",
        source_doc="docs.md",
    )

    milvus_client.upsert_vectors(client, [record], "knowledge")

    assert client.upserted == (
        "knowledge",
        [
            {
                "id": 1,
                "text": "分类\n问题\n答案",
                "dense": [0.1, 0.2],
                "category": "分类",
                "content_type": "policy",
                "section_path": "分类 / 问题",
                "source_doc": "docs.md",
            }
        ],
    )


def test_dense_search_returns_id_score_pairs():
    client = FakeClient()
    client.search_results = [
        {"id": 2, "distance": 0.91},
        {"id": 1, "distance": 0.42},
    ]

    hits = milvus_client.dense_search(client, [0.1, 0.2], 5, "knowledge")

    assert hits == [(2, 0.91), (1, 0.42)]
    assert client.search_kwargs["collection_name"] == "knowledge"
    assert client.search_kwargs["limit"] == 5
    assert client.search_kwargs["output_fields"] == ["id"]
    assert client.search_kwargs["search_params"] == {"metric_type": "COSINE"}


def test_delete_vectors_and_has_ids():
    client = FakeClient()

    assert milvus_client.delete_vectors(client, [1, 2], "knowledge") == 2
    assert client.deleted == ("knowledge", [1, 2])
    assert milvus_client.has_ids(client, [3, 4], "knowledge") == {3, 4}
