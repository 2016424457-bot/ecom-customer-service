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

    def add_field(self, name, dtype, **kwargs) -> None:
        self.fields.append((name, dtype, kwargs))


class FakeIndexParams:
    def __init__(self) -> None:
        self.indexes: list[dict] = []

    def add_index(self, **kwargs) -> None:
        self.indexes.append(kwargs)


class FakeClient:
    def __init__(self) -> None:
        self.collections: set[str] = set()
        self.schema = FakeSchema()
        self.index_params = FakeIndexParams()
        self.auto_id: bool | None = None
        self.created: tuple | None = None
        self.loaded: str | None = None
        self.upserted: tuple | None = None
        self.search_kwargs: dict | None = None
        self.search_results: list[dict] = []
        self.deleted: tuple | None = None

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


def test_ensure_collection_creates_int64_id_and_dense_vector():
    client = FakeClient()

    milvus_client.ensure_collection(client, "knowledge", dim=1024)

    assert client.auto_id is False
    assert [field[0] for field in client.schema.fields] == ["id", "dense"]
    assert client.schema.fields[0][2]["is_primary"] is True
    assert client.schema.fields[1][2]["dim"] == 1024
    assert client.index_params.indexes == [
        {
            "field_name": "dense",
            "index_type": "AUTOINDEX",
            "metric_type": "COSINE",
        }
    ]
    assert client.created[0] == "knowledge"
    assert client.loaded == "knowledge"


def test_ensure_collection_is_idempotent():
    client = FakeClient()
    client.collections.add("knowledge")

    milvus_client.ensure_collection(client, "knowledge")

    assert client.created is None
    assert client.loaded == "knowledge"


def test_upsert_vectors_maps_ids_and_dense_field():
    client = FakeClient()

    milvus_client.upsert_vectors(
        client, [1, 2], [[0.1, 0.2], [0.3, 0.4]], "knowledge"
    )

    assert client.upserted == (
        "knowledge",
        [
            {"id": 1, "dense": [0.1, 0.2]},
            {"id": 2, "dense": [0.3, 0.4]},
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
