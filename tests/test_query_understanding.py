from app.knowledge.query_understanding import QueryRewrite, QueryRewriter
from app.knowledge.synonyms import expand_synonyms


def test_expand_synonyms_adds_group_terms_once():
    assert expand_synonyms("邮费怎么算") == "邮费怎么算 运费"


def test_expand_synonyms_does_not_duplicate_existing_terms():
    assert expand_synonyms("运费和邮费") == "运费和邮费"


def test_expand_synonyms_leaves_model_number_untouched():
    text = expand_synonyms("ANC-06 的降噪深度")
    assert "ANC-06" in text


class FakeRewriteModel:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return self.result


async def test_rewrite_returns_standard_query(tmp_path):
    prompt = tmp_path / "rewrite.txt"
    prompt.write_text("改写规则", encoding="utf-8")
    model = FakeRewriteModel(QueryRewrite(standard_query="蓝牙耳机怎么配对"))
    rewriter = QueryRewriter(model, prompt)

    assert await rewriter.rewrite("耳机咋连手机") == "蓝牙耳机怎么配对"
    assert "改写规则" in str(model.calls[0][0].content)


async def test_rewrite_falls_back_on_error(tmp_path):
    prompt = tmp_path / "rewrite.txt"
    prompt.write_text("规则", encoding="utf-8")
    rewriter = QueryRewriter(
        FakeRewriteModel(error=RuntimeError("boom")), prompt
    )

    assert await rewriter.rewrite("原问题") == "原问题"


async def test_rewrite_falls_back_on_empty_output(tmp_path):
    prompt = tmp_path / "rewrite.txt"
    prompt.write_text("规则", encoding="utf-8")
    rewriter = QueryRewriter(
        FakeRewriteModel(QueryRewrite(standard_query="  ")), prompt
    )

    assert await rewriter.rewrite("原问题") == "原问题"
