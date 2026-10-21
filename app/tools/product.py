import random

from langchain_core.tools import tool


@tool(parse_docstring=True)
def query_product(product_name: str) -> dict:
    """查询商品价格、库存和规格。用户询问某个商品多少钱、有没有货时调用。

    Args:
        product_name: 商品名称
    """
    stock = random.randint(0, 200)
    return {
        "product_name": product_name,
        "price": round(random.uniform(29, 999), 2),
        "stock": stock,
        "specs": random.choice(["标准版", "升级版", "礼盒装"]),
        "available": stock > 0,
    }
