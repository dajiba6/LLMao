from duckduckgo_search import DDGS


def duckducknews(query: str) -> list:
    with DDGS() as ddgs:
        return list(ddgs.news(keywords=query, region="cn-zh", max_results=5))


def duckducktext(query: str) -> list:
    with DDGS() as ddgs:
        return list(ddgs.text(keywords=query, region="cn-zh", max_results=5))
