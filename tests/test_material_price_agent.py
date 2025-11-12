import time
import types

import pytest

from material_price_agent import (
    MaterialPriceAgent,
    MaterialQueryAnalysis,
    SupplierQuote,
    SearchTimeoutError,
)


class DummyOpenAIClient:
    def __init__(self):
        completions = types.SimpleNamespace(create=lambda **_: None)
        self.chat = types.SimpleNamespace(completions=completions)


@pytest.fixture
def base_agent():
    return MaterialPriceAgent(openai_client=DummyOpenAIClient())


def test_collect_quotes_primary_strategy(monkeypatch, base_agent):
    analysis = MaterialQueryAnalysis(
        pt_name="cimento",
        search_queries=["primary query"],
        key_specs=["spec"],
    )
    expected = [
        SupplierQuote(supplier="Supplier", price="10 EUR", url="https://example.com"),
    ]
    calls = []

    def fake_fetch(self, queries, use_scraping, use_real_scraping, use_gpt_scraping, deadline):
        calls.append(list(queries))
        if list(queries) == ["primary query"]:
            return expected
        return []

    monkeypatch.setattr(
        base_agent,
        "_fetch_quotes",
        types.MethodType(fake_fetch, base_agent),
    )

    result = base_agent._collect_quotes(
        "Cement",
        analysis,
        use_scraping=False,
        enable_known_sites=False,
        known_sites_only=False,
    )

    assert result == expected
    assert calls == [["primary query"]]


def test_collect_quotes_fallback_to_portuguese(monkeypatch, base_agent):
    analysis = MaterialQueryAnalysis(
        pt_name="cimento",
        search_queries=[""],
        key_specs=[],
    )
    expected = [
        SupplierQuote(supplier="PT Supplier", price="15 EUR", url="https://pt.example.com"),
    ]
    calls = []

    def fake_fetch(self, queries, use_scraping, use_real_scraping, use_gpt_scraping, deadline):
        calls.append(list(queries))
        if any("Portugal" in query for query in queries):
            return expected
        return []

    monkeypatch.setattr(
        base_agent,
        "_fetch_quotes",
        types.MethodType(fake_fetch, base_agent),
    )

    result = base_agent._collect_quotes(
        "Cement",
        analysis,
        use_scraping=False,
        enable_known_sites=False,
        known_sites_only=False,
    )

    assert result == expected
    assert any("Portugal" in q for q in calls[-1])


def test_collect_quotes_known_site_fallback(monkeypatch, base_agent):
    analysis = MaterialQueryAnalysis(
        pt_name="cimento",
        search_queries=["primary"],
        key_specs=[],
    )
    expected = [
        SupplierQuote(supplier="Known", price="20 EUR", url="https://known.example.com"),
    ]

    def fake_fetch(self, *_, **__):
        return []

    monkeypatch.setattr(
        base_agent,
        "_fetch_quotes",
        types.MethodType(fake_fetch, base_agent),
    )

    monkeypatch.setattr(
        base_agent,
        "_search_known_site_quotes",
        types.MethodType(lambda self, *args, **kwargs: expected, base_agent),
    )

    result = base_agent._collect_quotes(
        "Cement",
        analysis,
        use_scraping=False,
        enable_known_sites=True,
        known_sites_only=False,
    )

    assert result == expected


def test_collect_quotes_timeout(monkeypatch, base_agent):
    analysis = MaterialQueryAnalysis(
        pt_name="cimento",
        search_queries=["primary"],
        key_specs=[],
    )

    with pytest.raises(SearchTimeoutError):
        base_agent._collect_quotes(
            "Cement",
            analysis,
            use_scraping=False,
            enable_known_sites=False,
            known_sites_only=False,
            deadline=time.time() - 1,
        )
