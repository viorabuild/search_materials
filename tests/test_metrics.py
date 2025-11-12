import sys
from types import SimpleNamespace
from typing import Any, Dict

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unified_agent import (  # noqa: E402
    ConstructionAIAgent,
    ConstructionAIAgentConfig,
    MATERIAL_CACHE_HITS,
    MATERIAL_CACHE_MISSES,
    MATERIAL_SEARCH_DURATION,
    MATERIAL_SEARCH_REQUESTS,
    SHEETS_COMMAND_DURATION,
    SHEETS_COMMAND_REQUESTS,
    ESTIMATE_CHECK_DURATION,
    ESTIMATE_CHECK_REQUESTS,
    LLM_REQUESTS,
    MaterialQueryAnalysis,
    MaterialResult,
    BestOffer,
    SupplierQuote,
    _instrument_openai_client,
)


class DummyCache:
    def __init__(self, contents: Dict[str, Dict[str, Any]] | None = None) -> None:
        self.contents = contents or {}
        self.set_calls = 0

    def get_material(self, name: str) -> Dict[str, Any] | None:
        return self.contents.get(name)

    def set_material(self, **payload: Any) -> None:
        self.set_calls += 1
        self.contents[payload["material_name"]] = payload


class DummyMaterialAgent:
    def analyze_material(self, material_name: str) -> MaterialQueryAnalysis:
        return MaterialQueryAnalysis(pt_name=material_name, search_queries=[], key_specs=[])

    def _resolve_material(self, material_name: str, analysis: MaterialQueryAnalysis, *_, **__) -> MaterialResult:
        quote = SupplierQuote(supplier="Stub Supplier", price="10", url="http://example.com", notes="stub")
        return MaterialResult(
            material_name=material_name,
            analysis=analysis,
            quotes=[quote],
            best_offer=BestOffer(
                best_supplier=quote.supplier,
                price=quote.price,
                url=quote.url,
                reasoning="stub",
            ),
        )


def _build_agent() -> ConstructionAIAgent:
    agent = ConstructionAIAgent.__new__(ConstructionAIAgent)
    agent.config = ConstructionAIAgentConfig(openai_api_key="test")
    agent.default_timeout_seconds = 5.0
    agent.local_materials_db = None
    agent.advanced_agent = None
    agent.material_agent = DummyMaterialAgent()
    agent.cache = DummyCache()
    return agent


def test_material_search_metrics_cache_hit() -> None:
    agent = _build_agent()
    cached_payload = {
        "pt_name": "Steel Beam",
        "search_queries": ["steel beam"],
        "key_specs": ["H-Beam"],
        "suppliers": [],
        "best_supplier": "Cached Supplier",
        "price": "123",
        "url": "http://cached.example.com",
        "reasoning": "cached",
    }
    agent.cache = DummyCache({"Steel Beam": cached_payload})

    search_before = MATERIAL_SEARCH_REQUESTS._value.get()
    hits_before = MATERIAL_CACHE_HITS._value.get()
    misses_before = MATERIAL_CACHE_MISSES._value.get()
    duration_before = MATERIAL_SEARCH_DURATION._count.get()

    result = agent.find_material_price("Steel Beam")

    assert result.best_offer.best_supplier == "Cached Supplier"
    assert MATERIAL_SEARCH_REQUESTS._value.get() == search_before + 1
    assert MATERIAL_CACHE_HITS._value.get() == hits_before + 1
    assert MATERIAL_CACHE_MISSES._value.get() == misses_before
    assert MATERIAL_SEARCH_DURATION._count.get() == duration_before + 1


def test_material_search_metrics_cache_miss() -> None:
    agent = _build_agent()

    search_before = MATERIAL_SEARCH_REQUESTS._value.get()
    hits_before = MATERIAL_CACHE_HITS._value.get()
    misses_before = MATERIAL_CACHE_MISSES._value.get()
    duration_before = MATERIAL_SEARCH_DURATION._count.get()

    result = agent.find_material_price("Concrete")

    assert result.material_name == "Concrete"
    assert MATERIAL_SEARCH_REQUESTS._value.get() == search_before + 1
    assert MATERIAL_CACHE_HITS._value.get() == hits_before
    assert MATERIAL_CACHE_MISSES._value.get() == misses_before + 1
    assert MATERIAL_SEARCH_DURATION._count.get() == duration_before + 1


def test_sheets_command_metrics() -> None:
    agent = _build_agent()
    agent.sheets_ai = SimpleNamespace(process_command=lambda command: f"processed: {command}")

    requests_before = SHEETS_COMMAND_REQUESTS._value.get()
    duration_before = SHEETS_COMMAND_DURATION._count.get()

    response = agent.process_sheets_command("list materials")

    assert response == "processed: list materials"
    assert SHEETS_COMMAND_REQUESTS._value.get() == requests_before + 1
    assert SHEETS_COMMAND_DURATION._count.get() == duration_before + 1


def test_estimate_check_metrics() -> None:
    class DummyWorksheet:
        def __init__(self, data: list[list[str]]):
            self._data = data

        def get_all_values(self) -> list[list[str]]:
            return self._data

    class DummyEstimateChecker:
        def validate_estimate(self, estimate_data, master_data, quantity_col):
            return {
                "estimate": estimate_data,
                "master": master_data,
                "quantity_col": quantity_col,
            }

        def format_validation_report(self, payload):
            return f"OK ({payload['quantity_col']})"

    class DummySpreadsheet:
        def __init__(self):
            self._worksheets = {
                "Sheet1": DummyWorksheet([["header"]]),
                "Master List": DummyWorksheet([["master"]]),
            }

        def worksheet(self, name: str):
            return self._worksheets[name]

    agent = _build_agent()
    agent.sheets_ai = SimpleNamespace(
        estimate_checker=DummyEstimateChecker(),
        spreadsheet=DummySpreadsheet(),
    )

    requests_before = ESTIMATE_CHECK_REQUESTS._value.get()
    duration_before = ESTIMATE_CHECK_DURATION._count.get()

    report = agent.check_estimate(quantity_col="G")

    assert report == "OK (G)"
    assert ESTIMATE_CHECK_REQUESTS._value.get() == requests_before + 1
    assert ESTIMATE_CHECK_DURATION._count.get() == duration_before + 1


def test_instrument_openai_client_counts_requests() -> None:
    class DummyFallback:
        def __init__(self) -> None:
            self.called = 0

        def _call_with_fallback(self, caller, payload):
            self.called += 1
            return caller("primary", payload)

    dummy = DummyFallback()

    baseline = LLM_REQUESTS._value.get()
    _instrument_openai_client(dummy)  # type: ignore[arg-type]

    result = dummy._call_with_fallback(lambda *_: "ok", {})

    assert result == "ok"
    assert dummy.called == 1
    assert LLM_REQUESTS._value.get() == baseline + 1
