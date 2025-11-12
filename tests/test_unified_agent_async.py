import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from material_price_agent import (
    BestOffer,
    MaterialQueryAnalysis,
    MaterialResult,
    SearchTimeoutError,
    SupplierQuote,
)
from unified_agent import ConstructionAIAgent


class DummyCache:
    def __init__(self):
        self.storage = {}

    def get_material(self, material_name: str):
        return None

    def set_material(self, **kwargs):
        self.storage[kwargs["material_name"]] = kwargs


class MockMaterialAgent:
    def __init__(self, analyze_delay: float, resolve_delay: float, supplier: str):
        self.analyze_delay = analyze_delay
        self.resolve_delay = resolve_delay
        self.supplier = supplier
        self.cancelled = False

    def analyze_material(self, material_name: str) -> MaterialQueryAnalysis:
        event = getattr(self, "_search_cancel_event", None)
        start = time.perf_counter()
        while time.perf_counter() - start < self.analyze_delay:
            if event and event.is_set():
                self.cancelled = True
                raise SearchTimeoutError("Analysis cancelled")
            time.sleep(0.01)
        return MaterialQueryAnalysis(
            pt_name=f"{material_name}-pt",
            search_queries=[f"query-{material_name}"],
            key_specs=["spec"],
        )

    def _resolve_material(
        self,
        material_name: str,
        analysis: MaterialQueryAnalysis,
        use_scraping: bool,
        enable_known_sites: bool,
        known_sites_only: bool,
        *,
        use_real_scraping: bool,
        use_gpt_scraping: bool,
        deadline,
    ) -> MaterialResult:
        event = getattr(self, "_search_cancel_event", None)
        start = time.perf_counter()
        while time.perf_counter() - start < self.resolve_delay:
            if event and event.is_set():
                self.cancelled = True
                raise SearchTimeoutError("Resolution cancelled")
            time.sleep(0.01)

        self.cancelled = bool(event and event.is_set())
        quote = SupplierQuote(
            supplier=self.supplier,
            price="10",
            url="https://example.com",
            notes="material",
        )
        best_offer = BestOffer(
            best_supplier=self.supplier,
            price="10",
            url="https://example.com",
            reasoning="fast path",
        )
        return MaterialResult(
            material_name=material_name,
            analysis=analysis,
            quotes=[quote],
            best_offer=best_offer,
        )


class MockAdvancedAgent:
    def __init__(self, delay: float, supplier: str):
        self.delay = delay
        self.supplier = supplier
        self.cancelled = False

    def find_best_price(self, material_name: str):
        event = getattr(self, "_search_cancel_event", None)
        start = time.perf_counter()
        while time.perf_counter() - start < self.delay:
            if event and event.is_set():
                self.cancelled = True
                raise SearchTimeoutError("Advanced search cancelled")
            time.sleep(0.01)

        self.cancelled = bool(event and event.is_set())
        return SimpleNamespace(
            material_name=material_name,
            pt_name=f"{material_name}-pt",
            suppliers=[
                {
                    "supplier": self.supplier,
                    "price": "15",
                    "url": "https://advanced.example.com",
                    "notes": "advanced",
                }
            ],
            best_supplier=self.supplier,
            best_price="15",
            best_url="https://advanced.example.com",
            reasoning="slow path",
        )


def build_agent(material_agent, advanced_agent=None) -> ConstructionAIAgent:
    agent = object.__new__(ConstructionAIAgent)
    agent.material_agent = material_agent
    agent.advanced_agent = advanced_agent
    agent.cache = DummyCache()
    agent.local_materials_db = None
    agent.config = SimpleNamespace(
        exact_match_only_default=False,
        enable_known_sites=True,
        known_sites_only=False,
        enable_real_scraping=False,
        enable_gpt_scraping=False,
    )
    agent.default_timeout_seconds = 5.0
    return agent


def test_fast_material_path_finishes_before_slow_advanced():
    material_agent = MockMaterialAgent(analyze_delay=0.05, resolve_delay=0.05, supplier="Local")
    advanced_agent = MockAdvancedAgent(delay=0.3, supplier="Advanced")
    agent = build_agent(material_agent, advanced_agent)

    start = time.perf_counter()
    result = agent.find_material_price(
        "brick",
        use_cache=False,
        use_scraping=False,
        use_advanced_search=True,
    )
    elapsed = time.perf_counter() - start

    assert result.best_offer.best_supplier == "Local"
    assert elapsed < 0.25
    assert advanced_agent.cancelled is True


def test_timeout_cancels_both_paths():
    material_agent = MockMaterialAgent(analyze_delay=0.3, resolve_delay=0.3, supplier="Local")
    advanced_agent = MockAdvancedAgent(delay=0.4, supplier="Advanced")
    agent = build_agent(material_agent, advanced_agent)

    start = time.perf_counter()
    result = agent.find_material_price(
        "steel",
        use_cache=False,
        use_scraping=False,
        use_advanced_search=True,
        timeout_seconds=0.2,
    )
    elapsed = time.perf_counter() - start

    assert result.best_offer.best_supplier == "N/A"
    assert "⏱" in result.best_offer.reasoning
    assert elapsed == pytest.approx(0.2, abs=0.15)
    assert material_agent.cancelled is True
    assert advanced_agent.cancelled is True
