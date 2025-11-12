import types
from pathlib import Path

import pytest

from cache_manager import CacheManager
from material_price_agent import BestOffer, MaterialQueryAnalysis, MaterialResult, SupplierQuote
from unified_agent import ConstructionAIAgent


class FakeMaterialAgent:
    def __init__(self):
        self.analysis_calls = 0
        self.resolve_calls = 0
        self.next_result_factory = None

    def analyze_material(self, material_name: str) -> MaterialQueryAnalysis:
        self.analysis_calls += 1
        return MaterialQueryAnalysis(
            pt_name=f"{material_name} PT",
            search_queries=[f"search {material_name}"],
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
        deadline=None,
    ) -> MaterialResult:
        self.resolve_calls += 1
        if self.next_result_factory is None:
            return MaterialResult(
                material_name=material_name,
                analysis=analysis,
                quotes=[
                    SupplierQuote(
                        supplier="Default Supplier",
                        price="$10",
                        url="https://example.com",
                        notes=None,
                    )
                ],
                best_offer=BestOffer(
                    best_supplier="Default Supplier",
                    price="$10",
                    url="https://example.com",
                    reasoning="Автоматический выбор",
                ),
            )
        return self.next_result_factory(material_name, analysis)


def build_agent(tmp_path: Path):
    agent = ConstructionAIAgent.__new__(ConstructionAIAgent)
    fake_agent = FakeMaterialAgent()
    agent.material_agent = fake_agent
    agent.advanced_agent = None
    agent.default_timeout_seconds = 30
    agent.local_materials_db = None
    agent.materials_db_assistant = None

    config = types.SimpleNamespace(
        enable_known_sites=True,
        known_sites_only=False,
        enable_real_scraping=False,
        enable_gpt_scraping=False,
        request_delay_seconds=0.0,
        failure_cache_ttl_seconds=120,
        timeout_cache_ttl_seconds=60,
        exact_match_only_default=False,
    )
    agent.config = config

    cache_path = tmp_path / "cache.db"
    agent.cache = CacheManager(
        db_path=cache_path,
        cache_ttl_seconds=3600,
        negative_cache_ttl_seconds=config.failure_cache_ttl_seconds,
    )

    return agent, fake_agent


def test_negative_cache_prevents_repeated_search(tmp_path):
    agent, fake_agent = build_agent(tmp_path)

    def failure_factory(name: str, analysis: MaterialQueryAnalysis) -> MaterialResult:
        return MaterialResult(
            material_name=name,
            analysis=analysis,
            quotes=[],
            best_offer=BestOffer(
                best_supplier="N/A",
                price="N/A",
                url="N/A",
                reasoning="Нет доступных предложений",
            ),
        )

    fake_agent.next_result_factory = failure_factory

    first_result = agent.find_material_price("Test Material", use_cache=True)
    assert first_result.best_offer.best_supplier == "N/A"
    assert fake_agent.analysis_calls == 1
    assert fake_agent.resolve_calls == 1

    second_result = agent.find_material_price("Test Material", use_cache=True)
    assert second_result.best_offer.best_supplier == "N/A"
    assert second_result.best_offer.reasoning == "Нет доступных предложений"
    assert fake_agent.analysis_calls == 1
    assert fake_agent.resolve_calls == 1

    cached_failure = agent.cache.get_failed_material("Test Material")
    assert cached_failure is not None


def test_success_result_overrides_failure_cache(tmp_path):
    agent, fake_agent = build_agent(tmp_path)

    failure_result = MaterialResult(
        material_name="Concrete",
        analysis=MaterialQueryAnalysis(
            pt_name="Concrete PT",
            search_queries=["concrete"],
            key_specs=["M300"],
        ),
        quotes=[],
        best_offer=BestOffer(
            best_supplier="N/A",
            price="N/A",
            url="N/A",
            reasoning="Нет данных",
        ),
    )

    agent._cache_material_result("Concrete", failure_result)
    assert agent.cache.get_failed_material("Concrete") is not None

    success_result = MaterialResult(
        material_name="Concrete",
        analysis=failure_result.analysis,
        quotes=[
            SupplierQuote(
                supplier="Supplier X",
                price="$100",
                url="https://supplier.example.com",
                notes=None,
            )
        ],
        best_offer=BestOffer(
            best_supplier="Supplier X",
            price="$100",
            url="https://supplier.example.com",
            reasoning="Лучшее предложение",
        ),
    )

    agent._cache_material_result("Concrete", success_result)

    assert agent.cache.get_failed_material("Concrete") is None
    cached_success = agent.cache.get_material("Concrete")
    assert cached_success is not None
    assert cached_success["best_supplier"] == "Supplier X"
