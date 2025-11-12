import cache_manager
from cache_manager import CacheManager


def test_cache_manager_set_get_material(tmp_path):
    db_path = tmp_path / "cache.db"
    manager = CacheManager(db_path=db_path, cache_ttl_seconds=3600)

    manager.set_material(
        material_name="Steel Beam",
        pt_name="Viga de aço",
        search_queries=["steel beam price"],
        key_specs=["length 6m"],
        best_supplier="Supplier A",
        price="100 EUR",
        url="https://example.com/steel",
        reasoning="Best price",
        suppliers=[{"supplier": "Supplier A", "price": "100 EUR"}],
    )

    cached = manager.get_material("Steel Beam")
    assert cached is not None
    assert cached["material_name"] == "Steel Beam"
    assert cached["pt_name"] == "Viga de aço"
    assert cached["search_queries"] == ["steel beam price"]
    assert cached["suppliers"] == [{"supplier": "Supplier A", "price": "100 EUR"}]


def test_cache_manager_clear_expired(tmp_path, monkeypatch):
    db_path = tmp_path / "cache.db"
    manager = CacheManager(db_path=db_path, cache_ttl_seconds=10)

    monkeypatch.setattr(cache_manager.time, "time", lambda: 100.0)
    manager.set_material(
        material_name="Concrete",
        pt_name="Concreto",
        search_queries=["concrete price"],
        key_specs=[],
        best_supplier="Supplier B",
        price="200 EUR",
        url="https://example.com/concrete",
        reasoning="Historical price",
        suppliers=[{"supplier": "Supplier B", "price": "200 EUR"}],
    )
    manager.set_search_results("concrete price", "{}")

    monkeypatch.setattr(cache_manager.time, "time", lambda: 200.0)
    removed = manager.clear_expired()
    assert removed == 2

    assert manager.get_material("Concrete") is None
    assert manager.get_search_results("concrete price") is None
