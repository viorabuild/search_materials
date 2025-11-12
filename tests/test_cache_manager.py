from __future__ import annotations

import json
from pathlib import Path

from cache_manager import CacheManager


def test_search_results_cache_survives_process_restart(tmp_path: Path) -> None:
    db_path = tmp_path / "materials.db"
    cache = CacheManager(db_path=db_path, cache_ttl_seconds=3600)

    payload = json.dumps({"suppliers": ["A", "B"]})
    cache.set_search_results("cement", payload)

    # Re-create manager to mimic process restart and ensure deterministic hashing
    new_cache = CacheManager(db_path=db_path, cache_ttl_seconds=3600)
    assert new_cache.get_search_results("cement") == payload


def test_search_results_cache_distinguishes_queries(tmp_path: Path) -> None:
    db_path = tmp_path / "materials.db"
    cache = CacheManager(db_path=db_path, cache_ttl_seconds=3600)

    cache.set_search_results("cement", "one")
    cache.set_search_results("cement 42", "two")

    assert cache.get_search_results("cement") == "one"
    assert cache.get_search_results("cement 42") == "two"
