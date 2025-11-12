"""Tests for automatic cache cleanup in ConstructionAIAgent."""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unified_agent import ConstructionAIAgent, ConstructionAIAgentConfig


def test_cache_cleanup_thread_removes_expired_entries(tmp_path) -> None:
    """Ensure expired cache entries are removed automatically."""

    db_path = tmp_path / "cache.db"
    config = ConstructionAIAgentConfig(
        openai_api_key="test-key",
        cache_db_path=str(db_path),
        cache_ttl_seconds=1,
        cache_cleanup_interval=0.2,
        enable_local_db=False,
        enable_materials_db_assistant=False,
        enable_project_chat=False,
    )

    agent = ConstructionAIAgent(config)
    try:
        agent.cache.set_material(
            "Test Material",
            "PT",
            ["query"],
            ["spec"],
            "Supplier",
            "$10",
            "http://example.com",
            "reason",
            [],
        )

        with sqlite3.connect(db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM material_cache").fetchone()[0]
        assert count == 1

        time.sleep(config.cache_ttl_seconds + config.cache_cleanup_interval + 0.5)

        with sqlite3.connect(db_path) as conn:
            remaining = conn.execute("SELECT COUNT(*) FROM material_cache").fetchone()[0]

        assert remaining == 0
    finally:
        agent.close()
