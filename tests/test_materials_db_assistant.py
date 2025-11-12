"""Tests for materials_db_assistant pending state persistence."""

from __future__ import annotations

import json
from types import SimpleNamespace

from materials_db_assistant import MaterialsDatabaseAssistant


class _DummyLLM:
    def __init__(self) -> None:
        message = SimpleNamespace(content="{}")
        choice = SimpleNamespace(message=message)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_: SimpleNamespace(choices=[choice]))
        )


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("ID,Название RU ,Стоимость\n")
        for row in rows:
            handle.write(",".join(row) + "\n")


def test_pending_changes_survive_restart(tmp_path):
    csv_path = tmp_path / "materials.csv"
    cache_path = tmp_path / "cache" / "materials_pending.json"
    _write_csv(csv_path, [])

    assistant = MaterialsDatabaseAssistant(
        csv_path=csv_path,
        llm_client=_DummyLLM(),
        llm_model="dummy",
        sync_csv=False,
        pending_cache_path=cache_path,
        pending_expiration_seconds=None,
    )

    assistant._stage_insert(
        {"ID": "MAT-001", "Название RU ": "Материал", "Стоимость": "100"},
        reason=None,
        worksheet_hint=None,
    )

    assert assistant.pending_count() == 1
    assert cache_path.exists()

    assistant_reloaded = MaterialsDatabaseAssistant(
        csv_path=csv_path,
        llm_client=_DummyLLM(),
        llm_model="dummy",
        sync_csv=False,
        pending_cache_path=cache_path,
        pending_expiration_seconds=None,
    )

    assert assistant_reloaded.pending_count() == 1
    change_ids = sorted(assistant_reloaded._pending.keys())
    assert change_ids == ["CHG-0001"]

    assistant_reloaded._stage_insert(
        {"ID": "MAT-002", "Название RU ": "Материал 2", "Стоимость": "150"},
        reason=None,
        worksheet_hint=None,
    )

    reloaded_ids = sorted(assistant_reloaded._pending.keys())
    assert reloaded_ids == ["CHG-0001", "CHG-0002"]

    with cache_path.open("r", encoding="utf-8") as handle:
        persisted = json.load(handle)

    assert persisted["counter"] >= 2
    stored_ids = [item["change_id"] for item in persisted["changes"]]
    assert stored_ids == ["CHG-0001", "CHG-0002"]


def test_pending_cleanup_drops_missing_records(tmp_path):
    csv_path = tmp_path / "materials.csv"
    cache_path = tmp_path / "cache" / "materials_pending.json"
    _write_csv(csv_path, [["MAT-100", "Материал", "50"]])

    assistant = MaterialsDatabaseAssistant(
        csv_path=csv_path,
        llm_client=_DummyLLM(),
        llm_model="dummy",
        sync_csv=False,
        pending_cache_path=cache_path,
        pending_expiration_seconds=None,
    )

    assistant._stage_update(
        "MAT-100",
        {"Стоимость": "55"},
        reason=None,
        worksheet_hint=None,
    )

    assert assistant.pending_count() == 1

    # Remove the underlying record to emulate an outdated staging entry.
    _write_csv(csv_path, [])

    assistant_reloaded = MaterialsDatabaseAssistant(
        csv_path=csv_path,
        llm_client=_DummyLLM(),
        llm_model="dummy",
        sync_csv=False,
        pending_cache_path=cache_path,
        pending_expiration_seconds=None,
    )

    assert assistant_reloaded.pending_count() == 0

    with cache_path.open("r", encoding="utf-8") as handle:
        persisted = json.load(handle)

    assert persisted["changes"] == []
