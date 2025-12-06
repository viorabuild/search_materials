import builtins

import pytest

from unified_agent import ConstructionAIAgent


def _make_agent() -> ConstructionAIAgent:
    """Create a minimal agent instance without running __init__."""
    agent = ConstructionAIAgent.__new__(ConstructionAIAgent)  # type: ignore
    return agent


def test_normalize_payload_id_text():
    agent = _make_agent()
    payload = {"id": "42", "text": "Перевод строки"}
    assert agent._normalize_translation_payload(payload) == {"42": "Перевод строки"}


def test_normalize_payload_translations_map():
    agent = _make_agent()
    payload = {"translations": {"1": "one", 2: "two"}}
    assert agent._normalize_translation_payload(payload) == {"1": "one", "2": "two"}


def test_normalize_payload_list_items():
    agent = _make_agent()
    payload = [
        {"id": "1", "translation": "uno"},
        {"id": 2, "name_en": "dos"},
        {"key": "3", "value": "tres"},
    ]
    assert agent._normalize_translation_payload(payload) == {"1": "uno", "2": "dos", "3": "tres"}


def test_build_format2_rows_injects_translations():
    agent = _make_agent()
    items = [
        {"translation_id": "1", "number": "1", "description": "desc1", "unit": "m2", "quantity": 1, "unit_price": 2},
        {"translation_id": "2", "number": "1.1", "description": "desc2", "unit": "m", "quantity": 3, "unit_price": 4},
    ]
    translations = {"1": "tr1", "2": "tr2"}

    rows, section_rows = agent._build_format2_rows(items, translations)

    # header + 2 data rows
    assert len(rows) == 3
    assert len(section_rows) == 0
    assert rows[1][7] == "tr1"
    assert rows[2][7] == "tr2"
