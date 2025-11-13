from __future__ import annotations

import json
import sys
import types
from threading import Lock
from types import MethodType, SimpleNamespace

import pytest


def _install_gspread_stub() -> None:
    existing = sys.modules.get("gspread")
    if existing is not None and not getattr(existing, "__is_stub__", False):
        return

    if existing is not None and getattr(existing, "__is_stub__", False):
        sys.modules.pop("gspread", None)
        try:
            import gspread  # noqa: F401
            return
        except Exception:
            pass

    try:
        import gspread  # noqa: F401
        return
    except Exception:
        module = types.ModuleType("gspread")

        class _WorksheetNotFound(Exception):
            pass

        def _service_account_stub(*_: object, **__: object) -> None:
            return None

        module.Client = object  # type: ignore[attr-defined]
        module.service_account = _service_account_stub
        module.service_account_from_dict = _service_account_stub
        module.exceptions = types.SimpleNamespace(WorksheetNotFound=_WorksheetNotFound)
        module.__is_stub__ = True  # type: ignore[attr-defined]

        sys.modules["gspread"] = module


_install_gspread_stub()


def _install_dotenv_stub() -> None:
    existing = sys.modules.get("dotenv")
    if existing is not None and not getattr(existing, "__is_stub__", False):
        return

    if existing is not None and getattr(existing, "__is_stub__", False):
        sys.modules.pop("dotenv", None)
        try:
            import dotenv  # noqa: F401
            return
        except Exception:
            pass

    try:
        import dotenv  # noqa: F401
        return
    except Exception:
        module = types.ModuleType("dotenv")

        def _noop(*_: object, **__: object) -> None:
            return None

        module.load_dotenv = _noop
        module.__is_stub__ = True  # type: ignore[attr-defined]
        sys.modules["dotenv"] = module


_install_dotenv_stub()

from unified_agent import ConstructionAIAgent, ConstructionAIAgentConfig


class _StubOpenAIClient:
    """Deterministic stub for openai_client.chat.completions.create."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        completions = SimpleNamespace(create=self._create)
        self.chat = SimpleNamespace(completions=completions)

    def _create(self, **_: object) -> SimpleNamespace:
        if not self._responses:
            raise AssertionError("No stub responses left")
        content = self._responses.pop(0)
        message = SimpleNamespace(content=content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


def _build_agent(responses: list[str]) -> ConstructionAIAgent:
    agent = ConstructionAIAgent.__new__(ConstructionAIAgent)
    agent.config = ConstructionAIAgentConfig(openai_api_key="test-key")
    agent.openai_client = _StubOpenAIClient(responses)
    agent.material_agent = None
    agent.sheets_ai = None
    agent.cache = None
    agent.advanced_agent = None
    agent.local_materials_db = None
    agent.materials_db_assistant = None
    agent.project_chat_agent = None
    agent.default_timeout_seconds = agent.config.default_timeout_seconds
    agent._batch_rate_lock = Lock()
    agent._last_batch_request_ts = 0.0
    return agent


def test_process_command_routes_material_search() -> None:
    payload = json.dumps(
        {
            "command_type": "MATERIAL_SEARCH",
            "parameters": {"material_name": "Bricks"},
        }
    )
    agent = _build_agent([payload])
    captured: dict = {}

    def fake_handler(self, params, command, classification):  # type: ignore[override]
        captured["params"] = params
        captured["command"] = command
        captured["classification"] = classification
        return "handled-material"

    agent._handle_material_search = MethodType(fake_handler, agent)  # type: ignore[assignment]

    result = agent.process_command("найди цену на кирпич")

    assert result == "handled-material"
    assert captured["params"]["material_name"] == "Bricks"
    assert captured["command"] == "найди цену на кирпич"


def test_process_command_handles_project_chat_context() -> None:
    payload = json.dumps(
        {
            "command_type": "PROJECT_CHAT",
            "parameters": {"topic": "архитектура"},
        }
    )
    agent = _build_agent([payload])
    captured: dict = {}

    def fake_project_chat(self, params, command, classification, context=None):  # type: ignore[override]
        captured["params"] = params
        captured["context"] = context
        return "chat-response"

    agent._handle_project_chat = MethodType(fake_project_chat, agent)  # type: ignore[assignment]

    result = agent.process_command(
        "расскажи про проект",
        context={"project_context": "README"},
    )

    assert result == "chat-response"
    assert captured["params"]["topic"] == "архитектура"
    assert captured["context"]["project_context"] == "README"


class DummyFallbackOpenAI:
    def __init__(self, config):
        self.config = config
        completions = types.SimpleNamespace(create=lambda **_: None)
        self.chat = types.SimpleNamespace(completions=completions)
        self.responses = types.SimpleNamespace(create=lambda **_: None)


def test_construction_agent_requires_credentials():
    config = ConstructionAIAgentConfig(openai_api_key=None, enable_local_llm=False)
    with pytest.raises(ValueError):
        ConstructionAIAgent(config)


def test_construction_agent_normalizes_temperature(monkeypatch, tmp_path):
    monkeypatch.setattr("unified_agent.FallbackOpenAI", DummyFallbackOpenAI)

    config = ConstructionAIAgentConfig(
        openai_api_key="test-key",
        llm_model="gpt-5-nano",
        temperature=0.4,
        cache_db_path=str(tmp_path / "cache.db"),
        enable_local_db=False,
        enable_materials_db_assistant=False,
        enable_project_chat=False,
    )

    agent = ConstructionAIAgent(config)

    assert agent.config.temperature == 1.0
    assert agent.openai_client.config.primary_api_key == "test-key"
    assert agent.sheets_ai is None


def test_construction_agent_local_only(monkeypatch, tmp_path):
    monkeypatch.setattr("unified_agent.FallbackOpenAI", DummyFallbackOpenAI)

    config = ConstructionAIAgentConfig(
        openai_api_key=None,
        enable_local_llm=True,
        local_llm_base_url="http://localhost:9000",
        local_llm_model="local-model",
        cache_db_path=str(tmp_path / "cache.db"),
        enable_local_db=False,
        enable_materials_db_assistant=False,
        enable_project_chat=False,
    )

    agent = ConstructionAIAgent(config)

    assert agent.openai_client.config.local_enabled is True
    assert agent.openai_client.config.local_base_url == "http://localhost:9000"
