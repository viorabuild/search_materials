import types

import pytest

import llm_provider
import unified_agent
from unified_agent import ConstructionAIAgent, ConstructionAIAgentConfig


class DummyOpenAI:
    def __init__(self, *_, **__):
        noop = lambda *args, **kwargs: None  # noqa: ARG005
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=noop))
        self.responses = types.SimpleNamespace(create=noop)


@pytest.fixture(autouse=True)
def stub_openai(monkeypatch):
    monkeypatch.setattr(llm_provider, "OpenAI", DummyOpenAI)
    yield


def test_get_diagnostics_reports_disabled_components(tmp_path, monkeypatch):
    monkeypatch.delenv("GOOGLE_SERVICE_ACCOUNT_JSON", raising=False)

    config = ConstructionAIAgentConfig(
        openai_api_key="test-key",
        google_sheet_id=None,
        enable_web_search=False,
        enable_local_db=False,
        enable_materials_db_assistant=False,
        enable_project_chat=False,
        cache_db_path=str(tmp_path / "cache" / "materials.db"),
    )

    agent = ConstructionAIAgent(config)
    diagnostics = agent.get_diagnostics()

    components = diagnostics["components"]

    assert components["google_sheets_ai"]["status"] == "disabled"
    assert components["advanced_agent"]["status"] == "disabled"
    assert components["local_materials_db"]["status"] == "disabled"
    assert components["materials_db_assistant"]["status"] == "disabled"
    assert components["project_chat_agent"]["status"] == "disabled"
    assert components["openai_client"]["status"] == "success"
    assert components["cache_manager"]["status"] == "success"
    assert diagnostics["advanced_agent_error"] is None
    assert "estimate_checker" in components


def test_get_diagnostics_reports_errors(tmp_path, monkeypatch):
    monkeypatch.delenv("GOOGLE_SERVICE_ACCOUNT_JSON", raising=False)

    class BoomSheetsAI:
        def __init__(self, *_, **__):
            raise RuntimeError("sheets boom")

    class BoomLocalDB:
        def __init__(self, *_, **__):
            raise RuntimeError("local db boom")

    class BoomAssistant:
        def __init__(self, *_, **__):
            raise RuntimeError("assistant boom")

    class BoomProjectChat:
        def __init__(self, *_, **__):
            raise RuntimeError("chat boom")

    original_import_module = unified_agent.importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == "advanced_agent":
            raise RuntimeError("advanced boom")
        return original_import_module(name, *args, **kwargs)

    monkeypatch.setattr(unified_agent, "GoogleSheetsAI", BoomSheetsAI)
    monkeypatch.setattr(unified_agent, "LocalMaterialDatabase", BoomLocalDB)
    monkeypatch.setattr(unified_agent, "MaterialsDatabaseAssistant", BoomAssistant)
    monkeypatch.setattr(unified_agent, "ProjectChatAgent", BoomProjectChat)
    monkeypatch.setattr(unified_agent.importlib, "import_module", fake_import_module)

    config = ConstructionAIAgentConfig(
        openai_api_key="test-key",
        google_sheet_id="sheet123",
        enable_web_search=True,
        enable_local_db=True,
        local_materials_csv_path=str(tmp_path / "materials.csv"),
        enable_materials_db_assistant=True,
        materials_db_sheet_id="sheet-db",
        enable_project_chat=True,
        cache_db_path=str(tmp_path / "cache" / "materials.db"),
    )

    agent = ConstructionAIAgent(config)
    diagnostics = agent.get_diagnostics()
    components = diagnostics["components"]

    assert components["google_sheets_ai"]["status"] == "error"
    assert "sheets boom" in components["google_sheets_ai"]["details"]

    assert components["advanced_agent"]["status"] == "error"
    assert "advanced boom" in components["advanced_agent"]["details"]
    assert diagnostics["advanced_agent_error"] == "advanced boom"

    assert components["gspread_client"]["status"] == "error"
    assert components["local_materials_db"]["status"] == "error"
    assert "local db boom" in components["local_materials_db"]["details"]

    assert components["materials_db_assistant"]["status"] == "error"
    assert "assistant boom" in components["materials_db_assistant"]["details"]
    assert "pending_changes" not in components["materials_db_assistant"]

    assert components["project_chat_agent"]["status"] == "error"
    assert "chat boom" in components["project_chat_agent"]["details"]

    assert components["openai_client"]["status"] == "success"
    assert components["cache_manager"]["status"] == "success"
    assert diagnostics["config"]["enable_web_search"] is True
    assert diagnostics["config"]["google_sheet_configured"] is True

