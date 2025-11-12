import types

import pytest

from unified_agent import ConstructionAIAgent, ConstructionAIAgentConfig


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
