import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unified_agent import ConstructionAIAgentConfig


def _base_config(**overrides):
    defaults = dict(
        openai_api_key="test-key",
        enable_local_db=False,
        enable_materials_db_assistant=False,
        local_materials_csv_path=None,
    )
    defaults.update(overrides)
    return ConstructionAIAgentConfig(**defaults)


def test_validate_requires_openai_or_local() -> None:
    config = _base_config(openai_api_key=None, enable_local_llm=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
        config.validate()


def test_validate_accepts_local_llm_without_openai() -> None:
    config = _base_config(
        openai_api_key=None,
        enable_local_llm=True,
        local_llm_base_url="http://127.0.0.1:1234",
        local_llm_model="qwen",
    )

    config.validate()


def test_validate_requires_service_account_for_sheets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_SERVICE_ACCOUNT_JSON", raising=False)

    config = _base_config(
        google_sheet_id="sheet",
        google_service_account_path=None,
    )

    with pytest.raises(ValueError, match="Google Sheets integration requires"):
        config.validate()


def test_validate_requires_existing_local_csv(tmp_path: Path) -> None:
    missing_csv = tmp_path / "missing.csv"
    config = _base_config(
        local_materials_csv_path=str(missing_csv),
        enable_local_db=True,
    )

    with pytest.raises(ValueError, match="Local materials CSV not found"):
        config.validate()


def test_validate_accepts_materials_db_assistant_with_sheet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_JSON", "{}")

    config = _base_config(
        enable_materials_db_assistant=True,
        materials_db_sheet_id="sheet",
    )

    config.validate()


def test_validate_requires_source_for_materials_db_assistant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_SERVICE_ACCOUNT_JSON", raising=False)

    config = _base_config(
        enable_materials_db_assistant=True,
        materials_db_sheet_id=None,
    )

    with pytest.raises(ValueError, match="Materials DB assistant requires"):
        config.validate()
