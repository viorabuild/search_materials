from pathlib import Path
import sys
from unittest import mock

import pytest
import requests

from gspread.exceptions import APIError

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import markdown_ai
from markdown_ai import GoogleAPIRateLimitError, GoogleSheetsAI


def _build_ai(monkeypatch, *, tokens="3", interval="60"):
    worksheet_mock = mock.Mock()
    worksheet_mock.title = "Sheet1"
    worksheet_mock.get_all_values.return_value = [["header"]]

    spreadsheet_mock = mock.Mock()
    spreadsheet_mock.title = "Test"
    spreadsheet_mock.worksheet.return_value = worksheet_mock
    spreadsheet_mock.worksheets.return_value = [worksheet_mock]

    client_mock = mock.Mock()
    client_mock.open_by_key.return_value = spreadsheet_mock

    monkeypatch.setenv("GOOGLE_API_RATE_TOKENS", tokens)
    monkeypatch.setenv("GOOGLE_API_RATE_INTERVAL_SECONDS", interval)
    monkeypatch.setenv("NETWORK_RETRY_ATTEMPTS", "3")
    monkeypatch.setenv("NETWORK_RETRY_MAX_WAIT_SECONDS", "1")
    monkeypatch.setenv("GOOGLE_API_RETRY_ATTEMPTS", "2")
    monkeypatch.setenv("GOOGLE_API_RETRY_MAX_WAIT_SECONDS", "1")

    monkeypatch.setattr(GoogleSheetsAI, "_create_gspread_client", lambda self: client_mock)
    monkeypatch.setattr(GoogleSheetsAI, "_load_allowed_domains", staticmethod(lambda: []))

    ai = GoogleSheetsAI(sheet_id="sheet", worksheet_name="Sheet1", openai_client=mock.Mock())
    ai.spreadsheet = spreadsheet_mock
    ai.worksheet = worksheet_mock
    return ai, worksheet_mock


def test_fetch_web_content_retries(monkeypatch):
    ai, _ = _build_ai(monkeypatch)

    response_mock = mock.Mock()
    response_mock.headers = {"Content-Type": "text/html"}
    response_mock.text = "hello"
    response_mock.raise_for_status.return_value = None

    call_results = [
        requests.exceptions.Timeout("timeout"),
        requests.exceptions.ConnectionError("conn"),
        response_mock,
    ]

    def side_effect(*args, **kwargs):
        result = call_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    request_mock = mock.Mock(side_effect=side_effect)
    monkeypatch.setattr(markdown_ai.requests, "get", request_mock)

    result = ai.fetch_web_content("https://example.com")

    assert result == "hello"
    assert request_mock.call_count == 3


def _make_api_error(status_code=500):
    response = requests.Response()
    response.status_code = status_code
    response._content = b"{}"
    return APIError(response)


def test_append_to_sheet_retries_after_api_error(monkeypatch):
    ai, worksheet = _build_ai(monkeypatch)

    worksheet.append_rows.side_effect = [_make_api_error(), None]

    ai.append_to_sheet([["header"], ["value"]])

    assert worksheet.append_rows.call_count == 2


def test_google_api_rate_limit_raises(monkeypatch):
    ai, worksheet = _build_ai(monkeypatch, tokens="1", interval="3600")

    ai.append_to_sheet([["header"], ["value"]])

    worksheet.append_rows.reset_mock()

    with pytest.raises(GoogleAPIRateLimitError) as exc:
        ai.append_to_sheet([["header"], ["other"]])

    assert exc.value.code == "GOOGLE_RATE_LIMIT_EXCEEDED"
    worksheet.append_rows.assert_not_called()
