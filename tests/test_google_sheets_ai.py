import types

import pytest

from markdown_ai import GoogleSheetsAI


class FakeWorksheet:
    def __init__(self, title="Sheet1"):
        self.title = title
        self.id = 123
        self._properties = {"sheetId": 123}
        self.cleared = False
        self.updated_calls = []
        self.append_calls = []

    def clear(self):
        self.cleared = True

    def update(self, range_name, values, value_input_option=None):
        self.updated_calls.append(
            {
                "range": range_name,
                "values": values,
                "value_input_option": value_input_option,
            }
        )

    def append_rows(self, rows, value_input_option=None):
        self.append_calls.append({"rows": rows, "value_input_option": value_input_option})

    def get_all_values(self):
        return []

    def update_cell(self, row, col, value):
        self.updated_calls.append({"cell": (row, col), "value": value})


class FakeSpreadsheet:
    def __init__(self, worksheet):
        self._worksheet = worksheet
        self.title = "Demo"
        self.updated_title = None
        self.batch_requests = None

    def worksheet(self, name):
        if name != self._worksheet.title:
            from gspread.exceptions import WorksheetNotFound

            raise WorksheetNotFound(name)
        return self._worksheet

    def add_worksheet(self, title, rows, cols):
        self._worksheet = FakeWorksheet(title)
        return self._worksheet

    def update_title(self, title):
        self.updated_title = title

    def worksheets(self):
        return [self._worksheet]

    def batch_update(self, payload):
        self.batch_requests = payload


class FakeGspreadClient:
    def __init__(self, spreadsheet):
        self._spreadsheet = spreadsheet
        self.opened_key = None

    def open_by_key(self, key):
        self.opened_key = key
        return self._spreadsheet


@pytest.fixture
def sheets_ai(monkeypatch):
    worksheet = FakeWorksheet()
    spreadsheet = FakeSpreadsheet(worksheet)
    client = FakeGspreadClient(spreadsheet)

    monkeypatch.setattr(GoogleSheetsAI, "_create_gspread_client", lambda self: client)

    ai = GoogleSheetsAI(
        sheet_id="sheet-id",
        worksheet_name="Sheet1",
        openai_client=types.SimpleNamespace(),
        llm_model="gpt-test",
    )

    return ai, worksheet, spreadsheet


def test_write_sheet_data_clears_and_updates(sheets_ai):
    ai, worksheet, spreadsheet = sheets_ai

    ai.write_sheet_data([["A", "B"]], title="Updated Title")

    assert worksheet.cleared is True
    assert worksheet.updated_calls[-1] == {
        "range": "A1",
        "values": [["A", "B"]],
        "value_input_option": "USER_ENTERED",
    }
    assert spreadsheet.updated_title == "Updated Title"


def test_format_range_builds_batch_request(sheets_ai):
    ai, worksheet, spreadsheet = sheets_ai

    message = ai.format_range("A1:B2", {"backgroundColor": "#ff0000", "bold": True})

    assert "Форматирование применено" in message

    request = spreadsheet.batch_requests["requests"][0]["repeatCell"]
    assert request["range"]["sheetId"] == 123

    user_format = request["cell"]["userEnteredFormat"]
    assert user_format["backgroundColor"] == {"red": 1.0, "green": 0.0, "blue": 0.0}
    assert user_format["textFormat"]["bold"] is True

    fields = request["fields"].split(",")
    assert "userEnteredFormat.backgroundColor" in fields
    assert "userEnteredFormat.textFormat.bold" in fields
