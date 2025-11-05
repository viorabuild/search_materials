from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Iterable

from google.oauth2 import service_account
from googleapiclient.discovery import build

from app.core.settings import get_settings


def _scopes_list(raw: str) -> list[str]:
    if raw.strip().startswith("["):
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return [str(x) for x in data]
        except json.JSONDecodeError:
            pass
    return [s.strip() for s in raw.split(",") if s.strip()]


@lru_cache
def _get_credentials():
    settings = get_settings()
    scopes = _scopes_list(settings.google_scopes)

    if settings.google_sa_json:
        info: dict[str, Any] = json.loads(settings.google_sa_json)
        creds = service_account.Credentials.from_service_account_info(info, scopes=scopes)
    elif settings.google_sa_file:
        creds = service_account.Credentials.from_service_account_file(settings.google_sa_file, scopes=scopes)
    else:
        raise RuntimeError("Google SA credentials are not configured: set GOOGLE_SA_JSON or GOOGLE_SA_FILE")

    if settings.google_workspace_subject:
        creds = creds.with_subject(settings.google_workspace_subject)
    return creds


@lru_cache
def get_drive_service():
    creds = _get_credentials()
    return build("drive", "v3", credentials=creds, cache_discovery=False)


@lru_cache
def get_sheets_service():
    creds = _get_credentials()
    return build("sheets", "v4", credentials=creds, cache_discovery=False)


def list_all_spreadsheets(page_size: int = 1000) -> list[dict[str, Any]]:
    drive = get_drive_service()
    q = "mimeType='application/vnd.google-apps.spreadsheet' and trashed=false"
    fields = "nextPageToken, files(id, name, modifiedTime, owners(emailAddress, displayName))"
    include = {
        "includeItemsFromAllDrives": True,
        "supportsAllDrives": True,
        "corpora": "allDrives",
    }

    files: list[dict[str, Any]] = []
    page_token: str | None = None
    while True:
        req = (
            drive.files()
            .list(q=q, fields=fields, pageSize=page_size, pageToken=page_token, **include)
        )
        resp = req.execute()
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return files


def get_sheet_values(spreadsheet_id: str, rng: str) -> dict[str, Any]:
    sheets = get_sheets_service()
    req = sheets.spreadsheets().values().get(spreadsheetId=spreadsheet_id, range=rng)
    resp = req.execute()
    return resp
