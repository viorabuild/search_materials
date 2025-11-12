"""Assistant for managing the materials and works database with staged changes."""

from __future__ import annotations

import csv
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from local_materials_db import LocalMaterialDatabase

logger = logging.getLogger(__name__)


def _normalize_key(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


_FIELD_ALIASES: Dict[str, str] = {
    "name": "Nome ",
    "nampt": "Nome ",
    "nomeru": "Название RU ",
    "nomeruai": "Название RU [AI]",
    "id": "ID",
    "category": "Категория",
    "group": "Группа",
    "unit": "Ед. Измерения",
    "supplier": "Последний Поставщик",
    "lastsupplier": "Последний Поставщик",
    "price": "Стоимость",
    "cost": "Стоимость",
    "quote": "Последняя Котировка",
    "quotes": "💯 Котировки поставщиков",
    "url": "Поставщики",
    "created": "Создано",
    "date": "Дата Последней Котировки",
    "notes": "💯 Котировки поставщиков",
}


def _norm_sheet_name(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str):
        return None
    return value.strip().lower() or None


@dataclass
class PendingChange:
    """Represents a staged modification awaiting confirmation."""

    change_id: str
    action: str  # update, insert, delete
    record_id: Optional[str]
    fields: Dict[str, str]
    reason: Optional[str] = None
    before: Optional[Dict[str, str]] = None
    after: Optional[Dict[str, str]] = None
    worksheet: Optional[str] = None
    created_at: float = field(default_factory=time.time)


try:
    import gspread
except ImportError:  # pragma: no cover
    gspread = None

if gspread:
    try:
        from gspread.exceptions import WorksheetNotFound
    except Exception:  # pragma: no cover
        WorksheetNotFound = Exception
else:  # pragma: no cover
    class WorksheetNotFound(Exception):
        """Placeholder when gspread is unavailable."""
        pass


class MaterialsDatabaseAssistant:
    """LLM-assisted workflow for safely managing the materials database (CSV or Google Sheets)."""

    def __init__(
        self,
        csv_path: Optional[Path],
        llm_client: Any,
        llm_model: str,
        local_db: Optional[LocalMaterialDatabase] = None,
        *,
        sheet_id: Optional[str] = None,
        worksheet_name: Optional[str] = None,
        worksheet_names: Optional[Sequence[str]] = None,
        gspread_client: Optional[Any] = None,
        sync_csv: bool = True,
        pending_cache_path: Optional[Path] = None,
        pending_expiration_seconds: Optional[int] = 7 * 24 * 60 * 60,
    ) -> None:
        if csv_path is not None and not isinstance(csv_path, Path):
            csv_path = Path(csv_path)
        self.csv_path = csv_path
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.local_db = local_db
        self.sheet_id = sheet_id
        self._worksheet_names = [
            name.strip() for name in (worksheet_names or []) if name and name.strip()
        ]
        if worksheet_name and worksheet_name.strip() and not self._worksheet_names:
            self._worksheet_names = [worksheet_name.strip()]
        self.worksheet_name = (
            self._worksheet_names[0] if self._worksheet_names else (worksheet_name or None)
        )
        self._gspread_client = gspread_client
        self._spreadsheet = None
        self._sheets: Dict[str, Optional[Any]] = {}
        self._sync_csv = bool(sync_csv and self.csv_path is not None)
        self._pending_cache_path = (
            Path(pending_cache_path)
            if pending_cache_path is not None
            else Path("cache") / "materials_pending.json"
        )
        self._pending_expiration_seconds = pending_expiration_seconds
        self._last_row_fetch_failed = False

        if self.sheet_id and gspread is None:
            logger.warning(
                "Google Sheets backend requested but gspread is not installed."
            )

        if self.sheet_id and self._gspread_client is None and gspread is not None:
            try:
                self._gspread_client = gspread.service_account()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Default gspread.service_account() failed: %s", exc)

        if self.sheet_id and self._gspread_client and gspread is not None:
            try:
                self._spreadsheet = self._gspread_client.open_by_key(self.sheet_id)
                if self._worksheet_names:
                    for name in self._worksheet_names:
                        sheet_obj = None
                        try:
                            sheet_obj = self._spreadsheet.worksheet(name)
                        except WorksheetNotFound:
                            logger.warning(
                                "Worksheet '%s' not found in spreadsheet '%s'.",
                                name,
                                self.sheet_id,
                            )
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(
                                "Failed to load worksheet '%s': %s",
                                name,
                                exc,
                            )
                        self._sheets[name] = sheet_obj
                    available = [
                        sheet_name
                        for sheet_name, sheet_obj in self._sheets.items()
                        if sheet_obj is not None
                    ]
                    if available:
                        self.worksheet_name = available[0]
                    elif self._spreadsheet is not None:
                        default_sheet = self._spreadsheet.sheet1
                        self._sheets[default_sheet.title] = default_sheet
                        self.worksheet_name = default_sheet.title
                        self._worksheet_names = [default_sheet.title]
                else:
                    sheet_obj = None
                    try:
                        if self.worksheet_name:
                            sheet_obj = self._spreadsheet.worksheet(self.worksheet_name)
                        else:
                            sheet_obj = self._spreadsheet.sheet1
                    except WorksheetNotFound:
                        sheet_obj = self._spreadsheet.sheet1
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Failed to load worksheet '%s': %s", self.worksheet_name, exc)
                        sheet_obj = self._spreadsheet.sheet1
                    self.worksheet_name = sheet_obj.title
                    self._worksheet_names = [sheet_obj.title]
                    self._sheets[sheet_obj.title] = sheet_obj
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to bind Google Sheet for materials DB: %s", exc)
                self._spreadsheet = None
                self._sheets = {}
        else:
            if not self._worksheet_names and self.worksheet_name:
                self._worksheet_names = [self.worksheet_name]

        self._headers: List[str] = []
        self._header_lookup: Dict[str, str] = {}
        self._pending: Dict[str, PendingChange] = {}
        self._pending_counter: int = 0
        self._load_headers()
        self._load_pending_state()
        if (
            any(sheet is not None for sheet in self._sheets.values())
            and self._sync_csv
            and self.csv_path
            and not self.csv_path.exists()
        ):
            try:
                snapshot = self._read_all_rows()
                if snapshot:
                    self._write_csv_file(snapshot)
            except Exception:  # noqa: BLE001
                logger.debug("Initial CSV mirror sync failed.", exc_info=True)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def handle_request(
        self,
        command: str,
        hints: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Interpret and execute a user request for the materials database."""
        if not command or not command.strip():
            return "❌ Пустой запрос для базы материалов."

        classification = self._classify_command(command, hints=hints)
        if not classification:
            return "❌ Не удалось понять запрос. Попробуйте сформулировать его иначе."

        intent = (classification.get("intent") or "help").strip().lower()
        record_id = classification.get("record_id") or classification.get("id")
        if isinstance(record_id, str):
            record_id = record_id.strip() or None
        query = classification.get("query")
        fields_raw = classification.get("fields") or {}
        mapped_fields = self._map_fields(fields_raw)
        reason = classification.get("reason")
        confirmation_id = classification.get("confirmation_id")
        if isinstance(confirmation_id, str):
            confirmation_id = confirmation_id.strip() or None
        worksheet_hint = (
            classification.get("worksheet")
            or classification.get("sheet")
            or classification.get("tab")
        )
        if isinstance(worksheet_hint, str):
            worksheet_hint = worksheet_hint.strip() or None

        if not self._headers and intent not in {"help", "list_pending"}:
            return (
                "❌ Не удалось определить заголовки таблицы материалов. "
                "Убедитесь, что в первой строке указаны названия колонок."
            )

        if intent == "lookup":
            return self._handle_lookup(record_id=record_id, query=query, worksheet=worksheet_hint)
        if intent == "stage_update":
            return self._stage_update(record_id, mapped_fields, reason, worksheet_hint)
        if intent == "stage_insert":
            return self._stage_insert(mapped_fields, reason, worksheet_hint)
        if intent == "stage_delete":
            return self._stage_delete(record_id, reason, worksheet_hint)
        if intent == "confirm_change":
            return self._confirm_change(confirmation_id)
        if intent == "cancel_change":
            return self._cancel_change(confirmation_id)
        if intent == "list_pending":
            return self._list_pending()
        if intent == "help":
            return self._help_message()

        # Fallback: attempt lookup if query provided
        if query or record_id:
            return self._handle_lookup(record_id=record_id, query=query, worksheet=worksheet_hint)
        return self._help_message()

    def pending_count(self) -> int:
        """Return the number of staged changes."""
        if self._cleanup_pending():
            self._save_pending_state()
        return len(self._pending)

    def quick_lookup(
        self,
        query: str,
        *,
        limit: int = 5,
        worksheet: Optional[str] = None,
    ) -> Optional[str]:
        """Return a short summary for the provided query without engaging the LLM."""
        if not query or not query.strip():
            return None
        summary = self._handle_lookup(None, query.strip(), limit=limit, worksheet=worksheet)
        if not summary:
            return None
        default_responses = {
            "ℹ️ Уточните, что именно нужно найти (укажите ID или запрос).",
        }
        normalized = summary.strip()
        if normalized in default_responses or normalized.startswith("ℹ️ По запросу"):
            return None
        return summary

    # ------------------------------------------------------------------ #
    # Classification and normalization helpers                           #
    # ------------------------------------------------------------------ #

    def _classify_command(
        self,
        command: str,
        hints: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        headers_text = ", ".join(self._headers) if self._headers else "(столбцы не обнаружены)"
        sheets_text = ", ".join(self._worksheet_names) if self._worksheet_names else "(не заданы)"
        system_prompt = (
            "Ты помогаешь вести таблицу строительных материалов и работ (CSV или Google Sheets).\n"
            "В таблице есть столбцы: "
            f"{headers_text}.\n"
            "Доступные листы: "
            f"{sheets_text}.\n\n"
            "Определи намерение пользователя и верни JSON вида:\n"
            "{\n"
            '  "intent": "lookup|stage_update|stage_insert|stage_delete|confirm_change|cancel_change|list_pending|help",\n'
            '  "record_id": "ID записи или null",\n'
            '  "query": "строка поиска или null",\n'
            '  "fields": { "Колонка": "значение", ... },\n'
            '  "reason": "Обоснование изменений или null",\n'
            '  "confirmation_id": "ID отложенного изменения для подтверждения/отмены или null",\n'
            '  "worksheet": "название листа или null"\n'
            "}\n\n"
            "Правила:\n"
            "- Для обновления или удаления обязательно указывай record_id.\n"
            "- Для добавления стремись задать хотя бы поля ID, Название RU , Категория, Стоимость.\n"
            "- Никогда не выполняй изменения напрямую — используй stage_* и жди подтверждения через confirm_change.\n"
            "- Если листов несколько, обязательно указывай worksheet (используй одно из доступных названий).\n"
            "- Если запрос не ясен, возвращай intent=help.\n"
            "- Всегда возвращай строго валидный JSON без комментариев."
        )

        messages = [{"role": "system", "content": system_prompt}]
        if hints:
            try:
                hints_json = json.dumps(hints, ensure_ascii=False)
                messages.append(
                    {"role": "system", "content": f"Контекст от приложения: {hints_json}"}
                )
            except Exception:  # noqa: BLE001
                logger.debug("Failed to serialize hints for materials assistant.")

        messages.append({"role": "user", "content": command.strip()})

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM classification failed: %s", exc)
            return None

        content = ""
        if response and getattr(response, "choices", None):
            content = response.choices[0].message.content or ""

        return self._parse_json(content)

    def _parse_json(self, payload: str) -> Optional[Dict[str, Any]]:
        text = (payload or "").strip()
        if not text:
            return None

        if text.startswith("```"):
            fence_end = text.rfind("```")
            if fence_end != -1:
                text = text.split("\n", 1)[-1]
                text = text[:fence_end].strip()

        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            logger.warning("Materials DB assistant: invalid JSON response: %s", text)
        return None

    def _map_fields(self, fields: Dict[str, Any]) -> Dict[str, str]:
        mapped: Dict[str, str] = {}
        for raw_key, raw_value in (fields or {}).items():
            header = self._match_header(str(raw_key))
            if not header:
                continue

            value = raw_value
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            if value is None:
                continue
            mapped[header] = str(value).strip()
        return mapped

    def _match_header(self, key: str) -> Optional[str]:
        normalized = _normalize_key(key)
        if not normalized:
            return None
        if normalized in self._header_lookup:
            return self._header_lookup[normalized]
        alias_target = _FIELD_ALIASES.get(normalized)
        if alias_target:
            alias_norm = _normalize_key(alias_target)
            return self._header_lookup.get(alias_norm, alias_target)
        return None

    # ------------------------------------------------------------------ #
    # Lookup helpers                                                     #
    # ------------------------------------------------------------------ #

    def _handle_lookup(
        self,
        record_id: Optional[str],
        query: Optional[str],
        *,
        limit: int = 5,
        worksheet: Optional[str] = None,
    ) -> str:
        if record_id:
            record = self._find_row_by_id(record_id)
            if not record:
                return f"ℹ️ Запись с ID {record_id} не найдена."
            return self._format_record_details(record)

        search_query = (query or "").strip()
        if not search_query:
            return "ℹ️ Уточните, что именно нужно найти (укажите ID или запрос)."

        target_sheet_norm = _norm_sheet_name(worksheet)
        matches: List[str] = []
        if self.local_db and self.local_db.available:
            results = self.local_db.search(search_query, limit=limit)
            for match in results:
                matches.append(
                    f"- {match.display_name} | {match.supplier} | {match.price} | {match.url}"
                )

        if not matches:
            # Fallback: simple substring search in CSV
            rows = self._read_all_rows(limit=limit, query=search_query, worksheet=worksheet)
            for row in rows:
                sheet_label = row.get("_worksheet")
                if target_sheet_norm and _norm_sheet_name(sheet_label) != target_sheet_norm:
                    continue
                sheet_suffix = f" (лист: {sheet_label})" if sheet_label else ""
                matches.append(
                    f"- {row.get('ID', 'без ID')} | {row.get('Название RU ', row.get('Nome ', ''))} | {row.get('Стоимость', 'N/A')}{sheet_suffix}"
                )

        if not matches:
            return f"ℹ️ По запросу '{search_query}' ничего не найдено."

        header = f"🔎 Найдено {len(matches)} совпадений для '{search_query}':"
        return "\n".join([header, *matches])

    def _format_record_details(self, record: Dict[str, str]) -> str:
        parts = [f"🔎 Запись {record.get('ID', 'без ID')}"]
        sheet_name = record.get("_worksheet")
        if sheet_name:
            parts.append(f"Лист: {sheet_name}")
        parts.extend(
            [
                f"Название: {record.get('Название RU ', record.get('Nome ', 'N/A'))}",
                f"Категория: {record.get('Категория', 'N/A')}",
                f"Группа: {record.get('Группа', 'N/A')}",
                f"Поставщик: {record.get('Последний Поставщик', 'N/A')}",
                f"Стоимость: {record.get('Стоимость', 'N/A')}",
            ]
        )
        url = record.get("Поставщики")
        if url:
            parts.append(f"Ссылки: {url}")
        notes = record.get("💯 Котировки поставщиков")
        if notes:
            parts.append(f"Заметки: {notes}")
        return "\n".join(parts)

    # ------------------------------------------------------------------ #
    # Staging operations                                                 #
    # ------------------------------------------------------------------ #

    def _stage_update(
        self,
        record_id: Optional[str],
        fields: Dict[str, str],
        reason: Optional[str],
        worksheet_hint: Optional[str],
    ) -> str:
        if not record_id:
            return "❌ Для обновления нужно указать ID записи."
        if not fields:
            return "ℹ️ Укажите хотя бы одно поле для изменения."

        current = self._find_row_by_id(record_id)
        if not current:
            return f"❌ Запись с ID {record_id} не найдена."

        target_sheet = current.get("_worksheet") or worksheet_hint
        try:
            resolved_sheet = self._require_sheet_name(target_sheet)
        except ValueError as exc:
            return f"❌ {exc}"

        updated = dict(current)
        for header, value in fields.items():
            updated[header] = value
        if resolved_sheet:
            updated["_worksheet"] = resolved_sheet

        diff_lines = self._diff_records(current, updated)
        if resolved_sheet and _norm_sheet_name(current.get("_worksheet")) != _norm_sheet_name(resolved_sheet):
            diff_lines.insert(0, f"- Лист: '{current.get('_worksheet', 'не указан')}' → '{resolved_sheet}'")

        if not diff_lines:
            return "ℹ️ Новые значения совпадают с текущими — изменения не требуются."

        change = PendingChange(
            change_id=self._generate_change_id(),
            action="update",
            record_id=record_id,
            fields=fields,
            reason=reason,
            before=current,
            after=updated,
            worksheet=resolved_sheet,
        )
        self._pending[change.change_id] = change
        self._save_pending_state()

        summary = "\n".join(diff_lines)
        return (
            f"📝 Подготовлено обновление записи {record_id} (изменение {change.change_id}).\n"
            f"{summary}\n"
            "Чтобы применить изменения, отправьте команду вроде «подтверди изменение "
            f"{change.change_id}». Для отмены — «отмени изменение {change.change_id}»."
        )

    def _stage_insert(
        self,
        fields: Dict[str, str],
        reason: Optional[str],
        worksheet_hint: Optional[str],
    ) -> str:
        if not fields.get("ID"):
            return "❌ Для добавления записи необходимо указать поле ID."

        if self._find_row_by_id(fields["ID"]):
            return f"❌ Запись с ID {fields['ID']} уже существует."

        try:
            resolved_sheet = self._require_sheet_name(worksheet_hint)
        except ValueError as exc:
            return f"❌ {exc}"

        new_row = {header: "" for header in self._headers}
        for header, value in fields.items():
            new_row[header] = value
        if resolved_sheet:
            new_row["_worksheet"] = resolved_sheet

        change = PendingChange(
            change_id=self._generate_change_id(),
            action="insert",
            record_id=fields["ID"],
            fields=fields,
            reason=reason,
            before=None,
            after=new_row,
            worksheet=resolved_sheet,
        )
        self._pending[change.change_id] = change
        self._save_pending_state()

        preview = self._format_record_details(new_row)
        return (
            f"📝 Подготовлено добавление новой записи (изменение {change.change_id}).\n"
            f"{preview}\n"
            "Подтвердите действие командой «подтверди изменение "
            f"{change.change_id}», либо отмените его."
        )

    def _stage_delete(
        self,
        record_id: Optional[str],
        reason: Optional[str],
        worksheet_hint: Optional[str],
    ) -> str:
        if not record_id:
            return "❌ Для удаления требуется указать ID записи."

        current = self._find_row_by_id(record_id)
        if not current:
            return f"❌ Запись с ID {record_id} не найдена."

        try:
            resolved_sheet = self._require_sheet_name(current.get("_worksheet") or worksheet_hint)
        except ValueError as exc:
            return f"❌ {exc}"

        change = PendingChange(
            change_id=self._generate_change_id(),
            action="delete",
            record_id=record_id,
            fields={},
            reason=reason,
            before=current,
            after=None,
            worksheet=resolved_sheet,
        )
        self._pending[change.change_id] = change
        self._save_pending_state()

        preview = self._format_record_details(current)
        return (
            f"🗑 Подготовлено удаление записи {record_id} (изменение {change.change_id}).\n"
            f"{preview}\n"
            "Подтвердите действие командой «подтверди изменение "
            f"{change.change_id}», либо отмените его."
        )

    # ------------------------------------------------------------------ #
    # Pending change management                                          #
    # ------------------------------------------------------------------ #

    def _confirm_change(self, change_id: Optional[str]) -> str:
        if not change_id:
            return "ℹ️ Укажите ID изменения для подтверждения."

        change = self._pending.get(change_id)
        if not change:
            return f"ℹ️ Изменение {change_id} не найдено среди подготовленных."

        try:
            if change.action == "update":
                self._apply_update(change)
            elif change.action == "insert":
                self._apply_insert(change)
            elif change.action == "delete":
                self._apply_delete(change)
            else:
                return f"❌ Неизвестный тип изменения: {change.action}"
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to apply change %s: %s", change_id, exc)
            return f"❌ Не удалось применить изменение {change_id}: {exc}"

        self._pending.pop(change_id, None)
        self._save_pending_state()
        self._refresh_local_db()
        return f"✅ Изменение {change_id} применено."

    def _cancel_change(self, change_id: Optional[str]) -> str:
        if not change_id:
            return "ℹ️ Укажите ID изменения для отмены."
        if change_id not in self._pending:
            return f"ℹ️ Изменение {change_id} не найдено."
        self._pending.pop(change_id, None)
        self._save_pending_state()
        return f"✅ Изменение {change_id} отменено."

    # ------------------------------------------------------------------ #
    # Pending persistence helpers                                       #
    # ------------------------------------------------------------------ #

    def _load_pending_state(self) -> None:
        if not self._pending_cache_path:
            return

        data: Any = None
        try:
            if self._pending_cache_path.exists():
                with self._pending_cache_path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
        except Exception:  # noqa: BLE001
            logger.debug("Failed to load pending changes cache.", exc_info=True)
            data = None

        loaded: Dict[str, PendingChange] = {}
        saved_counter: int = 0
        if isinstance(data, dict):
            raw_changes = data.get("changes")
            if isinstance(data.get("counter"), int):
                saved_counter = max(int(data["counter"]), 0)
        else:
            raw_changes = data

        if isinstance(raw_changes, dict):
            iterable = raw_changes.values()
        elif isinstance(raw_changes, list):
            iterable = raw_changes
        else:
            iterable = []

        for item in iterable:
            change = self._deserialize_change(item)
            if change is None:
                continue
            loaded[change.change_id] = change

        self._pending = loaded
        self._restore_pending_counter(saved_counter)
        if self._cleanup_pending():
            self._save_pending_state()

    def _save_pending_state(self) -> None:
        if not self._pending_cache_path:
            return

        try:
            self._pending_cache_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:  # noqa: BLE001
            pass

        payload = {
            "version": 1,
            "counter": self._pending_counter,
            "changes": [
                self._serialize_change(change)
                for change in sorted(
                    self._pending.values(), key=lambda item: item.created_at
                )
            ],
        }

        try:
            with self._pending_cache_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
        except Exception:  # noqa: BLE001
            logger.debug("Failed to persist pending changes cache.", exc_info=True)

    def _serialize_change(self, change: PendingChange) -> Dict[str, Any]:
        return {
            "change_id": change.change_id,
            "action": change.action,
            "record_id": change.record_id,
            "fields": change.fields,
            "reason": change.reason,
            "before": change.before,
            "after": change.after,
            "worksheet": change.worksheet,
            "created_at": change.created_at,
        }

    def _deserialize_change(self, payload: Any) -> Optional[PendingChange]:
        if not isinstance(payload, dict):
            return None
        change_id = payload.get("change_id")
        change_id = str(change_id).strip() if change_id else ""
        action = payload.get("action")
        action = str(action).strip() if action else ""
        if not action:
            return None

        try:
            change = PendingChange(
                change_id=change_id,
                action=action,
                record_id=payload.get("record_id"),
                fields=dict(payload.get("fields") or {}),
                reason=payload.get("reason"),
                before=dict(payload.get("before") or {}) or None,
                after=dict(payload.get("after") or {}) or None,
                worksheet=payload.get("worksheet"),
                created_at=float(payload.get("created_at") or time.time()),
            )
        except Exception:  # noqa: BLE001
            logger.debug("Failed to deserialize pending change: %s", payload)
            return None

        change.change_id = change.change_id or self._generate_change_id()
        return change

    def _restore_pending_counter(self, saved_counter: int) -> None:
        self._pending_counter = max(int(saved_counter or 0), 0)
        for change_id in self._pending:
            extracted = self._extract_counter_suffix(change_id)
            if extracted is not None:
                self._pending_counter = max(self._pending_counter, extracted)

    def _cleanup_pending(self) -> bool:
        if not self._pending:
            return False

        expiration = None
        if self._pending_expiration_seconds and self._pending_expiration_seconds > 0:
            expiration = time.time() - self._pending_expiration_seconds

        try:
            rows = self._read_all_rows()
        except Exception:  # noqa: BLE001
            rows = []

        if self._last_row_fetch_failed:
            logger.debug(
                "Skipping pending cleanup because backing store could not be read."
            )
            return False

        existing: Dict[str, Dict[str, str]] = {}
        for row in rows:
            record_id = row.get("ID")
            if record_id:
                existing[record_id] = row

        changed = False
        for change_id, change in list(self._pending.items()):
            if expiration is not None and change.created_at < expiration:
                self._pending.pop(change_id, None)
                changed = True
                continue

            record_id = change.record_id or ""
            if change.action in {"update", "delete"}:
                if not record_id or record_id not in existing:
                    self._pending.pop(change_id, None)
                    changed = True
                    continue
            elif change.action == "insert":
                if record_id and record_id in existing:
                    self._pending.pop(change_id, None)
                    changed = True

        return changed

    @staticmethod
    def _extract_counter_suffix(change_id: str) -> Optional[int]:
        if not isinstance(change_id, str):
            return None
        match = re.search(r"(\d+)$", change_id)
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    def _list_pending(self) -> str:
        if self._cleanup_pending():
            self._save_pending_state()
        if not self._pending:
            return "ℹ️ Нет подготовленных изменений."

        lines = ["🟡 Подготовленные изменения:"]
        for change in sorted(self._pending.values(), key=lambda item: item.created_at):
            summary = self._summarize_change(change)
            lines.append(f"- {change.change_id}: {summary}")
        return "\n".join(lines)

    def _help_message(self) -> str:
        commands = [
            "🔎 «найди запись M-CON-001» — поиск по ID или названию",
            "📄 «покажи лист Материалы» или «найди записи на листе Работы» — поиск в конкретном листе",
            "📝 «обнови цену M-CON-001 на €90» — подготовка изменения",
            "➕ «добавь материал с ID ...» — добавление новой записи",
            "🗑 «удали работу M-WORK-010» — подготовка удаления",
            "📋 «список изменений» — показать подготовленные изменения",
            "✅ «подтверди изменение CHG-1234» — применить обновление",
            "⛔️ «отмени изменение CHG-1234» — отменить подготовку",
        ]
        if self._worksheet_names:
            commands.append("ℹ️ Доступные листы: " + ", ".join(self._worksheet_names))
        return "🤖 Помощник базы материалов готов.\n" + "\n".join(commands)

    # ------------------------------------------------------------------ #
    # Application helpers                                                #
    # ------------------------------------------------------------------ #

    def _require_sheet_name(self, sheet_hint: Optional[str]) -> Optional[str]:
        if not self._worksheet_names:
            return sheet_hint or self.worksheet_name
        if sheet_hint:
            normalized = _norm_sheet_name(sheet_hint)
            for name in self._worksheet_names:
                if _norm_sheet_name(name) == normalized:
                    return name
            raise ValueError(
                f"Лист '{sheet_hint}' не найден. Доступны: {', '.join(self._worksheet_names)}."
            )
        if len(self._worksheet_names) == 1:
            return self._worksheet_names[0]
        raise ValueError(
            "Укажите лист. Доступны: " + ", ".join(self._worksheet_names)
        )

    def _resolve_sheet_object(self, sheet_hint: Optional[str]) -> tuple[Optional[str], Optional[Any]]:
        if not self._sheets:
            return sheet_hint or self.worksheet_name, None

        if sheet_hint:
            normalized = _norm_sheet_name(sheet_hint)
            for name, sheet_obj in self._sheets.items():
                if _norm_sheet_name(name) == normalized:
                    return name, sheet_obj
            return sheet_hint, None

        preferred_names = self._worksheet_names or list(self._sheets.keys())
        for name in preferred_names:
            if name in self._sheets:
                return name, self._sheets.get(name)

        for name, sheet_obj in self._sheets.items():
            if sheet_obj is not None:
                return name, sheet_obj

        any_name = next(iter(self._sheets.keys()))
        return any_name, self._sheets.get(any_name)

    def _apply_update(self, change: PendingChange) -> None:
        sheet_name = change.worksheet or (change.before or {}).get("_worksheet")
        rows = self._read_all_rows(worksheet=sheet_name)
        if not sheet_name and rows:
            sheet_name = rows[0].get("_worksheet")
        updated = False
        for idx, row in enumerate(rows):
            if row.get("ID") == change.record_id:
                rows[idx] = {
                    header: change.after.get(header, "")
                    if change.after
                    else row.get(header, "")
                    for header in self._headers
                }
                if sheet_name:
                    rows[idx]["_worksheet"] = sheet_name
                updated = True
                break
        if not updated:
            raise ValueError(f"Запись с ID {change.record_id} не найдена.")
        self._write_rows(rows, sheet_name)

    def _apply_insert(self, change: PendingChange) -> None:
        sheet_name = change.worksheet or (self._worksheet_names[0] if self._worksheet_names else self.worksheet_name)
        rows = self._read_all_rows(worksheet=sheet_name)
        if not sheet_name and rows:
            sheet_name = rows[0].get("_worksheet")
        row_to_add = {
            header: change.after.get(header, "") if change.after else ""
            for header in self._headers
        }
        if sheet_name:
            row_to_add["_worksheet"] = sheet_name
        elif not row_to_add.get("_worksheet"):
            row_to_add["_worksheet"] = self.worksheet_name or "CSV"
        rows.append(row_to_add)
        self._write_rows(rows, sheet_name)

    def _apply_delete(self, change: PendingChange) -> None:
        sheet_name = change.worksheet or (change.before or {}).get("_worksheet")
        rows = self._read_all_rows(worksheet=sheet_name)
        if not sheet_name and rows:
            sheet_name = rows[0].get("_worksheet")
        remaining = [row for row in rows if row.get("ID") != change.record_id]
        if len(remaining) == len(rows):
            raise ValueError(f"Запись с ID {change.record_id} не найдена.")
        self._write_rows(remaining, sheet_name)

    # ------------------------------------------------------------------ #
    # CSV helpers                                                        #
    # ------------------------------------------------------------------ #

    def _load_headers(self) -> None:
        """Load column headers from Google Sheets or CSV."""
        headers: List[str] = []

        if self._sheets:
            ordered_names = self._worksheet_names or list(self._sheets.keys())
            for name in ordered_names:
                sheet_obj = self._sheets.get(name)
                if sheet_obj is None:
                    continue
                try:
                    headers = sheet_obj.row_values(1)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to read header row from worksheet '%s': %s", name, exc)
                    headers = []
                if headers:
                    break

        if (not headers) and self.csv_path and self.csv_path.exists():
            try:
                with self.csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                    reader = csv.reader(handle)
                    headers = next(reader, [])
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to read materials CSV headers: %s", exc)

        if headers:
            self._set_headers(headers)
        else:
            self._headers = []
            self._header_lookup = {}
            logger.warning(
                "Materials DB headers not detected. Ensure the first row contains column names."
            )

    def _set_headers(self, headers: List[str]) -> None:
        """Configure header list and lookup map."""
        processed: List[str] = []
        seen_keys: set[str] = set()

        for raw in headers:
            text = raw if isinstance(raw, str) else str(raw or "")
            clean = text.strip()
            if not clean:
                continue
            norm = _normalize_key(clean)
            dup_key = norm or clean.lower()
            if dup_key in seen_keys:
                continue
            seen_keys.add(dup_key)
            processed.append(text)

        if not processed:
            self._headers = []
            self._header_lookup = {}
            return

        self._headers = processed
        lookup: Dict[str, str] = {}
        for header in processed:
            normalized = _normalize_key(header)
            if normalized:
                lookup.setdefault(normalized, header)
            trimmed_norm = _normalize_key(header.strip())
            if trimmed_norm:
                lookup.setdefault(trimmed_norm, header)

        for alias_key, target in _FIELD_ALIASES.items():
            target_norm = _normalize_key(target)
            if target_norm in lookup:
                lookup.setdefault(alias_key, lookup[target_norm])

        self._header_lookup = lookup

    def _read_rows_for_sheet(
        self,
        sheet_hint: Optional[str],
        *,
        limit: Optional[int] = None,
        query: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        query_norm = query.lower() if query else None
        resolved_name, sheet_obj = self._resolve_sheet_object(sheet_hint)

        if sheet_obj is None:
            if self.sheet_id:
                self._last_row_fetch_failed = True
            return rows

        try:
            values = sheet_obj.get_all_values()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to read worksheet '%s': %s", resolved_name, exc)
            self._last_row_fetch_failed = True
            return rows

        if not values:
            return rows

        header_row = values[0]
        if header_row and header_row != self._headers:
            self._set_headers(header_row)

        for data_row in values[1:]:
            row_map: Dict[str, str] = {}
            for idx, header in enumerate(self._headers):
                value = data_row[idx] if idx < len(data_row) else ""
                row_map[header] = (value or "").strip()
            row_map["_worksheet"] = sheet_obj.title
            if query_norm:
                searchable = " ".join(row_map.get(header, "") for header in self._headers).lower()
                if query_norm not in searchable:
                    continue
            rows.append(row_map)
            if limit and len(rows) >= limit:
                break

        return rows

    def _read_rows_from_csv(
        self,
        *,
        limit: Optional[int] = None,
        query: Optional[str] = None,
        worksheet: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        if not self.csv_path or not self.csv_path.exists():
            return rows

        query_norm = query.lower() if query else None
        try:
            with self.csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames and reader.fieldnames != self._headers:
                    self._set_headers(reader.fieldnames)
                for row in reader:
                    normalized_row = {
                        header: (row.get(header) or "").strip()
                        for header in self._headers
                    }
                    if query_norm:
                        search_text = " ".join(normalized_row.values()).lower()
                        if query_norm not in search_text:
                            continue
                    normalized_row["_worksheet"] = (
                        worksheet
                        or self.worksheet_name
                        or (self._worksheet_names[0] if self._worksheet_names else "CSV")
                    )
                    rows.append(normalized_row)
                    if limit and len(rows) >= limit:
                        break
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to read materials CSV: %s", exc)
        return rows

    def _read_all_rows(
        self,
        limit: Optional[int] = None,
        query: Optional[str] = None,
        worksheet: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        self._last_row_fetch_failed = False
        if worksheet:
            rows = self._read_rows_for_sheet(worksheet, limit=limit, query=query)
            if rows:
                return rows[:limit] if limit else rows
            rows = self._read_rows_from_csv(limit=limit, query=query, worksheet=worksheet)
            if rows:
                return rows[:limit] if limit else rows
            return []

        rows: List[Dict[str, str]] = []
        remaining = limit
        ordered_names = self._worksheet_names or list(self._sheets.keys())
        for name in ordered_names:
            sheet_rows = self._read_rows_for_sheet(name, limit=remaining, query=query)
            rows.extend(sheet_rows)
            if limit:
                remaining = max(limit - len(rows), 0)
                if remaining <= 0:
                    return rows[:limit]

        if not rows:
            rows = self._read_rows_from_csv(limit=limit, query=query)

        return rows[:limit] if limit else rows

    def _write_rows(self, rows: List[Dict[str, str]], sheet_name: Optional[str]) -> None:
        if not self._headers:
            raise RuntimeError("Materials database headers are not configured.")

        resolved_name, sheet_obj = self._resolve_sheet_object(sheet_name)
        effective_name = sheet_obj.title if sheet_obj is not None else resolved_name

        clean_rows: List[Dict[str, str]] = []
        for row in rows:
            clean_rows.append({header: row.get(header, "") for header in self._headers})

        if sheet_obj is not None:
            values = [self._headers]
            for clean_row in clean_rows:
                values.append([clean_row.get(header, "") for header in self._headers])
            try:
                sheet_obj.update(
                    "A1",
                    values,
                    value_input_option="USER_ENTERED",
                )
                try:
                    desired_rows = max(len(values), 1)
                    desired_cols = max(len(self._headers), 1)
                    sheet_obj.resize(rows=desired_rows, cols=desired_cols)
                except Exception:  # noqa: BLE001
                    logger.debug("Unable to resize Google Sheet after update.", exc_info=True)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to write materials Google Sheet: %s", exc)
                raise

            if effective_name:
                for row in rows:
                    row["_worksheet"] = effective_name

        write_csv = (
            self.csv_path
            and self._sync_csv
            and (not self._worksheet_names or len(self._worksheet_names) <= 1)
        )
        if write_csv:
            self._write_csv_file(clean_rows, effective_name)

    def _write_csv_file(self, rows: List[Dict[str, str]], sheet_name: Optional[str]) -> None:
        """Persist rows to the CSV mirror if configured."""
        if not self.csv_path:
            return
        try:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:  # noqa: BLE001
            pass
        try:
            with self.csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self._headers)
                writer.writeheader()
                for row in rows:
                    payload = {header: row.get(header, "") for header in self._headers}
                    writer.writerow(payload)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to write materials CSV: %s", exc)
            raise

    def _find_row_by_id(self, record_id: str) -> Optional[Dict[str, str]]:
        for row in self._read_all_rows():
            if row.get("ID") == record_id:
                return row
        return None

    def _diff_records(
        self,
        before: Dict[str, str],
        after: Dict[str, str],
    ) -> List[str]:
        changes = []
        for header in self._headers:
            old = (before.get(header) or "").strip()
            new = (after.get(header) or "").strip()
            if old == new:
                continue
            changes.append(f"- {header}: '{old}' → '{new}'")
        return changes

    def _summarize_change(self, change: PendingChange) -> str:
        action = change.action
        target = change.record_id or "без ID"
        reason = change.reason or ""

        if action == "update":
            diff = self._diff_records(change.before or {}, change.after or {})
            desc = "; ".join(diff[:4]) if diff else "без изменений"
            if len(diff) > 4:
                desc += f" … (+{len(diff) - 4} еще)"
            return (
                f"обновление {target} ({desc}){self._sheet_suffix(change.worksheet)}"
                f"{self._reason_suffix(reason)}"
            )
        if action == "insert":
            title = (change.after or {}).get("Название RU ", (change.after or {}).get("Nome ", ""))
            return (
                f"добавление {target} / {title}{self._sheet_suffix(change.worksheet)}"
                f"{self._reason_suffix(reason)}"
            )
        if action == "delete":
            title = (change.before or {}).get("Название RU ", (change.before or {}).get("Nome ", ""))
            return (
                f"удаление {target} / {title}{self._sheet_suffix(change.worksheet)}"
                f"{self._reason_suffix(reason)}"
            )
        return f"{action} {target}{self._sheet_suffix(change.worksheet)}{self._reason_suffix(reason)}"

    @staticmethod
    def _reason_suffix(reason: str) -> str:
        return f" — {reason}" if reason else ""

    @staticmethod
    def _sheet_suffix(sheet_name: Optional[str]) -> str:
        return f" [лист: {sheet_name}]" if sheet_name else ""

    @property
    def worksheet_names(self) -> List[str]:
        return list(self._worksheet_names)

    def _refresh_local_db(self) -> None:
        if self.local_db:
            try:
                self.local_db.refresh()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to refresh local materials DB: %s", exc)

    def _generate_change_id(self) -> str:
        self._pending_counter += 1
        return f"CHG-{self._pending_counter:04d}"
