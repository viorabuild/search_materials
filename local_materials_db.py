"""Local material database helpers for quick lookups.

This module loads the CSV provided by the user and exposes fuzzy/exact
search utilities so that the unified agent can augment remote search results
with locally curated suggestions.
"""

from __future__ import annotations

import csv
import logging
import os
import re
import threading
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Iterable, List, Optional

try:  # pragma: no cover - optional dependency
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
except Exception:  # noqa: BLE001 - best effort optional import
    Observer = None
    FileSystemEventHandler = object  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _norm(text: Optional[str]) -> str:
    """Normalize text for comparisons."""
    if not text:
        return ""
    return text.strip().lower()


def _extract_first_url(*values: Optional[str]) -> Optional[str]:
    """Extract the first URL from the provided text snippets."""
    pattern = re.compile(r"https?://[\w\-./%#?=&:+]+", re.IGNORECASE)
    for value in values:
        if not value:
            continue
        match = pattern.search(value)
        if match:
            return match.group(0)
    return None


@dataclass
class LocalMaterialMatch:
    """Structured result returned by the local materials database."""

    display_name: str
    supplier: str
    price: str
    url: str
    notes: str
    score: float


@dataclass
class _LocalMaterialEntry:
    """Internal representation of a single CSV row."""

    names: List[tuple[str, str]]  # (display, normalized)
    category: str
    group: str
    unit: str
    supplier: str
    price: str
    url: str
    notes: str

    def match_score(self, query: str) -> float:
        """Compute how well this entry matches the query (0-1 range)."""
        if not query:
            return 0.0

        best = 0.0
        for _, normalized in self.names:
            if not normalized:
                continue
            if query == normalized:
                return 1.0
            if normalized.startswith(query):
                best = max(best, 0.9)
            if query in normalized:
                best = max(best, 0.82)
            ratio = SequenceMatcher(None, query, normalized).ratio()
            if ratio > 0.6:
                best = max(best, ratio * 0.85)

        cat = _norm(self.category)
        grp = _norm(self.group)
        if cat and query in cat:
            best = max(best, 0.75)
        if grp and query in grp:
            best = max(best, 0.7)

        return best

    def to_match(self, score: float) -> LocalMaterialMatch:
        """Convert to an outward-facing match."""
        primary_name = next((display for display, _ in self.names if display), "")
        display_name = primary_name or self.category or "Материал"
        notes = self.notes
        if self.category and self.category.lower() not in notes.lower():
            notes = f"{notes} | Категория: {self.category}".strip()

        if self.unit and self.unit.lower() not in notes.lower():
            notes = f"{notes} | Ед.: {self.unit}".strip()

        return LocalMaterialMatch(
            display_name=display_name,
            supplier=self.supplier or "Локальная база",
            price=self.price or "N/A",
            url=self.url or "N/A",
            notes=notes.strip() or "Локальная база материалов",
            score=score,
        )


class LocalMaterialDatabase:
    """Loads and searches a CSV with locally curated materials."""

    def __init__(self, csv_path: Path, *, auto_refresh: Optional[bool] = None) -> None:
        self._entries: List[_LocalMaterialEntry] = []
        self._loaded = False
        self._path = csv_path
        self._last_mtime: Optional[float] = None
        if auto_refresh is None:
            env_value = os.getenv("LOCAL_DB_AUTO_REFRESH", "true").lower()
            auto_refresh = env_value in {"1", "true", "yes", "on"}
        self._auto_refresh = auto_refresh
        self._observer: Optional[Observer] = None
        self._watchdog_handler: Optional[_CSVWatchdogHandler] = None
        debounce_raw = os.getenv("LOCAL_DB_WATCHDOG_DEBOUNCE_SECONDS", "1.0")
        try:
            self._watchdog_debounce = float(debounce_raw)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid LOCAL_DB_WATCHDOG_DEBOUNCE_SECONDS value %r; using 1.0", debounce_raw
            )
            self._watchdog_debounce = 1.0

        self._load()
        if self._should_use_watchdog():  # pragma: no cover - optional runtime path
            self._start_watchdog()

    def _load(self) -> None:
        if not self._path.exists():
            logger.info("Local materials CSV not found at %s", self._path)
            self._entries = []
            self._loaded = False
            self._last_mtime = None
            return

        try:
            current_mtime = self._path.stat().st_mtime
        except OSError as exc:  # pragma: no cover - stat errors are rare
            logger.error("Failed to stat local materials CSV %s: %s", self._path, exc)
            self._entries = []
            self._loaded = False
            self._last_mtime = None
            return

        new_entries: List[_LocalMaterialEntry] = []

        try:
            with self._path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    logger.warning("Local materials CSV at %s has no headers", self._path)
                    self._entries = []
                    self._loaded = False
                    self._last_mtime = current_mtime
                    return

                normalized_headers = [header.strip() for header in reader.fieldnames]

                for raw_row in reader:
                    if not raw_row:
                        continue
                    normalized_row = {
                        normalized_headers[idx]: (value.strip() if isinstance(value, str) else value)
                        for idx, value in enumerate(raw_row.values())
                    }

                    entry = self._build_entry(normalized_row)
                    if entry:
                        new_entries.append(entry)

            self._entries = new_entries
            self._loaded = True
            self._last_mtime = current_mtime
            logger.info(
                "Loaded %d materials from local CSV %s",
                len(self._entries),
                self._path,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load local materials CSV %s: %s", self._path, exc)
            self._entries = []
            self._loaded = False
            self._last_mtime = current_mtime

    def _build_entry(self, row: dict) -> Optional[_LocalMaterialEntry]:
        raw_name_values = [
            row.get("Nome"),
            row.get("Nome "),  # trailing space header support
            row.get("Название RU"),
            row.get("Название RU [AI]"),
        ]

        names: List[tuple[str, str]] = []
        seen_normalized: set[str] = set()
        for raw_value in raw_name_values:
            if raw_value is None:
                continue
            display_value = raw_value.strip()
            normalized_value = _norm(raw_value)
            if not normalized_value or normalized_value in seen_normalized:
                continue
            seen_normalized.add(normalized_value)
            names.append((display_value or raw_value, normalized_value))

        if not names:
            return None

        category = row.get("Категория", "").strip()
        group = row.get("Группа", "").strip()
        unit = row.get("Ед. Измерения", "").strip()

        supplier = (
            row.get("Последний Поставщик")
            or row.get("Поставщик / Подрядчик")
            or row.get("Поставщики")
            or "Локальная база"
        ).strip()

        price = (
            row.get("Стоимость")
            or row.get("Последняя Котировка")
            or row.get("Последняя стоимость")
            or ""
        ).strip()

        raw_urls: Iterable[Optional[str]] = (
            row.get("Поставщики"),
            row.get("💯 Котировки поставщиков"),
            row.get("Последняя Котировка"),
            row.get("Последняя стоимость"),
        )
        url = _extract_first_url(*raw_urls) or ""

        notes_parts = [
            row.get("Название RU") or "",
            row.get("Название RU [AI]") or "",
            row.get("💯 Котировки поставщиков") or "",
        ]
        notes = " | ".join(part.strip() for part in notes_parts if part and part.strip())

        return _LocalMaterialEntry(
            names=names,
            category=category,
            group=group,
            unit=unit,
            supplier=supplier,
            price=price,
            url=url,
            notes=notes,
        )

    @property
    def available(self) -> bool:
        return self._loaded and bool(self._entries)

    def search(
        self,
        query: str,
        *,
        exact_only: bool = False,
        limit: int = 10,
    ) -> List[LocalMaterialMatch]:
        if self._auto_refresh:
            self._ensure_fresh()

        if not self.available:
            return []

        normalized_query = _norm(query)
        if not normalized_query:
            return []

        matches: List[LocalMaterialMatch] = []

        if exact_only:
            for entry in self._entries:
                if any(normalized_query == normalized for _, normalized in entry.names):
                    matches.append(entry.to_match(1.0))
            return matches[:limit]

        scored: List[tuple[float, _LocalMaterialEntry]] = []
        for entry in self._entries:
            score = entry.match_score(normalized_query)
            if score <= 0:
                continue
            scored.append((score, entry))

        scored.sort(key=lambda item: item[0], reverse=True)
        for score, entry in scored[:limit]:
            matches.append(entry.to_match(score))

        return matches

    def refresh(self) -> None:
        """Reload the CSV file to reflect external updates."""
        self._entries.clear()
        self._loaded = False
        self._load()

    def close(self) -> None:  # pragma: no cover - cleanup helper
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=2)
            self._observer = None
        if self._watchdog_handler:
            self._watchdog_handler.shutdown()
            self._watchdog_handler = None

    def _ensure_fresh(self) -> None:
        if not self._path.exists():
            if self._entries:
                logger.info("Local materials CSV at %s removed; clearing cache", self._path)
                self._entries.clear()
            self._loaded = False
            self._last_mtime = None
            return

        try:
            current_mtime = self._path.stat().st_mtime
        except OSError as exc:  # pragma: no cover - stat errors are rare
            logger.debug("Unable to stat %s for auto refresh: %s", self._path, exc)
            return

        if self._last_mtime is None or current_mtime > self._last_mtime:
            logger.info("Detected change in %s; refreshing local DB", self._path)
            self.refresh()

    def _should_use_watchdog(self) -> bool:
        if not self._auto_refresh:
            return False
        use_watchdog = os.getenv("LOCAL_DB_USE_WATCHDOG", "false").lower()
        return Observer is not None and use_watchdog in {"1", "true", "yes", "on"}

    def _start_watchdog(self) -> None:  # pragma: no cover - optional runtime path
        assert Observer is not None  # noqa: S101 - ensured by caller

        handler = _CSVWatchdogHandler(
            target_path=self._path,
            callback=self.refresh,
            debounce_seconds=self._watchdog_debounce,
        )
        observer = Observer()
        observer.schedule(handler, self._path.parent.as_posix(), recursive=False)
        observer.daemon = True
        observer.start()
        self._observer = observer
        self._watchdog_handler = handler


class _CSVWatchdogHandler(FileSystemEventHandler):  # pragma: no cover - optional runtime path
    """Debounced watchdog handler that triggers refreshes."""

    def __init__(
        self,
        *,
        target_path: Path,
        callback: Callable[[], None],
        debounce_seconds: float,
    ) -> None:
        self._target_path = target_path.resolve()
        self._callback = callback
        self._debounce = max(debounce_seconds, 0.0)
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

    def on_modified(self, event) -> None:  # type: ignore[override]
        self._handle_event(event)

    def on_created(self, event) -> None:  # type: ignore[override]
        self._handle_event(event)

    def on_moved(self, event) -> None:  # type: ignore[override]
        self._handle_event(event)

    def shutdown(self) -> None:
        with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None

    def _handle_event(self, event) -> None:
        try:
            is_directory = bool(getattr(event, "is_directory", False))
            src_path = Path(getattr(event, "src_path", ""))
            dest_path = Path(getattr(event, "dest_path", ""))
        except TypeError:  # pragma: no cover - defensive
            return

        if is_directory:
            return

        if src_path.resolve() != self._target_path and dest_path.resolve() != self._target_path:
            return

        with self._lock:
            if self._timer:
                self._timer.cancel()

            if self._debounce <= 0:
                self._callback()
                return

            self._timer = threading.Timer(self._debounce, self._callback)
            self._timer.daemon = True
            self._timer.start()
