"""Processing helper for zipped estimate packages.

Раскрывает архив, раскладывает файлы по типам, анализирует их через локальную
LLM и добавляет текстовые заметки рядом с исходниками.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI

from llm_provider import _normalize_base_url

logger = logging.getLogger(__name__)


TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".csv",
    ".tsv",
    ".json",
    ".yaml",
    ".yml",
    ".xml",
}
TABLE_EXTENSIONS = {".xls", ".xlsx"}
CAD_EXTENSIONS = {".dwg", ".dxf", ".dfwx", ".dwfx", ".rvt"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
ARCHIVE_EXTENSIONS = {".zip", ".rar", ".7z"}


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


class EstimatePackageProcessor:
    """Обработка архивов со сметными материалами."""

    def __init__(
        self,
        storage_dir: Path,
        local_model: str = "qwen/qwen3-vl-4b",
        local_base_url: Optional[str] = None,
        local_api_key: str = "lm-studio",
        request_timeout: Optional[float] = None,
    ) -> None:
        self.storage_dir = _ensure_directory(Path(storage_dir))
        self.uploads_dir = _ensure_directory(self.storage_dir / "uploads")
        self.local_model = local_model or "qwen/qwen3-vl-4b"
        self._client = self._build_local_client(
            base_url=local_base_url,
            api_key=local_api_key,
            timeout=request_timeout,
        )

    def _build_local_client(
        self,
        base_url: Optional[str],
        api_key: str,
        timeout: Optional[float],
    ) -> Optional[OpenAI]:
        if not base_url:
            logger.warning(
                "LOCAL_LLM_BASE_URL не задан. Просчёт архива будет без анализа файлов (только распаковка)."
            )
            return None
        try:
            normalized = _normalize_base_url(base_url)
            return OpenAI(api_key=api_key or "lm-studio", base_url=normalized, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Не удалось инициализировать локальную LLM: %s", exc)
            return None

    # ------------------------------------------------------------------ публичные методы
    def process_archive(
        self,
        zip_path: Path,
        reorganize: bool = True,
    ) -> Dict[str, Any]:
        """Распаковать архив, разложить файлы, добавить заметки и вернуть отчёт."""
        if not zip_path.exists():
            raise FileNotFoundError(f"Архив не найден: {zip_path}")
        if not zipfile.is_zipfile(zip_path):
            raise ValueError("Ожидался zip-архив для просчёта смет")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        package_root = _ensure_directory(self.storage_dir / f"{zip_path.stem}_{timestamp}")
        extracted_dir = package_root / "original"
        organized_dir = package_root / "organized"
        _ensure_directory(extracted_dir)
        _ensure_directory(organized_dir)

        self._safe_extract(zip_path, extracted_dir)
        files = self._list_files(extracted_dir)

        if reorganize:
            self._organize_files(files, extracted_dir, organized_dir)

        results: List[Dict[str, Any]] = []
        for file_path in files:
            try:
                summary, snippet = self._analyze_file(file_path, root=extracted_dir)
            except Exception as exc:  # noqa: BLE001
                summary = f"Не удалось проанализировать файл: {exc}"
                snippet = ""
            try:
                note_path = self._write_note(file_path, summary, snippet, root=extracted_dir)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Не удалось записать заметку для %s: %s", file_path, exc)
                note_path = None

            results.append(
                {
                    "file": str(file_path.relative_to(extracted_dir)),
                    "note": str(note_path.relative_to(extracted_dir)) if note_path else None,
                    "size_bytes": file_path.stat().st_size,
                    "category": self._detect_category(file_path),
                }
            )

        report_path = package_root / "package_report.json"
        report_payload = {
            "source_archive": str(zip_path),
            "processed_at": timestamp,
            "model": self.local_model,
            "extracted_dir": str(extracted_dir),
            "organized_dir": str(organized_dir),
            "files_processed": len(results),
            "results": results,
        }
        report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        return report_payload

    def stage_upload(self, filename: str, data: bytes) -> Path:
        """Сохранить загруженный архив в рабочую директорию."""
        safe_name = Path(filename).name or "upload.zip"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = self.uploads_dir / f"{timestamp}__{safe_name}"
        dest.write_bytes(data)
        return dest

    # ------------------------------------------------------------------ вспомогательные
    def _safe_extract(self, zip_path: Path, target_dir: Path) -> None:
        """Распаковать zip без возможности выхода за пределы target_dir."""
        with zipfile.ZipFile(zip_path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                dest_path = target_dir / info.filename
                if not self._is_within_directory(target_dir, dest_path):
                    raise ValueError(f"Запрещённый путь в архиве: {info.filename}")
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(dest_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)

    @staticmethod
    def _is_within_directory(directory: Path, target: Path) -> bool:
        try:
            target.resolve().relative_to(directory.resolve())
            return True
        except ValueError:
            return False

    def _list_files(self, root: Path) -> List[Path]:
        files = []
        for path in root.rglob("*"):
            if path.is_dir() or path.is_symlink():
                continue
            if path.name.endswith("__note.txt"):
                continue
            files.append(path)
        files.sort()
        return files

    def _detect_category(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in TEXT_EXTENSIONS:
            return "text"
        if suffix in TABLE_EXTENSIONS:
            return "table"
        if suffix in CAD_EXTENSIONS:
            return "cad"
        if suffix in IMAGE_EXTENSIONS:
            return "image"
        if suffix in ARCHIVE_EXTENSIONS:
            return "archive"
        return "other"

    def _organize_files(self, files: Iterable[Path], source_root: Path, organized_root: Path) -> None:
        """Создать структуру organized/ с разложенными ссылками на файлы."""
        for file_path in files:
            category = self._detect_category(file_path)
            target_dir = organized_root / category
            target_dir.mkdir(parents=True, exist_ok=True)
            relative_name = file_path.relative_to(source_root)
            target_path = target_dir / relative_name.name
            if target_path.exists():
                continue
            try:
                target_path.symlink_to(file_path)
            except (OSError, NotImplementedError):
                shutil.copy2(file_path, target_path)

    def _read_file_preview(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in TEXT_EXTENSIONS:
            return self._read_text(path, limit=8000)
        if suffix in {".csv", ".tsv"}:
            return self._read_text(path, limit=8000)
        if suffix in TABLE_EXTENSIONS:
            return self._read_excel(path)
        if suffix in {".xml"}:
            return self._read_text(path, limit=6000)
        if suffix in {".dfwx", ".dwfx"}:
            return self._read_dfwx(path)
        if suffix in ARCHIVE_EXTENSIONS:
            return f"Вложенный архив {path.name}. Содержимое не распаковывалось автоматически."
        return ""

    def _read_text(self, path: Path, limit: int = 4000) -> str:
        try:
            data = path.read_text(encoding="utf-8", errors="ignore")
            if len(data) > limit:
                return data[:limit] + "\n... (усечено)"
            return data
        except Exception as exc:  # noqa: BLE001
            return f"Не удалось прочитать текст: {exc}"

    def _read_excel(self, path: Path) -> str:
        try:
            import openpyxl
        except Exception as exc:  # noqa: BLE001
            return f"Не удалось загрузить openpyxl: {exc}"
        try:
            workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
            sheet = workbook.active
            rows: List[List[str]] = []
            for row in sheet.iter_rows(min_row=1, max_row=15, max_col=8):
                rows.append(
                    [
                        "" if cell.value is None else str(cell.value)
                        for cell in row
                    ]
                )
            workbook.close()
            preview_lines = ["; ".join(r).strip() for r in rows if any(cell.strip() for cell in r)]
            return "\n".join(preview_lines)
        except Exception as exc:  # noqa: BLE001
            return f"Не удалось прочитать Excel: {exc}"

    def _read_dfwx(self, path: Path) -> str:
        """DWFx/DFWX — это zip с XML. Читаем первые куски для контекста."""
        try:
            with zipfile.ZipFile(path, "r") as zf:
                snippets: List[str] = []
                for name in zf.namelist():
                    lower = name.lower()
                    if not lower.endswith(".xml"):
                        continue
                    try:
                        with zf.open(name) as fh:
                            data = fh.read(4000).decode("utf-8", errors="ignore")
                            snippets.append(f"### {name}\n{data}")
                    except Exception:
                        continue
                    if len(snippets) >= 3:
                        break
                if not snippets:
                    return "DWFx файл без видимых XML-данных (пустой превью)."
                joined = "\n\n".join(snippets)
                if len(joined) > 7000:
                    return joined[:7000] + "\n... (усечено)"
                return joined
        except Exception as exc:  # noqa: BLE001
            return f"Не удалось раскрыть DFWX/DWFx: {exc}"

    def _analyze_file(self, path: Path, root: Path) -> tuple[str, str]:
        size_bytes = path.stat().st_size
        category = self._detect_category(path)
        preview = self._read_file_preview(path)
        user_payload = [
            f"Файл: {path.name}",
            f"Путь: {path.relative_to(root)}",
            f"Категория: {category}",
            f"Размер: {size_bytes} байт",
            "",
            "Извлечённый фрагмент (может быть усечён):",
            preview or "(нет текста, ориентируйся по расширению и названию)",
        ]
        messages = [
            {
                "role": "system",
                "content": (
                    "Ты инженер-консультант по строительным сметам. "
                    "Кратко опиши, что лежит в файле, как его использовать при просчёте сметы "
                    "и на что обратить внимание. Не придумывай содержимое, если его нет в превью."
                ),
            },
            {"role": "user", "content": "\n".join(user_payload)},
        ]
        if not self._client:
            summary = "Анализ пропущен: локальная модель не настроена. Файл распакован."
            return summary, preview
        try:
            response = self._client.chat.completions.create(
                model=self.local_model,
                messages=messages,
                temperature=0.2,
                max_tokens=320,
            )
            summary = (
                response.choices[0].message.content.strip()
                if response.choices
                else "Не удалось получить ответ от модели."
            )
        except Exception as exc:  # noqa: BLE001
            summary = f"Не удалось обратиться к локальной модели: {exc}"
        return summary, preview

    def _write_note(self, file_path: Path, summary: str, preview: str, root: Path) -> Path:
        note_name = f"{file_path.stem}__note.txt"
        note_path = file_path.with_name(note_name)
        lines = [
            f"Файл: {file_path.name}",
            f"Относительный путь: {file_path.relative_to(root)}",
            f"Описание: {summary}",
            "",
            "Фрагмент содержимого:",
            preview or "(нет текстового превью)",
        ]
        note_path.write_text("\n".join(lines), encoding="utf-8")
        return note_path
