"""
Excel → Estimate converter.

Переносит произвольную Excel-смету в внутренний формат `EstimateConstructor`
с попыткой авто-распознавания колонок (описание, ед., количество, цена, итого, код).
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from estimate_constructor import Estimate, EstimateConstructor, EstimateItem, ItemType

logger = logging.getLogger(__name__)


class ExcelEstimateImporter:
    """Импорт смет из Excel в формат EstimateConstructor."""

    COLUMN_ALIASES: Dict[str, Iterable[str]] = {
        "number": ("#", "no", "n", "№", "item", "linha", "linha n"),
        "description": (
            "описание",
            "описание работ",
            "description",
            "descrição",
            "descricao",
            "scope",
            "item description",
        ),
        "unit": ("ед", "ед.", "unit", "unid", "unidade", "measure", "uom", "единица"),
        "quantity": ("кол", "кол-во", "qty", "quantity", "quantidade", "qtd"),
        "unit_price": (
            "цена",
            "цена за ед",
            "unit price",
            "preço",
            "preço por un",
            "preco",
            "preco por un",
            "price",
        ),
        "total_price": ("итого", "сумма", "total", "valor total", "total price"),
        "code": ("код", "code", "ref", "reference", "art", "article"),
    }

    def __init__(self, constructor: EstimateConstructor):
        self.constructor = constructor

    def import_from_excel(
        self,
        path: Path | str,
        sheet_name: Optional[str] = None,
        column_map: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
        description: str = "",
        client_name: str = "",
        currency: str = "€",
        skip_rows: int = 0,
        default_item_type: ItemType | str = ItemType.WORK,
    ) -> Estimate:
        """
        Прочитать Excel, сконвертировать и сохранить смету в EstimateConstructor.

        Args:
            path: путь к .xlsx/.xls файлу.
            sheet_name: имя листа (если None — первый лист).
            column_map: явное сопоставление полей {"description": "Описание", "quantity": "Qty"}.
            name: имя сметы (по умолчанию — имя файла).
            description: описание проекта.
            client_name: имя клиента.
            currency: валюта сметы.
            skip_rows: сколько верхних строк пропустить (логотипы/шапки).
            default_item_type: тип позиций (work/material/...).
        """
        df = self._load_dataframe(path, sheet_name=sheet_name, skip_rows=skip_rows)
        mapping = self._build_mapping(list(df.columns), column_map=column_map)
        items = self._rows_to_items(df, mapping, default_item_type=default_item_type)

        estimate_name = name or Path(path).stem
        estimate = self.constructor.create_estimate(
            name=estimate_name,
            description=description,
            client_name=client_name,
            currency=currency,
        )
        self.constructor.add_items_to_estimate(estimate.metadata.id, items)
        logger.info(
            "Imported estimate from %s (items=%s, sheet=%s)",
            path,
            len(items),
            sheet_name or "first",
        )
        return estimate

    def _load_dataframe(self, path: Path | str, sheet_name: Optional[str], skip_rows: int):
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                "pandas не установлен. Добавьте pandas/openpyxl в зависимости, чтобы импортировать Excel."
            ) from exc

        excel = pd.read_excel(path, sheet_name=sheet_name, skiprows=skip_rows)
        # Если workbook содержит несколько листов — берём первый словарь
        if isinstance(excel, dict):
            df = next(iter(excel.values()))
        else:
            df = excel

        df = df.dropna(how="all").copy()
        df.columns = [str(col).strip() for col in df.columns]
        return df

    def _normalize(self, value: Any) -> str:
        return str(value).strip().lower()

    def _build_mapping(self, columns: List[str], column_map: Optional[Dict[str, str]]) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        normalized = {col: self._normalize(col) for col in columns}

        # Явная карта пользователя имеет приоритет
        if column_map:
            for key, col in column_map.items():
                if col in columns:
                    mapping[key] = col
                else:
                    # пробуем по нормализованному варианту
                    match = next((orig for orig, norm in normalized.items() if norm == self._normalize(col)), None)
                    if match:
                        mapping[key] = match

        # Автоматический подбор для отсутствующих ключей
        for field, aliases in self.COLUMN_ALIASES.items():
            if field in mapping:
                continue
            match = next(
                (col for col, norm in normalized.items() if any(alias in norm for alias in aliases)),
                None,
            )
            if match:
                mapping[field] = match

        if "description" not in mapping:
            raise ValueError("Не найдена колонка с описанием (description). Добавьте column_map.")

        return mapping

    def _to_float(self, value: Any) -> float:
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        try:
            cleaned = (
                str(value)
                .replace("€", "")
                .replace(" ", "")
                .replace("\u00a0", "")
                .replace(",", ".")
            )
            return float(cleaned)
        except Exception:
            return 0.0

    def _rows_to_items(
        self,
        df,
        mapping: Dict[str, str],
        default_item_type: ItemType | str,
    ) -> List[EstimateItem]:
        items: List[EstimateItem] = []
        for _, row in df.iterrows():
            desc_raw = row.get(mapping["description"])
            if desc_raw is None or str(desc_raw).strip() == "":
                continue

            quantity = self._to_float(row.get(mapping.get("quantity", ""), 0.0))
            unit_price = self._to_float(row.get(mapping.get("unit_price", ""), 0.0))
            total_price = self._to_float(row.get(mapping.get("total_price", ""), 0.0))
            if not total_price and quantity and unit_price:
                total_price = round(quantity * unit_price, 2)

            item = EstimateItem(
                id=str(uuid.uuid4()),
                name=str(desc_raw).strip(),
                description=str(desc_raw).strip(),
                item_type=default_item_type,
                quantity=quantity,
                unit=str(row.get(mapping.get("unit", ""), "") or "шт").strip(),
                unit_price=unit_price,
                total_price=total_price,
                code=str(row.get(mapping.get("code", ""), "") or "").strip(),
            )
            items.append(item)
        return items
