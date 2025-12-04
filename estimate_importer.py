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
        return self.import_from_dataframe(
            df=df,
            column_map=column_map,
            name=name or Path(path).stem,
            description=description,
            client_name=client_name,
            currency=currency,
            default_item_type=default_item_type,
        )

    def import_from_dataframe(
        self,
        df,
        column_map: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
        description: str = "",
        client_name: str = "",
        currency: str = "€",
        default_item_type: ItemType | str = ItemType.WORK,
    ) -> Estimate:
        """Импорт из DataFrame (Excel/Google Sheets)."""
        mapping = self._build_mapping(list(df.columns), df=df, column_map=column_map)
        items = self._rows_to_items(df, mapping, default_item_type=default_item_type)

        estimate_name = name or "Estimate Import"
        estimate = self.constructor.create_estimate(
            name=estimate_name,
            description=description,
            client_name=client_name,
            currency=currency,
        )
        self.constructor.add_items_to_estimate(estimate.metadata.id, items)
        logger.info("Imported estimate (items=%s, name=%s)", len(items), estimate_name)
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

    def dataframe_from_values(self, headers: List[str], rows: List[List[Any]]):
        """Построить DataFrame из "сырых" строк (например, Google Sheets)."""
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pandas не установлен, импорт невозможен") from exc

        max_cols = max(len(headers), max((len(r) for r in rows), default=0))
        if len(headers) < max_cols:
            headers = headers + [f"col_{idx+1}" for idx in range(len(headers), max_cols)]
        padded_rows: List[List[Any]] = []
        for r in rows:
            current = list(r)
            if len(current) < max_cols:
                current += [""] * (max_cols - len(current))
            elif len(current) > max_cols:
                current = current[:max_cols]
            padded_rows.append(current)
        df = pd.DataFrame(padded_rows, columns=headers)
        df = df.dropna(how="all").copy()
        df.columns = [str(col).strip() for col in df.columns]
        return df

    def _normalize(self, value: Any) -> str:
        return str(value).strip().lower()

    def _build_mapping(self, columns: List[str], column_map: Optional[Dict[str, str]], df=None) -> Dict[str, str]:
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
            guessed = self._guess_description_column(columns, column_map, df=df)
            if guessed:
                mapping["description"] = guessed
            else:
                raise ValueError("Не найдена колонка с описанием (description). Добавьте column_map.")

        return mapping

    def _guess_description_column(self, columns: List[str], column_map: Optional[Dict[str, str]] = None, df=None) -> Optional[str]:
        """Эвристика: выбрать самую «текстовую» колонку, если описание не найдено по заголовку."""
        excluded = set((column_map or {}).values())
        # Без DataFrame выбираем первый не исключенный столбец
        if df is None:
            for col in columns:
                if col not in excluded:
                    return col
            return None

        best_col = None
        best_score = -1
        for col in columns:
            if col in excluded:
                continue
            series = df[col]
            text_like = 0
            unique_nonempty = set()
            for val in series:
                if val is None:
                    continue
                sval = str(val).strip()
                if not sval:
                    continue
                unique_nonempty.add(sval)
                # числовые значения не считаем текстом
                try:
                    float(sval.replace(",", "."))
                    continue
                except Exception:
                    pass
                # короткие коды тоже пропустим
                if len(sval) <= 2:
                    continue
                text_like += 1
            # Если нет явного текста, используем разнообразие значений
            score = text_like if text_like > 0 else len(unique_nonempty)
            if score > best_score:
                best_score = score
                best_col = col
        return best_col

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
