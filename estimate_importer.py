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
        if column_map is not None and not isinstance(column_map, dict):
            try:
                import pandas as pd  # type: ignore

                if isinstance(column_map, pd.Series):
                    column_map = column_map.to_dict()
                else:
                    column_map = dict(column_map)
            except Exception:
                column_map = None
        try:
            mapping = self._build_mapping(list(df.columns), df=df, column_map=column_map)
        except Exception as exc:
            logger.warning("Failed to build column mapping, using fallback: %s", exc)
            first_col = str(df.columns[0]) if len(df.columns) else "Описание"
            mapping = {"description": first_col}
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

    def import_from_rows(
        self,
        headers: List[str],
        rows: List[List[Any]],
        column_map: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
        description: str = "",
        client_name: str = "",
        currency: str = "€",
        default_item_type: ItemType | str = ItemType.WORK,
    ) -> Estimate:
        """Импорт из простой матрицы (без pandas), используется как запасной путь."""
        # Нормализуем заголовки
        norm_headers = [str(h or "").strip() for h in headers]
        mapping: Dict[str, str] = {}

        def find_by_alias(aliases: Iterable[str]) -> Optional[str]:
            for h in norm_headers:
                low = h.lower()
                if any(alias in low for alias in aliases):
                    return h
            return None

        mapping["description"] = (
            (column_map or {}).get("description")
            or find_by_alias(self.COLUMN_ALIASES["description"])
            or (norm_headers[0] if norm_headers else "Описание")
        )
        for field in ("unit", "quantity", "unit_price", "total_price", "code"):
            val = (column_map or {}).get(field) or find_by_alias(self.COLUMN_ALIASES.get(field, []))
            if val:
                mapping[field] = val

        header_index = {h: idx for idx, h in enumerate(norm_headers)}
        items: List[EstimateItem] = []
        for row in rows:
            desc_val = row[header_index.get(mapping["description"], 0)] if row else ""
            if desc_val is None or str(desc_val).strip() == "":
                continue

            quantity = self._to_float(row[header_index.get(mapping.get("quantity", ""), -1)] if row else None)
            unit_price = self._to_float(row[header_index.get(mapping.get("unit_price", ""), -1)] if row else None)
            total_price = self._to_float(row[header_index.get(mapping.get("total_price", ""), -1)] if row else None)
            if not total_price and quantity and unit_price:
                total_price = round(quantity * unit_price, 2)

            unit = ""
            if "unit" in mapping:
                unit = str(row[header_index.get(mapping["unit"], -1)] if row else "").strip()
            code = ""
            if "code" in mapping:
                code = str(row[header_index.get(mapping["code"], -1)] if row else "").strip()

            item = EstimateItem(
                id=str(uuid.uuid4()),
                name=str(desc_val).strip(),
                description=str(desc_val).strip(),
                item_type=default_item_type,
                quantity=quantity,
                unit=unit or "шт",
                unit_price=unit_price,
                total_price=total_price,
                code=code,
            )
            items.append(item)

        estimate_name = name or "Estimate Import"
        estimate = self.constructor.create_estimate(
            name=estimate_name,
            description=description,
            client_name=client_name,
            currency=currency,
        )
        self.constructor.add_items_to_estimate(estimate.metadata.id, items)
        logger.info("Imported estimate via rows fallback (items=%s, name=%s)", len(items), estimate_name)
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

        return self._normalize_headers_and_rows(df)

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
        return self._normalize_headers_and_rows(df)

    def _normalize_headers_and_rows(self, df):
        """Сдвинуть заголовок на первую непустую строку и привести названия колонок."""
        # Найдём первую строку, где есть непустые значения
        header_idx = None
        for idx, row in df.iterrows():
            row_values = [v for v in row.values if v is not None]
            has_value = any(str(v).strip() != "" for v in row_values)
            if has_value:
                header_idx = idx
                break

        if header_idx is None:
            df = df.dropna(how="all")
            df.columns = [str(col).strip() for col in df.columns]
            return df

        header = [str(col).strip() for col in df.iloc[header_idx].tolist()]
        data_rows = df.iloc[header_idx + 1:].copy()
        data_rows = data_rows.reset_index(drop=True)
        data_rows.columns = header
        data_rows = data_rows.dropna(how="all")
        data_rows.columns = [str(col).strip() for col in data_rows.columns]
        return data_rows

    def _normalize(self, value: Any) -> str:
        return str(value).strip().lower()

    def _build_mapping(self, columns: List[str], column_map: Optional[Dict[str, str]], df=None) -> Dict[str, str]:
        # Приведём карту в словарь, чтобы избежать truthiness-проверок pandas.Series
        if column_map is not None and not isinstance(column_map, dict):
            try:
                import pandas as pd  # type: ignore

                if isinstance(column_map, pd.Series):
                    column_map = column_map.to_dict()
                else:
                    column_map = dict(column_map)
            except Exception:
                column_map = None

        mapping: Dict[str, str] = {}
        normalized = {col: self._normalize(col) for col in columns}

        # Явная карта пользователя имеет приоритет
        if column_map is not None:
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
                # финальный fallback: первая доступная колонка
                for col in columns:
                    if col not in (column_map or {}).values():
                        mapping["description"] = col
                        break

        return mapping

    def _guess_description_column(self, columns: List[str], column_map: Optional[Dict[str, str]] = None, df=None) -> Optional[str]:
        """Эвристика: выбрать самую «текстовую» колонку, если описание не найдено по заголовку."""
        excluded = set((column_map or {}).values())
        # Без DataFrame или без данных выбираем первый не исключенный столбец
        if df is None or getattr(df, "empty", False):
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
        # Если пришла серия/список — берём первое непустое значение
        try:
            import pandas as pd  # type: ignore
            if isinstance(value, pd.Series):
                for v in value:
                    if v is None or str(v).strip() == "":
                        continue
                    return self._to_float(v)
                return 0.0
        except Exception:
            pass
        if isinstance(value, (list, tuple)):
            for v in value:
                if v is None or str(v).strip() == "":
                    continue
                return self._to_float(v)
            return 0.0

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
            try:
                import pandas as pd  # type: ignore
                if isinstance(desc_raw, pd.Series):
                    desc_raw = next((v for v in desc_raw if str(v).strip()), "")
            except Exception:
                pass
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
