"""
Excel → Estimate converter.

Переносит произвольную Excel-смету в внутренний формат `EstimateConstructor`
с попыткой авто-распознавания колонок (описание, ед., количество, цена, итого, код).
"""

from __future__ import annotations

import logging
import re
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
        column_map = self._coerce_column_map(column_map)
        try:
            mapping = self._build_mapping(list(df.columns), df=df, column_map=column_map)
        except Exception as exc:
            logger.warning("Failed to build column mapping, using fallback: %s", exc)
            first_col = str(df.columns[0]) if len(df.columns) else "Описание"
            mapping = {"description": first_col}
        try:
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
        except ValueError as exc:
            # Частые ошибки pandas "truth value of a Series is ambiguous" — уходим в безопасный режим
            msg = str(exc).lower()
            if "ambiguous" in msg or "truth value of a series" in msg:
                logger.warning("DataFrame import hit ambiguous Series; retrying via rows fallback: %s", exc)
                try:
                    headers = [str(h) for h in df.columns]
                    safe_rows = df.fillna("").values.tolist()
                    return self.import_from_rows(
                        headers=headers,
                        rows=safe_rows,
                        column_map=column_map,
                        name=name,
                        description=description,
                        client_name=client_name,
                        currency=currency,
                        default_item_type=default_item_type,
                    )
                except Exception as fallback_exc:
                    logger.warning("Rows fallback failed after ambiguous Series: %s", fallback_exc)
                    raise fallback_exc
            raise

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
        column_map = self._coerce_column_map(column_map)
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
        """Сдвинуть заголовок на первую непустую строку и привести названия колонок.

        Если в DataFrame уже есть внятные заголовки (не только Unnamed/числа/col_N),
        не переопределяем их первой строкой с данными — это важно для Google Sheets,
        где заголовок уже передан отдельно (иначе «Описание» превращается в номера).
        """
        norm_columns = [str(col).strip() for col in df.columns]

        def _is_placeholder(name: str, idx: int) -> bool:
            if not name or name.lower() in {"nan", "none"}:
                return True
            low = name.lower()
            if low.startswith("unnamed"):
                return True
            if low.startswith("col_"):
                return True
            if name == str(idx):
                return True
            try:
                float(name.replace(",", "."))
                return True
            except Exception:
                return False

        placeholder_count = sum(_is_placeholder(name, idx) for idx, name in enumerate(norm_columns))
        meaningful_headers = len(norm_columns) - placeholder_count

        # Если большинство колонок уже названы осмысленно — оставляем их как есть
        if meaningful_headers >= max(1, len(norm_columns) // 2):
            cleaned = df.dropna(how="all").copy()
            cleaned.columns = norm_columns
            cleaned.columns = [str(col).strip() for col in cleaned.columns]
            return cleaned.reset_index(drop=True)

        # Иначе ищем строку-заголовок внутри данных (старое поведение)
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

    def _coerce_column_map(self, column_map: Any) -> Optional[Dict[str, str]]:
        """Привести карту колонок к словарю str->str, избегая pandas Series/DataFrame."""
        if column_map is None:
            return None
        # Пробуем быстрый путь: уже dict
        if isinstance(column_map, dict):
            cleaned = {}
            for key, val in column_map.items():
                if val is None:
                    continue
                sval = str(val).strip()
                if not sval:
                    continue
                cleaned[str(key).strip()] = sval
            return cleaned or None
        try:
            import pandas as pd  # type: ignore

            if isinstance(column_map, pd.Series):
                return self._coerce_column_map(column_map.to_dict())
            if isinstance(column_map, pd.DataFrame):
                if len(column_map) >= 1:
                    return self._coerce_column_map(column_map.iloc[0].to_dict())
                return None
        except Exception:
            pass

        try:
            tentative = dict(column_map)
            return self._coerce_column_map(tentative)
        except Exception:
            return None

    def _build_mapping(self, columns: List[str], column_map: Optional[Dict[str, str]], df=None) -> Dict[str, str]:
        column_map = self._coerce_column_map(column_map)

        mapping: Dict[str, str] = {}
        normalized = {col: self._normalize(col) for col in columns}

        # Явная карта пользователя имеет приоритет
        if column_map:
            for key, col in column_map.items():
                col_name = str(col).strip()
                if col_name in columns:
                    mapping[key] = col_name
                else:
                    # пробуем по нормализованному варианту
                    match = next((orig for orig, norm in normalized.items() if norm == self._normalize(col_name)), None)
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

        # Подсказка количества по данным (если не нашли по названию)
        if "quantity" not in mapping and df is not None:
            candidates: list[tuple[int, int, str]] = []
            for idx, col in enumerate(columns):
                if col in mapping.values():
                    continue
                norm = normalized.get(col, "")
                if any(key in norm for key in ("id", "master", "код", "code", "ref", "art")):
                    continue
                series = df[col]
                numeric_count = 0
                non_empty = 0
                for val in series:
                    if val is None or str(val).strip() == "":
                        continue
                    non_empty += 1
                    sval = str(val)
                    try:
                        float(
                            sval.replace("€", "")
                            .replace(" ", "")
                            .replace("\u00a0", "")
                            .replace(",", ".")
                        )
                        numeric_count += 1
                    except Exception:
                        continue
                if non_empty and numeric_count >= max(1, int(non_empty * 0.6)):
                    # При равенстве количества чисел берем самый левый столбец
                    candidates.append((numeric_count, -idx, col))
            if candidates:
                candidates.sort(reverse=True)
                mapping["quantity"] = candidates[0][2]

        # Подсказка номера позиции (№) по данным, если не нашли по заголовку
        if "number" not in mapping and df is not None:
            candidates_num: list[tuple[int, int, str]] = []
            for idx, col in enumerate(columns):
                if col in mapping.values():
                    continue
                series = df[col]
                score = 0
                non_empty = 0
                for val in series:
                    if val is None or str(val).strip() == "":
                        continue
                    sval = str(val).strip()
                    if any(sym in sval for sym in ("€", "$", "руб", "eur", "usd")):
                        continue
                    if len(sval) > 12:
                        continue
                    if re.search(r"[A-Za-z]", sval):
                        continue
                    non_empty += 1
                    if re.fullmatch(r"\d+(\.\d+)+", sval):
                        score += 3  # иерархическая нумерация 1.1.2
                    elif re.fullmatch(r"\d+[.,]\d+", sval):
                        score += 1  # дробные числа, например 1,1
                    elif re.fullmatch(r"\d+", sval) and len(sval) <= 4:
                        score += 1  # короткие целые
                if score and non_empty:
                    candidates_num.append((score, -idx, col))
            if candidates_num:
                candidates_num.sort(reverse=True)
                mapping["number"] = candidates_num[0][2]

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
        best_score = (-1, -1)  # (has_text_flag, score_value)
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
            # Предпочитаем колонки с текстом; при равенстве выигрывает та, где есть текст
            has_text = int(text_like > 0)
            score = (has_text, text_like if has_text else len(unique_nonempty))
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
