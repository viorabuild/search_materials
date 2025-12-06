from pathlib import Path

import pandas as pd

from estimate_constructor import EstimateConstructor
from estimate_importer import ExcelEstimateImporter


def test_excel_importer_basic(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "Описание работ": ["Демонтаж плитки", "Укладка новой плитки"],
            "Ед.": ["м2", "м2"],
            "Qty": [12, 8],
            "Unit Price": [15, 22.5],
        }
    )
    xlsx_path = tmp_path / "estimate.xlsx"
    df.to_excel(xlsx_path, index=False)

    constructor = EstimateConstructor(
        llm_client=None,
        llm_model="dummy",
        material_agent=None,
        storage_path=tmp_path,
    )
    importer = ExcelEstimateImporter(constructor)

    estimate = importer.import_from_excel(
        path=xlsx_path,
        column_map={
            "description": "Описание работ",
            "unit": "Ед.",
            "quantity": "Qty",
            "unit_price": "Unit Price",
        },
        name="Импорт",
        currency="€",
    )

    assert estimate.metadata.name == "Импорт"
    assert len(estimate.items) == 2
    first, second = estimate.items
    assert first.quantity == 12
    assert first.total_price == 180  # 12 * 15
    assert second.quantity == 8
    assert second.unit_price == 22.5


def test_import_from_gsheet_values(tmp_path: Path) -> None:
    constructor = EstimateConstructor(
        llm_client=None,
        llm_model="dummy",
        material_agent=None,
        storage_path=tmp_path,
    )
    importer = ExcelEstimateImporter(constructor)

    headers = ["Описание", "Ед.", "Qty", "Unit Price"]
    rows = [
        ["Штукатурка стен", "м2", "20", "8"],
        ["Грунтовка", "л", "5", "3.5"],
    ]
    df = importer.dataframe_from_values(headers, rows)
    estimate = importer.import_from_dataframe(
        df=df,
        column_map={
            "description": "Описание",
            "unit": "Ед.",
            "quantity": "Qty",
            "unit_price": "Unit Price",
        },
        name="GSheet import",
    )

    assert estimate.metadata.name == "GSheet import"
    assert len(estimate.items) == 2
    assert estimate.items[0].total_price == 160  # 20 * 8


def test_description_guess_prefers_text_over_numbers(tmp_path: Path) -> None:
    constructor = EstimateConstructor(
        llm_client=None,
        llm_model="dummy",
        material_agent=None,
        storage_path=tmp_path,
    )
    importer = ExcelEstimateImporter(constructor)

    headers = ["Artº Nº", "Designacao", "Unid"]
    rows = [
        ["1.1", "Porta interior", "un"],
        ["1.2", "Janela de aluminio", "un"],
    ]
    df = importer.dataframe_from_values(headers, rows)
    mapping = importer._build_mapping(list(df.columns), None, df=df)

    assert mapping.get("number") == "Artº Nº"
    assert mapping["description"] == "Designacao"


def test_quantity_fallback_picks_numeric_column(tmp_path: Path) -> None:
    constructor = EstimateConstructor(
        llm_client=None,
        llm_model="dummy",
        material_agent=None,
        storage_path=tmp_path,
    )
    importer = ExcelEstimateImporter(constructor)

    headers = ["Col1", "Col2", "Col3"]
    rows = [
        ["Work A", "12", "100"],
        ["Work B", "5", "200"],
    ]
    df = importer.dataframe_from_values(headers, rows)
    mapping = importer._build_mapping(list(df.columns), None, df=df)

    assert mapping["description"] == "Col1"
    assert mapping["quantity"] == "Col2"
