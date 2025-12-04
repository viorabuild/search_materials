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
