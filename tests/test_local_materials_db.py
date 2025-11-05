import csv
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_materials_db import LocalMaterialDatabase


def _write_local_csv(path: Path) -> None:
    headers = [
        "Nome",
        "Категория",
        "Группа",
        "Ед. Измерения",
        "Последний Поставщик",
        "Стоимость",
        "Поставщики",
    ]
    rows = [
        [
            "Steel Beam",
            "Металлопрокат",
            "Балка",
            "шт",
            "Local Supplier",
            "1000",
            "https://example.com/steel",
        ]
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def test_search_preserves_original_display_name() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "materials.csv"
        _write_local_csv(csv_path)

        db = LocalMaterialDatabase(csv_path)
        assert db.available

        lowercase_results = db.search("steel beam")
        assert lowercase_results, "Expected to find results for lowercase query"
        assert lowercase_results[0].display_name == "Steel Beam"

        uppercase_results = db.search("STEEL BEAM")
        assert uppercase_results, "Expected to find results for uppercase query"
        assert uppercase_results[0].display_name == "Steel Beam"
        assert uppercase_results[0].display_name == lowercase_results[0].display_name
