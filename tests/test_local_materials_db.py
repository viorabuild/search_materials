import csv
import os
import sys
import tempfile
import time
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


def _append_row(path: Path, row: list[str]) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(row)
    current = path.stat().st_mtime
    new_time = max(time.time(), current + 1)
    os.utime(path, (new_time, new_time))


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


def test_auto_refresh_detects_external_updates() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "materials.csv"
        _write_local_csv(csv_path)

        db = LocalMaterialDatabase(csv_path)
        assert db.search("steel")

        _append_row(
            csv_path,
            [
                "Copper Pipe",
                "Трубы",
                "Медь",
                "м",
                "Metals Inc",
                "500",
                "https://example.com/copper",
            ],
        )

        refreshed_results = db.search("copper")
        assert refreshed_results, "Expected auto refresh to load new rows"
        assert refreshed_results[0].display_name == "Copper Pipe"


def test_auto_refresh_can_be_disabled() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "materials.csv"
        _write_local_csv(csv_path)

        original_env = os.environ.get("LOCAL_DB_AUTO_REFRESH")
        os.environ["LOCAL_DB_AUTO_REFRESH"] = "false"
        try:
            db = LocalMaterialDatabase(csv_path)
            assert db.search("steel")

            _append_row(
                csv_path,
                [
                    "Aluminum Sheet",
                    "Металлы",
                    "Листы",
                    "м2",
                    "Foil Corp",
                    "200",
                    "https://example.com/aluminum",
                ],
            )

            disabled_results = db.search("aluminum")
            assert disabled_results == []

            db.refresh()
            refreshed = db.search("aluminum")
            assert refreshed, "Manual refresh should load new rows when auto refresh disabled"
        finally:
            if original_env is None:
                del os.environ["LOCAL_DB_AUTO_REFRESH"]
            else:
                os.environ["LOCAL_DB_AUTO_REFRESH"] = original_env
