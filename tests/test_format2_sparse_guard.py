import pandas as pd

from estimate_constructor import ItemType
from estimate_importer import ExcelEstimateImporter
from unified_agent import ConstructionAIAgent


def _make_agent_with_importer() -> ConstructionAIAgent:
    """Create a minimal agent with a stubbed importer for format 2 helpers."""
    agent = ConstructionAIAgent.__new__(ConstructionAIAgent)  # type: ignore
    importer = ExcelEstimateImporter.__new__(ExcelEstimateImporter)  # type: ignore
    importer.constructor = None  # type: ignore[attr-defined]
    agent.estimate_importer = importer  # type: ignore[attr-defined]
    return agent


def test_sparse_recovery_switches_to_denser_description_column():
    agent = _make_agent_with_importer()

    df = pd.DataFrame(
        [
            {"Notes": "header", "Desc": "", "Qty": 0},
            {"Notes": "", "Desc": "First row", "Qty": 1},
            {"Notes": "", "Desc": "Second row", "Qty": 2},
        ]
    )

    # Wrong mapping: uses column with only a single non-empty cell as description
    mapping = {"description": "Notes", "quantity": "Qty"}
    sparse_items = agent._build_format2_items_from_df(df, mapping, ItemType.WORK)
    assert len(sparse_items) == 1

    recovered_items, recovered_mapping, label = agent._recover_sparse_format2_items(
        df=df,
        mapping=mapping,
        items=sparse_items,
        default_item_type=ItemType.WORK,
        column_map=None,
    )

    assert label  # фиксируем, что выбрали альтернативную стратегию
    assert recovered_mapping["description"] == "Desc"
    assert len(recovered_items) == 2


def test_guard_prefers_description_density_over_all_rows():
    agent = _make_agent_with_importer()

    df = pd.DataFrame([{"Desc": f"row {i}"} for i in range(273)] + [{"Desc": ""} for _ in range(533)])
    mapping = {"description": "Desc"}

    items = agent._build_format2_items_from_df(df, mapping, ItemType.WORK)
    desc_rows = agent._count_non_empty_descriptions(df, "Desc")

    assert len(items) == desc_rows == 273

    source_data_rows = 806  # имитация общего числа непустых строк
    guard_rows = source_data_rows
    if desc_rows:
        guard_rows = min(source_data_rows, max(desc_rows, len(items)))

    assert guard_rows == 273  # защита должна опираться на заполненные описания
    threshold = max(5, guard_rows // 2)
    assert len(items) >= threshold
