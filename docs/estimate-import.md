# Excel → VIORA формат

Модуль `ExcelEstimateImporter` конвертирует произвольные Excel-сметы в внутренний формат и (опционально) раскладывает их в Google Sheets с фирменной шапкой VIORA BUILD.

## Зависимости
- `pandas` + `openpyxl` (добавлены в `requirements.txt`)
- Настроенный `ConstructionAIAgent` (для записи в Google Sheets нужны `GOOGLE_SHEET_ID` и сервисный аккаунт)

## Python API
```python
from unified_agent import ConstructionAIAgent

agent = ConstructionAIAgent()
result = agent.import_estimate_from_excel(
    path="./data/source.xlsx",
    sheet_name="Sheet1",               # опционально
    column_map={"description": "Описание работ", "quantity": "Qty"},  # опционально
    name="Ремонт квартиры",
    description="Черновые + чистовые работы",
    client_name="Иван Иванов",
    currency="€",
    skip_rows=1,                       # пропустить верхние строки с логотипом
    default_item_type="work",
    create_sheet=True,                 # создать листы в Google Sheets (Estimate, Summary, Master)
    worksheet_name="Import 01",
)
print(result["estimate"]["summary"])
```

Полезные поля результата:
- `estimate` — сериализованный объект сметы (`metadata`, `items`, `summary`)
- `sheets` — ссылки на созданные листы (если `create_sheet=True`)

## HTTP API
`POST /api/estimate/import`

```json
{
  "path": "./data/source.xlsx",
  "sheet_name": "Sheet1",
  "column_map": {"description": "Описание", "quantity": "Qty"},
  "name": "Ремонт",
  "description": "Квартира",
  "client_name": "Клиент",
  "currency": "€",
  "skip_rows": 0,
  "default_item_type": "work",
  "create_sheet": true,
  "worksheet_name": "Import 01"
}
```

## Как определяется маппинг колонок
- Авто-поиск по синонимам (рус/eng/pt): описание, ед., количество, цена за ед., итого, код.
- Можно передать `column_map={"description": "Описание работ", "quantity": "Qtd"}` — явные названия колонок имеют приоритет.
- Значение `skip_rows` помогает отбросить шапку/логотип перед строкой заголовков.

## Формат итогового листа
- Шапка: `VIORA BUILD`, заголовок работ, фирменные цвета.
- Таблица: `№, DESCRIÇÃO/ОПИСАНИЕ, Unid./Ед., Qtd./Кол-во, Preço por Un/Цена за ед., Valor Total/Итого`.
- Итоги: `TOTAL`, `IVA/НДС`, `TOTAL COM IVA/ИТОГО с НДС`.
