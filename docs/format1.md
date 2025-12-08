# Формат 1 (классический импорт)

Описание того, как агент импортирует сметы в “обычном” режиме (без переписывания исходного листа и без перевода).

## Что делает
- Читает Excel или Google Sheet, распознаёт колонки (описание, количество, ед., цена, итого, код).
- Создаёт внутреннюю смету (`Estimate`) и, при `create_sheet=true`, добавляет оформленные листы в Google Sheets (Estimate/Summary/Master).
- Не переводит и не изменяет исходный лист (если только явно не попросили `rewrite_source_sheet`).

## Полезные поля `column_map`
Можно задать названия колонок вручную; ключи: `description`, `quantity`, `unit`, `unit_price`, `total_price`, `number`, `code`. Явная карта имеет приоритет над авто-распознаванием.

## HTTP API пример (Excel)
`POST /api/estimate/import`
```json
{
  "path": "./data/source.xlsx",
  "sheet_name": "Sheet1",
  "column_map": {"description": "Описание работ", "quantity": "Qty"},
  "name": "Ремонт квартиры",
  "description": "Черновые + чистовые работы",
  "client_name": "Иван Иванов",
  "currency": "€",
  "skip_rows": 1,
  "default_item_type": "work",
  "create_sheet": true,
  "worksheet_name": "Import 01",
  "create_new_spreadsheet": false,
  "target_spreadsheet_id": null
}
```

## HTTP API пример (Google Sheets)
```json
{
  "google_sheet_id": "SHEET_ID",
  "worksheet_gid": 0,
  "column_map": {"description": "Descrição", "quantity": "Qtd"},
  "name": "Obra Lisboa",
  "description": "Arquitetura",
  "client_name": "Cliente",
  "currency": "€",
  "default_item_type": "work",
  "create_sheet": true,
  "create_new_spreadsheet": false,
  "target_spreadsheet_id": "SHEET_ID",
  "rewrite_source_sheet": false
}
```

## Поведение
- Если `path` указан — берём Excel, иначе работаем с Google Sheets.
- Авто-распознавание колонок по синонимам (рус/eng/pt). Явный `column_map` перебивает авто-детект.
- `skip_rows` помогает отбросить шапку/логотип перед заголовками.
- `create_sheet=true` — создаёт оформленные листы в таблице (или в новой, если `create_new_spreadsheet=true` и задан `target_spreadsheet_id`).
- При форматировании листов ставит базовый шрифт Calibri 12 и авто-подбор ширины/высоты ячеек под содержимое.
- `rewrite_source_sheet` для Формата 1 по умолчанию `false`: исходный лист не трогаем.

## Что вернуть в ответе
`estimate` — сериализованная смета; `sheets` — ссылки на созданные листы (если создавались).***
