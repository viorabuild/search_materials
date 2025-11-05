# Migration Notes

Этот репозиторий объединяет ранее разрозненные проекты (Google Sheets AI, Material Price Finder и вспомогательные скрипты).

## Основные изменения
- Единый класс `ConstructionAIAgent` заменяет отдельные CLI/скрипты.
- Конфигурация перенесена в `.env` + `ConstructionAIAgentConfig`.
- Локальная база материалов переехала в `data/local_materials.csv` (ранее `База Материалов.csv`).
- Все устаревшие интерфейсы (FastAPI, Streamlit, Ollama-агенты) удалены.
- Документация перенесена в каталог `docs/`.

## Что делать при обновлении
1. Пересоздайте `.env` из нового шаблона и проверьте пути.
2. Переместите свой сервисный аккаунт в `credentials/service-account.json`.
3. Если использовали старые скрипты (`app.py`, `main.py`, `run.sh` и т.п.) — запустите `./run_unified.sh` или используйте `python unified_agent.py`.
4. Обновите ссылки на локальную CSV-базу, если у вас был собственный файл.

## Обратная совместимость
- API `ConstructionAIAgent` сохраняет основные методы: `find_material_price`, `find_materials_batch`, `process_command`, `read_sheet_data`, `check_estimate` и т.д.
- Продвинутый поиск (`advanced_agent.py`) остаётся модулем, подключаемым через `ENABLE_WEB_SEARCH=true`.
