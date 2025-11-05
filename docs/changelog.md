# Changelog

## [Unreleased]
- Удалены устаревшие сервисы (FastAPI/Ollama, Streamlit, GPT-parser) и дублирующиеся README.
- Перенесены документы в `docs/`, обновлён `README.md` и `.env.example`.
- CSV локальной базы перемещена в `data/local_materials.csv` (ASCII-имя).
- Добавлены placeholder-файлы в `credentials/`, `cache/`, `logs/`; удалены реальные ключи.
- Обновлены зависимости (`requirements.txt`) и `.gitignore`.

## [3.0.0] – 2024-10-23
- Объединены Google Sheets AI и Material Price Finder в `ConstructionAIAgent`.
- Добавлено Flask-приложение (`unified_app.py`) и новый фронтенд (`frontend/unified.html`).
- Реализован модуль `advanced_agent.py` (LangChain + DuckDuckGo/Bing/ChatGPT Search).
- Добавлены `CacheManager`, `EstimateChecker`, улучшенный `MaterialPriceAgent`, примеры и документация.
