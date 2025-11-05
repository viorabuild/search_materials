# Architecture

```
┌─────────────────────┐      ┌─────────────────────────┐
│  Web / CLI Clients  │────▶ │   ConstructionAIAgent    │
└─────────────────────┘      └──────────┬──────────────┘
                                        │
        ┌───────────────────────────────┼──────────────────────────────┐
        │                               │                              │
┌───────▼────────┐             ┌────────▼────────┐            ┌────────▼────────┐
│ MaterialPrice  │             │ GoogleSheetsAI  │            │ EstimateChecker │
│ Agent          │             │ (markdown_ai)   │            │                 │
└───────┬────────┘             └────────┬────────┘            └────────┬────────┘
        │                             CacheManager                     │
        │                             (SQLite)                         │
        │                                                            Local
┌───────▼────────┐       ┌──────────────┐       ┌──────────────┐      Data
│ SiteScrapers   │◀──────│ GPT Scraper  │◀──────│ LangChain     │◀───────┐
│ & Known DB     │       │ (optional)   │       │ AdvancedAgent │       │
└────────────────┘       └──────────────┘       └──────────────┘       │
                                                                       │
                                                              data/local_materials.csv
```

- **ConstructionAIAgent** (`unified_agent.py`) объединяет все сервисы, управляет конфигурацией и кэшированием.
- **MaterialPriceAgent** (`material_price_agent.py`) строит поисковые запросы, обращается к LLM и скрейперам.
- **SiteScrapers** (`site_scrapers.py`) и **GPTScraper** (`gpt_scraper.py`) собирают реальные цены, когда это разрешено.
- **GoogleSheetsAI** (`markdown_ai.py`) отвечает за взаимодействие с Google Sheets и формирование Markdown-отчётов.
- **EstimateChecker** (`estimate_checker.py`) анализирует сметы и формирует структурированные отчёты.
- **CacheManager** (`cache_manager.py`) использует SQLite (`cache/materials.db`) для хранения результатов.
- **AdvancedMaterialAgent** (`advanced_agent.py`) добавляет LangChain/duckduckgo-search для сложных запросов и включается через `ENABLE_WEB_SEARCH=true`.
- **MaterialsDatabaseAssistant** (`materials_db_assistant.py`) работает с базой материалов в CSV или Google Sheets: ищет записи, готовит вставки/обновления/удаления и требует подтверждения перед применением.

Файлы с конфигурацией: `.env`, `.env.example`, `credentials/README.md`.
