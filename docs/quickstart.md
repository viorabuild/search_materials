# Quickstart

## 1. Requirements
- Python 3.10+
- OpenAI API key with Responses API доступом
- (Опционально) сервисный аккаунт Google с правом редактирования нужных таблиц

## 2. Установка
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Конфигурация
1. Скопируйте шаблон:
   ```bash
   cp .env.example .env
   ```
2. Заполните как минимум:
   - `OPENAI_API_KEY` **или** установите `ENABLE_LOCAL_LLM=true` и задайте `LOCAL_LLM_BASE_URL` + `LOCAL_LLM_MODEL`
   - `GOOGLE_SERVICE_ACCOUNT_FILE` → путь к `credentials/service-account.json` (или используйте `GOOGLE_SERVICE_ACCOUNT_JSON`)
   - `GOOGLE_SHEET_ID` и `GOOGLE_SHEET_WORKSHEET`, если планируете работу с таблицами
3. Проверьте дополнительные параметры:
   - `ENABLE_WEB_SEARCH=true` для подключения LangChain-агента
   - `LOCAL_MATERIALS_CSV_PATH` — путь к локальной базе (по умолчанию `data/local_materials.csv`)
   - `ENABLE_LOCAL_MATERIAL_DB`, `ENABLE_MATERIALS_DB_ASSISTANT` — загрузка базы и активация помощника материалов
   - `MATERIALS_DB_SHEET_ID`, `MATERIALS_DB_WORKSHEET` — подключение Google Sheets с базой материалов (опционально). `MATERIALS_DB_WORKSHEET` может содержать несколько листов через запятую: `Материалы,Котировки,Работы`
   - `MATERIALS_DB_SYNC_CSV` — синхронизировать ли изменения из Google Sheets в локальный CSV
   - `ENABLE_LOCAL_LLM`, `LOCAL_LLM_BASE_URL=http://127.0.0.1:1234`, `LOCAL_LLM_MODEL=qwen/qwen3-vl-8b`, `LOCAL_LLM_API_KEY=lm-studio`, `LLM_REQUEST_TIMEOUT_SECONDS` — fallback на локальную LLM (например, LM Studio)
   - `CACHE_DB_PATH` — путь к SQLite-кэшу (создаётся автоматически)

> ⚠️ Если вы используете Google Sheets для базы материалов, поделитесь таблицей с сервисным аккаунтом (email из `credentials/service-account.json`) и выдайте права на редактирование.

### Проверка конфигурации
`ConstructionAIAgent` вызывает `ConstructionAIAgentConfig.validate()` во время запуска и сразу сообщает об отсутствующих параметрах. Наиболее частые сообщения:
- `OPENAI_API_KEY is required unless ENABLE_LOCAL_LLM=true` — нужно указать ключ OpenAI или включить локальную LLM.
- `Google Sheets integration requires GOOGLE_SERVICE_ACCOUNT_PATH or GOOGLE_SERVICE_ACCOUNT_JSON` — заданы Google Sheets, но не найден сервисный аккаунт.
- `Local materials CSV not found at ...` — включена локальная база материалов, но CSV отсутствует.
- `Materials DB assistant requires LOCAL_MATERIALS_CSV_PATH or MATERIALS_DB_SHEET_ID` — помощнику базы требуется CSV или Google Sheet.

Исправьте настройки `.env` или отключите связанные функции, чтобы продолжить инициализацию.

## 4. Размещение секретов
- Поместите сервисный аккаунт Google в `credentials/service-account.json`
- `.env`, `credentials/*.json`, `cache/*.db` игнорируются git и остаются локально

## 5. Запуск
```bash
./run_unified.sh
```
Скрипт автоматически создаст виртуальное окружение `.venv`, установит зависимости и стартует Flask-приложение (по умолчанию порт 8501).

## 6. Альтернативные варианты запуска
- CLI: `python unified_agent.py --interactive`
- Выполнить команду: `python unified_agent.py "Найди цену на цемент"`
- Использование в коде:
  ```python
  from unified_agent import ConstructionAIAgent
  agent = ConstructionAIAgent()
  print(agent.process_command("Прочитай таблицу"))
  ```

## 7. Проверка
После запуска откройте http://localhost:8501 и выполните тестовый запрос. Для проверки доступа к Google Sheets используйте вкладку "Проверка смет" или вызовите `agent.read_sheet_data()` в Python.
