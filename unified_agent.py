"""Construction AI Agent - объединяет все возможности проекта.

Этот модуль интегрирует:
1. Google Sheets AI (markdown_ai.py) - работа с таблицами через естественный язык
2. Material Price Agent (material_price_agent.py) - поиск цен на материалы
3. Advanced Agent (advanced_agent.py) - LangChain + веб-поиск
4. Cached Agent (cached_agent.py) - кэширование результатов
5. Estimate Checker - проверка строительных смет

Единый интерфейс для всех операций с максимальной функциональностью.
"""

from __future__ import annotations

import asyncio
import atexit
import importlib
import json
import logging
import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
from dataclasses import dataclass
from datetime import datetime

import gspread
from gspread.exceptions import APIError, WorksheetNotFound
from dotenv import load_dotenv

REFERENCE_DIR = Path("./data/estimate_references")
from llm_provider import FallbackOpenAI, LLMFallbackConfig

try:  # pragma: no cover - executed only when dependency is available
    from prometheus_client import Counter, Histogram, start_http_server
except ImportError:  # pragma: no cover - fallback for offline environments
    from observability_fallback import Counter, Histogram, start_http_server

# Импорты из существующих модулей
from material_price_agent import (
    MaterialPriceAgent,
    MaterialResult,
    MaterialQueryAnalysis,
    BestOffer,
    SupplierQuote,
    SearchTimeoutError,
)
from estimate_constructor import (
    Estimate,
    EstimateConstructor,
    EstimateItem,
    ItemType,
    create_quick_estimate,
)
from estimate_importer import ExcelEstimateImporter
from cache_manager import CacheManager
from markdown_ai import GoogleSheetsAI
from local_materials_db import LocalMaterialDatabase
from materials_db_assistant import MaterialsDatabaseAssistant
from project_chat_agent import ProjectChatAgent, ProjectChatError

try:
    from estimate_checker import EstimateChecker
    ESTIMATE_CHECKER_AVAILABLE = True
except ImportError:
    ESTIMATE_CHECKER_AVAILABLE = False
    logging.warning("Estimate checker not available")

if TYPE_CHECKING:
    from advanced_agent import MaterialSearchResult

load_dotenv()
logger = logging.getLogger(__name__)


MATERIAL_SEARCH_REQUESTS = Counter(
    "construction_material_search_requests_total",
    "Total number of material price searches.",
)
MATERIAL_SEARCH_DURATION = Histogram(
    "construction_material_search_duration_seconds",
    "Duration of material price search operations in seconds.",
)
MATERIAL_CACHE_HITS = Counter(
    "construction_material_cache_hits_total",
    "Number of material price cache hits.",
)
MATERIAL_CACHE_MISSES = Counter(
    "construction_material_cache_misses_total",
    "Number of material price cache misses.",
)
SHEETS_COMMAND_REQUESTS = Counter(
    "construction_sheets_commands_total",
    "Total number of Google Sheets AI commands processed.",
)
SHEETS_COMMAND_DURATION = Histogram(
    "construction_sheets_command_duration_seconds",
    "Duration of Google Sheets AI command processing in seconds.",
)
ESTIMATE_CHECK_REQUESTS = Counter(
    "construction_estimate_checks_total",
    "Total number of estimate checks performed.",
)
ESTIMATE_CHECK_DURATION = Histogram(
    "construction_estimate_check_duration_seconds",
    "Duration of estimate check operations in seconds.",
)
LLM_REQUESTS = Counter(
    "construction_llm_requests_total",
    "Total number of logical requests sent to an LLM backend.",
)

_METRICS_SERVER_STARTED = False


def _start_metrics_server() -> None:
    """Start the Prometheus metrics HTTP exporter if it is not running yet."""

    global _METRICS_SERVER_STARTED
    if _METRICS_SERVER_STARTED:
        return

    port = int(os.getenv("METRICS_PORT", "9000"))
    start_http_server(port)
    _METRICS_SERVER_STARTED = True
    logger.info("Prometheus metrics exporter started on port %s", port)


def _instrument_openai_client(client: FallbackOpenAI) -> None:
    """Wrap the fallback OpenAI client to count logical LLM requests."""

    if getattr(client, "_metrics_wrapped", False):
        return

    original_call = client._call_with_fallback

    def _wrapped_call(*args: Any, **kwargs: Any):  # type: ignore[override]
        LLM_REQUESTS.inc()
        return original_call(*args, **kwargs)

    client._call_with_fallback = _wrapped_call  # type: ignore[attr-defined]
    client._metrics_wrapped = True


def _normalize_query(value: Optional[str]) -> str:
    return value.strip().lower() if value else ""


@dataclass
class ConstructionAIAgentConfig:
    """Конфигурация для единого агента."""
    
    # OpenAI
    openai_api_key: Optional[str] = None
    llm_model: str = "openai/gpt-oss-20b"
    temperature: float = 1.0
    llm_request_timeout: Optional[float] = None
    enable_local_llm: bool = False
    local_llm_base_url: Optional[str] = None
    local_llm_model: Optional[str] = None
    local_llm_api_key: str = "lm-studio"
    
    # Google Sheets
    google_service_account_path: Optional[str] = None
    google_sheet_id: Optional[str] = None
    google_worksheet_name: str = "Sheet1"
    
    # Cache
    cache_db_path: str = "./cache/materials.db"
    cache_ttl_seconds: int = 86400
    cache_cleanup_interval: float = 3600.0
    
    # Material search
    request_delay_seconds: float = 1.0
    enable_known_sites: bool = True
    known_sites_only: bool = False
    max_parallel_material_searches: int = 4
    
    # Web search
    use_bing: bool = False
    bing_api_key: Optional[str] = None
    use_chatgpt_search: bool = False

    # Advanced features
    enable_web_search: bool = False
    enable_gpt_scraping: bool = False
    enable_real_scraping: bool = False
    enable_local_db: bool = True
    local_materials_csv_path: Optional[str] = "./data/local_materials.csv"
    default_timeout_seconds: float = 60.0
    exact_match_only_default: bool = False
    enable_materials_db_assistant: bool = True
    materials_db_sheet_id: Optional[str] = None
    materials_db_worksheet: Optional[str] = None
    materials_db_sync_csv: bool = True
    
    # Project chat
    enable_project_chat: bool = True
    project_chat_context_files: Optional[str] = None
    project_chat_system_prompt: Optional[str] = None
    project_chat_max_history_turns: int = 6
    enable_estimate_constructor: bool = True
    estimate_storage_path: str = "./data/estimates"
    
    @classmethod
    def from_env(cls) -> ConstructionAIAgentConfig:
        """Создать конфигурацию из переменных окружения."""
        use_chatgpt_search = os.getenv("USE_CHATGPT_SEARCH", "false").lower() == "true"
        enable_web_search = os.getenv("ENABLE_WEB_SEARCH", "false").lower() == "true"
        llm_model = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
        temperature = float(os.getenv("LLM_TEMPERATURE", "1.0"))
        
        if llm_model.startswith("gpt-5-nano") and temperature != 1.0:
            logger.warning(
                "gpt-5-nano поддерживает только temperature=1.0. Значение %.2f будет переопределено.",
                temperature,
            )
            temperature = 1.0

        enable_local_llm = os.getenv("ENABLE_LOCAL_LLM", "false").lower() == "true"
        local_llm_base_url = os.getenv("LOCAL_LLM_BASE_URL")
        if enable_local_llm and not local_llm_base_url:
            local_llm_base_url = "http://127.0.0.1:1234"
        local_llm_model = os.getenv("LOCAL_LLM_MODEL") or "qwen/qwen3-vl-4b"
        local_llm_api_key = os.getenv("LOCAL_LLM_API_KEY") or "lm-studio"

        enable_estimate_constructor = os.getenv("ENABLE_ESTIMATE_CONSTRUCTOR", "true").lower() == "true"
        estimate_storage_path = os.getenv("ESTIMATE_STORAGE_PATH", "./data/estimates")

        request_timeout_env = os.getenv("LLM_REQUEST_TIMEOUT_SECONDS")
        llm_request_timeout = float(request_timeout_env) if request_timeout_env else None

        worksheet_env = os.getenv("GOOGLE_SHEET_WORKSHEET")
        if worksheet_env is not None:
            worksheet_env = worksheet_env.strip() or None

        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            llm_model=llm_model,
            temperature=temperature,
            llm_request_timeout=llm_request_timeout,
            enable_local_llm=enable_local_llm,
            local_llm_base_url=local_llm_base_url,
            local_llm_model=local_llm_model,
            local_llm_api_key=local_llm_api_key,
            google_service_account_path=os.getenv("GOOGLE_SERVICE_ACCOUNT_PATH") or os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE"),
            google_sheet_id=os.getenv("GOOGLE_SHEET_ID"),
            google_worksheet_name=worksheet_env,
            cache_db_path=os.getenv("CACHE_DB_PATH", "./cache/materials.db"),
            cache_ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "86400")),
            cache_cleanup_interval=float(os.getenv("CACHE_CLEANUP_INTERVAL_SECONDS", "3600")),
            request_delay_seconds=float(os.getenv("REQUEST_DELAY_SECONDS", "1.0")),
            enable_known_sites=os.getenv("ENABLE_KNOWN_SITE_SEARCH", "true").lower() == "true",
            known_sites_only=os.getenv("KNOWN_SITE_SEARCH_ONLY", "false").lower() == "true",
            max_parallel_material_searches=max(
                1,
                int(os.getenv("MAX_PARALLEL_MATERIAL_SEARCHES", "4")),
            ),
            use_bing=os.getenv("USE_BING_SEARCH", "false").lower() == "true",
            bing_api_key=os.getenv("BING_API_KEY"),
            use_chatgpt_search=use_chatgpt_search,
            enable_web_search=enable_web_search or use_chatgpt_search,
            enable_gpt_scraping=os.getenv("ENABLE_GPT_SCRAPING", "false").lower() == "true",
            enable_real_scraping=os.getenv("ENABLE_REAL_SCRAPING", "false").lower() == "true",
            enable_local_db=os.getenv("ENABLE_LOCAL_MATERIAL_DB", "true").lower() == "true",
            local_materials_csv_path=os.getenv("LOCAL_MATERIALS_CSV_PATH") or "./data/local_materials.csv",
            default_timeout_seconds=float(os.getenv("DEFAULT_SEARCH_TIMEOUT_SECONDS", "60")),
            exact_match_only_default=os.getenv("EXACT_MATCH_ONLY", "false").lower() == "true",
            enable_materials_db_assistant=os.getenv("ENABLE_MATERIALS_DB_ASSISTANT", "true").lower() == "true",
            materials_db_sheet_id=os.getenv("MATERIALS_DB_SHEET_ID"),
            materials_db_worksheet=os.getenv("MATERIALS_DB_WORKSHEET"),
            materials_db_sync_csv=os.getenv("MATERIALS_DB_SYNC_CSV", "true").lower() == "true",
            enable_project_chat=os.getenv("ENABLE_PROJECT_CHAT", "true").lower() == "true",
            project_chat_context_files=os.getenv("PROJECT_CHAT_CONTEXT_FILES"),
            project_chat_system_prompt=os.getenv("PROJECT_CHAT_SYSTEM_PROMPT"),
            project_chat_max_history_turns=int(os.getenv("PROJECT_CHAT_MAX_HISTORY_TURNS", "6")),
            enable_estimate_constructor=enable_estimate_constructor,
            estimate_storage_path=estimate_storage_path,
        )


class ConstructionAIAgent:
    """
    Единый AI-агент, объединяющий все возможности проекта.
    
    Возможности:
    - Работа с Google Sheets через естественный язык
    - Поиск цен на строительные материалы
    - Веб-поиск и парсинг сайтов
    - Проверка строительных смет
    - Кэширование результатов
    - LangChain интеграция для сложных задач
    """
    
    def __init__(self, config: Optional[ConstructionAIAgentConfig] = None):
        """
        Инициализация единого агента.
        
        Args:
            config: Конфигурация агента (если None, загружается из .env)
        """
        self.config = config or ConstructionAIAgentConfig.from_env()

        if not self.config.openai_api_key and not self.config.enable_local_llm:
            raise ValueError("OPENAI_API_KEY is required unless ENABLE_LOCAL_LLM=true")

        if self.config.llm_model.startswith("gpt-5-nano") and self.config.temperature != 1.0:
            logger.warning(
                "gpt-5-nano поддерживает только temperature=1.0. Параметр будет переопределен."
            )
            self.config.temperature = 1.0

        logger.info("Initializing ConstructionAIAgent with model %s", self.config.llm_model)

        fallback_config = LLMFallbackConfig(
            primary_api_key=self.config.openai_api_key,
            primary_model=self.config.llm_model,
            request_timeout=self.config.llm_request_timeout,
            local_enabled=self.config.enable_local_llm,
            local_base_url=self.config.local_llm_base_url,
            local_model=self.config.local_llm_model,
            local_api_key=self.config.local_llm_api_key,
        )

        self.openai_client = FallbackOpenAI(fallback_config)
        _start_metrics_server()
        _instrument_openai_client(self.openai_client)
        
        # Инициализация базового агента для поиска материалов
        self.material_agent = MaterialPriceAgent(
            openai_api_key=self.config.openai_api_key,
            openai_client=self.openai_client,
            google_service_account_path=self.config.google_service_account_path if self.config.google_service_account_path and Path(self.config.google_service_account_path).exists() else None,
            llm_model=self.config.llm_model,
            temperature=self.config.temperature,
            request_delay_seconds=self.config.request_delay_seconds,
            enable_known_sites=self.config.enable_known_sites,
            known_sites_only=self.config.known_sites_only,
        )

        self.estimate_constructor: Optional[EstimateConstructor] = None
        self.estimate_importer: Optional[ExcelEstimateImporter] = None
        if self.config.enable_estimate_constructor:
            try:
                storage_path = Path(self.config.estimate_storage_path)
                REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
                self.estimate_constructor = EstimateConstructor(
                    llm_client=self.openai_client,
                    llm_model=self.config.llm_model,
                    material_agent=self.material_agent,
                    storage_path=storage_path,
                )
                self.estimate_importer = ExcelEstimateImporter(self.estimate_constructor)
                logger.info("Estimate Constructor initialized (storage=%s)", storage_path)
            except Exception as exc:
                logger.warning("Failed to initialize Estimate Constructor: %s", exc)
        
        # Инициализация кэш-менеджера
        self.cache = CacheManager(
            db_path=Path(self.config.cache_db_path),
            cache_ttl_seconds=self.config.cache_ttl_seconds,
        )
        self.cache.clear_expired()

        self._cache_cleanup_stop_event = threading.Event()
        self._cache_cleanup_thread: Optional[threading.Thread] = None
        self._start_cache_cleanup_task()
        atexit.register(self.close)
        
        # Инициализация Google Sheets AI (если настроен)
        self.sheets_ai = None
        if self.config.google_sheet_id:
            try:
                self.sheets_ai = GoogleSheetsAI(
                    sheet_id=self.config.google_sheet_id,
                    worksheet_name=self.config.google_worksheet_name,
                    openai_client=self.openai_client,
                    llm_model=self.config.llm_model,
                )
                logger.info("Google Sheets AI initialized")
            except Exception as e:
                logger.warning("Failed to initialize Google Sheets AI: %s", e)
        
        self._gspread_client: Optional[gspread.Client] = None
        if self.sheets_ai:
            self._gspread_client = self.sheets_ai.gspread_client
        else:
            self._gspread_client = self._create_gspread_client()
        
        # Инициализация продвинутого агента (если настроен)
        self.advanced_agent = None
        self._advanced_agent_error: Optional[Exception] = None
        if self.config.enable_web_search:
            try:
                advanced_module = importlib.import_module("advanced_agent")
                AdvancedMaterialAgent = getattr(advanced_module, "AdvancedMaterialAgent")
                self.advanced_agent = AdvancedMaterialAgent(
                    openai_api_key=self.config.openai_api_key,
                    openai_client=self.openai_client,
                    model_name=self.config.llm_model,
                    temperature=self.config.temperature,
                    use_bing=self.config.use_bing,
                    bing_api_key=self.config.bing_api_key,
                    use_chatgpt_search=self.config.use_chatgpt_search,
                    local_llm_enabled=self.config.enable_local_llm,
                    local_llm_base_url=self.config.local_llm_base_url,
                    local_llm_model=self.config.local_llm_model,
                    local_llm_api_key=self.config.local_llm_api_key,
                )
                logger.info("Advanced agent with web search initialized")
            except ImportError as exc:
                self._advanced_agent_error = exc
                logger.info(
                    "Advanced agent module not found. Install 'advanced_agent' package "
                    "and set ENABLE_WEB_SEARCH=true to enable this feature."
                )
            except Exception as e:
                self._advanced_agent_error = e
                logger.warning("Failed to initialize advanced agent: %s", e)

        materials_csv_path = (
            Path(self.config.local_materials_csv_path)
            if self.config.local_materials_csv_path
            else None
        )

        self.local_materials_db: Optional[LocalMaterialDatabase] = None
        if self.config.enable_local_db and materials_csv_path:
            try:
                self.local_materials_db = LocalMaterialDatabase(materials_csv_path)
                if not self.local_materials_db.available:
                    logger.info(
                        "Local materials database at %s is not available or empty",
                        materials_csv_path,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to initialize local materials database: %s", exc)
        else:
            logger.info("Local materials database is disabled")

        self.materials_db_assistant: Optional[MaterialsDatabaseAssistant] = None
        assistant_enabled = self.config.enable_materials_db_assistant and (
            materials_csv_path is not None or self.config.materials_db_sheet_id
        )
        if assistant_enabled:
            worksheet_names = None
            if self.config.materials_db_worksheet:
                worksheet_names = [
                    item.strip()
                    for item in self.config.materials_db_worksheet.split(",")
                    if item.strip()
                ]
            try:
                self.materials_db_assistant = MaterialsDatabaseAssistant(
                    csv_path=materials_csv_path,
                    sheet_id=self.config.materials_db_sheet_id,
                    worksheet_name=self.config.materials_db_worksheet
                    if worksheet_names is None or len(worksheet_names) <= 1
                    else None,
                    worksheet_names=worksheet_names,
                    gspread_client=self._gspread_client,
                    sync_csv=self.config.materials_db_sync_csv,
                    llm_client=self.openai_client,
                    llm_model=self.config.llm_model,
                    local_db=self.local_materials_db,
                )
                logger.info("Materials database assistant initialized")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to initialize materials DB assistant: %s", exc)
        else:
            logger.info("Materials database assistant is disabled")
    
        self.default_timeout_seconds = self.config.default_timeout_seconds
        self._batch_rate_lock = Lock()
        self._last_batch_request_ts = 0.0
        
        self.project_chat_agent: Optional[ProjectChatAgent] = None
        if self.config.enable_project_chat:
            try:
                context_files = None
                if self.config.project_chat_context_files:
                    context_files = [
                        Path(item.strip())
                        for item in self.config.project_chat_context_files.split(",")
                        if item.strip()
                    ]
                agent_kwargs = {}
                if self.config.project_chat_system_prompt:
                    agent_kwargs["system_prompt"] = self.config.project_chat_system_prompt
                self.project_chat_agent = ProjectChatAgent(
                    client=self.openai_client,
                    model=self.config.llm_model,
                    project_root=Path(__file__).resolve().parent,
                    context_files=context_files,
                    max_history_turns=self.config.project_chat_max_history_turns,
                    request_timeout=self.config.llm_request_timeout,
                    **agent_kwargs,
                )
                logger.info("Project chat agent initialized")
            except Exception as exc:  # noqa: BLE001
                self.project_chat_agent = None
                logger.warning("Failed to initialize project chat agent: %s", exc)
        else:
            logger.info("Project chat agent is disabled")

        logger.info("ConstructionAIAgent initialized successfully")

    def _start_cache_cleanup_task(self) -> None:
        """Запустить фоновую задачу очистки кэша."""
        interval = getattr(self.config, "cache_cleanup_interval", None)
        if not interval or interval <= 0:
            return

        if self._cache_cleanup_thread is not None:
            return

        def _cleanup_loop() -> None:
            while not self._cache_cleanup_stop_event.wait(interval):
                try:
                    self.cache.clear_expired()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Cache cleanup task failed: %s", exc)

        self._cache_cleanup_thread = threading.Thread(
            target=_cleanup_loop,
            name="ConstructionAIAgentCacheCleanup",
            daemon=True,
        )
        self._cache_cleanup_thread.start()

    def close(self) -> None:
        """Остановить фоновые задачи агента."""
        stop_event = getattr(self, "_cache_cleanup_stop_event", None)
        if stop_event and not stop_event.is_set():
            stop_event.set()

        thread = getattr(self, "_cache_cleanup_thread", None)
        if thread and thread.is_alive():
            thread.join(timeout=1.0)
        self._cache_cleanup_thread = None

    # ========================================================================
    # МАТЕРИАЛЫ: Поиск цен на строительные материалы
    # ========================================================================

    def _create_gspread_client(self) -> Optional[gspread.Client]:
        """Создать gspread клиента для работы с базой материалов."""
        json_credentials = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
        if json_credentials:
            try:
                credentials_dict = json.loads(json_credentials)
                return gspread.service_account_from_dict(credentials_dict)
            except json.JSONDecodeError as exc:
                logger.warning("GOOGLE_SERVICE_ACCOUNT_JSON содержит некорректный JSON: %s", exc)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to create gspread client from JSON: %s", exc)

        service_account_path = self.config.google_service_account_path
        if service_account_path:
            path = Path(service_account_path).expanduser()
            if not path.exists():
                logger.warning("Google service account file not found at %s", path)
            else:
                try:
                    return gspread.service_account(filename=str(path))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to create gspread client from file %s: %s", path, exc)

        return None

    def _ensure_gspread_client(self) -> gspread.Client:
        """Гарантировать наличие клиента gspread."""
        if not getattr(self, "_gspread_client", None):
            self._gspread_client = self._create_gspread_client()
        if not self._gspread_client:
            raise RuntimeError("Не удалось инициализировать Google Sheets client.")
        return self._gspread_client
    
    def find_material_price(
        self,
        material_name: str,
        use_cache: bool = True,
        use_scraping: bool = False,
        use_advanced_search: bool = False,
        timeout_seconds: Optional[float] = None,
        exact_match_only: Optional[bool] = None,
    ) -> MaterialResult:
        """
        Найти лучшую цену для материала.
        
        Args:
            material_name: Название материала
            use_cache: Использовать кэш
            use_scraping: Использовать парсинг сайтов
            use_advanced_search: Использовать продвинутый поиск с LangChain
            timeout_seconds: Лимит по времени на поиск (секунды)
            exact_match_only: Искомое название должно совпадать точно с локальной базой
        
        Returns:
            MaterialResult с информацией о лучшем предложении
        """
        MATERIAL_SEARCH_REQUESTS.inc()
        with MATERIAL_SEARCH_DURATION.time():
            logger.info("Searching for material: %s", material_name)
            timeout = (
                timeout_seconds
                if timeout_seconds is not None
                else self.default_timeout_seconds
            )
            deadline = time.time() + timeout if timeout else None
            analysis: Optional[MaterialQueryAnalysis] = None
            match_exact = (
                exact_match_only
                if exact_match_only is not None
                else self.config.exact_match_only_default
            )

            try:
                cache_checked = False
                # Проверка кэша
                if use_cache:
                    cache_checked = True
                    cached = self.cache.get_material(material_name)
                    if cached:
                        MATERIAL_CACHE_HITS.inc()
                        logger.info("Using cached result for '%s'", material_name)
                        return self._reconstruct_material_result(material_name, cached)

                if cache_checked:
                    MATERIAL_CACHE_MISSES.inc()

                if deadline and time.time() >= deadline:
                    raise SearchTimeoutError(
                        f"Time limit exceeded before starting search for '{material_name}'"
                    )

                # Продвинутый поиск через LangChain
                if use_advanced_search and self.advanced_agent:
                    logger.info("Using advanced agent for '%s'", material_name)
                    if deadline and time.time() >= deadline:
                        raise SearchTimeoutError(
                            f"Time limit exceeded before advanced search for '{material_name}'"
                        )
                    advanced_result = self.advanced_agent.find_best_price(material_name)
                    if deadline and time.time() >= deadline:
                        raise SearchTimeoutError(
                            f"Time limit exceeded after advanced search for '{material_name}'"
                        )
                    result = self._convert_advanced_result(advanced_result)
                else:
                    # Обычный поиск
                    if deadline and time.time() >= deadline:
                        raise SearchTimeoutError(
                            f"Time limit exceeded before analysis for '{material_name}'"
                        )
                    analysis = self.material_agent.analyze_material(material_name)
                    if deadline and time.time() >= deadline:
                        raise SearchTimeoutError(
                            f"Time limit exceeded after analysis for '{material_name}'"
                        )
                    result = self.material_agent._resolve_material(
                        material_name,
                        analysis,
                        use_scraping,
                        self.config.enable_known_sites,
                        self.config.known_sites_only,
                        use_real_scraping=self.config.enable_real_scraping,
                        use_gpt_scraping=self.config.enable_gpt_scraping,
                        deadline=deadline,
                    )
            except SearchTimeoutError as exc:
                logger.warning("Search for '%s' stopped due to time limit: %s", material_name, exc)
                result = self._timeout_material_result(material_name, analysis, timeout)

            result = self._attach_local_suggestions(result, material_name, match_exact)

            # Сохранение в кэш
            if use_cache and result.best_offer.best_supplier != "N/A":
                self._cache_material_result(material_name, result)

            return result
    
    def find_materials_batch(
        self,
        materials: List[str],
        use_cache: bool = True,
        use_scraping: bool = False,
        timeout_seconds: Optional[float] = None,
        exact_match_only: Optional[bool] = None,
    ) -> List[MaterialResult]:
        """
        Найти цены для списка материалов.
        
        Args:
            materials: Список названий материалов
            use_cache: Использовать кэш
            use_scraping: Использовать парсинг
            timeout_seconds: Лимит по времени на поиск одного материала
        
        Returns:
            Список результатов
        """
        if not materials:
            return []

        max_workers = max(
            1,
            min(self.config.max_parallel_material_searches, len(materials)),
        )

        def _search_material(index: int, name: str) -> tuple[int, MaterialResult]:
            self._wait_for_batch_slot()
            result = self.find_material_price(
                name,
                use_cache=use_cache,
                use_scraping=use_scraping,
                timeout_seconds=timeout_seconds,
                exact_match_only=exact_match_only,
            )
            return index, result

        if len(materials) == 1 or max_workers == 1:
            sequential_results: List[MaterialResult] = []
            for idx, material in enumerate(materials):
                _, result = _search_material(idx, material)
                sequential_results.append(result)
            return sequential_results

        ordered_results: List[Optional[MaterialResult]] = [None] * len(materials)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_search_material, idx, material)
                for idx, material in enumerate(materials)
            ]
            for future in as_completed(futures):
                index, result = future.result()
                ordered_results[index] = result

        return [result for result in ordered_results if result is not None]
    
    # ========================================================================
    # ПРОЕКТНЫЙ ЧАТ: Диалог об устройстве репозитория
    # ========================================================================
    
    def chat_about_project(
        self,
        message: str,
        extra_context: Optional[str] = None,
        reset_history: bool = False,
    ) -> str:
        """Побеседовать с проектным ассистентом."""
        if not self.project_chat_agent:
            raise RuntimeError("Project chat agent is not available")
        
        if reset_history:
            self.project_chat_agent.reset_history()
            if not message or not message.strip():
                return "История проектного чата сброшена."
        
        if not message or not message.strip():
            raise ValueError("Message is required for project chat")
        
        return self.project_chat_agent.chat(message, extra_context=extra_context)

    def _timeout_material_result(
        self,
        material_name: str,
        analysis: Optional[MaterialQueryAnalysis],
        timeout_seconds: float,
    ) -> MaterialResult:
        """Сформировать результат при превышении лимита времени."""
        effective_analysis = analysis or MaterialQueryAnalysis(
            pt_name=material_name,
            search_queries=[],
            key_specs=[],
        )
        return MaterialResult(
            material_name=material_name,
            analysis=effective_analysis,
            quotes=[],
            best_offer=BestOffer(
                best_supplier="N/A",
                price="N/A",
                url="N/A",
                reasoning=f"⏱ Поиск остановлен: превышен лимит {timeout_seconds:.0f} сек.",
            ),
        )

    def _attach_local_suggestions(
        self,
        result: MaterialResult,
        query: str,
        exact_match_only: bool,
    ) -> MaterialResult:
        if not self.local_materials_db or not self.local_materials_db.available:
            return result

        suggestions = self.local_materials_db.search(
            query,
            exact_only=exact_match_only,
            limit=5,
        )

        # Повторная попытка с португальским названием, если оно есть.
        if (
            not suggestions
            and result.analysis
            and result.analysis.pt_name
            and _normalize_query(result.analysis.pt_name) != _normalize_query(query)
        ):
            suggestions = self.local_materials_db.search(
                result.analysis.pt_name,
                exact_only=exact_match_only,
                limit=5,
            )

        if not suggestions:
            return result

        existing_keys = {
            (quote.supplier.lower(), quote.url)
            for quote in result.quotes
        }

        new_quotes: List[SupplierQuote] = []
        for match in suggestions:
            supplier = match.supplier or "Локальная база"
            key = (supplier.lower(), match.url)
            if key in existing_keys:
                continue

            base_notes = match.notes or match.display_name
            display = match.display_name or "Материал" if match.display_name else "Материал"
            notes = f"{display} | {base_notes} | Источник: локальная база"
            quote = SupplierQuote(
                supplier=supplier,
                price=match.price or "N/A",
                url=match.url or "N/A",
                notes=notes.strip(" |"),
            )
            new_quotes.append(quote)
            existing_keys.add(key)

        if not new_quotes:
            return result

        result.quotes.extend(new_quotes)

        should_override = (
            result.best_offer.best_supplier == "N/A"
            or (exact_match_only and suggestions)
        )

        if should_override:
            top_quote = new_quotes[0]
            result.best_offer = BestOffer(
                best_supplier=top_quote.supplier,
                price=top_quote.price,
                url=top_quote.url,
                reasoning="Предложение из локальной базы материалов.",
            )

        logger.info(
            "Added %d local suggestions for '%s' (exact_only=%s)",
            len(new_quotes),
            query,
            exact_match_only,
        )

        return result
    
    def _wait_for_batch_slot(self) -> None:
        """Throttle batch execution to respect request delay settings."""
        min_interval = max(0.0, self.config.request_delay_seconds)
        if min_interval <= 0:
            return

        with self._batch_rate_lock:
            now = time.monotonic()
            last = self._last_batch_request_ts
            if last == 0.0:
                self._last_batch_request_ts = now
                return

            elapsed = now - last
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
                now = time.monotonic()
            self._last_batch_request_ts = now
    
    def materials_to_markdown(self, results: List[MaterialResult]) -> str:
        """Конвертировать результаты в Markdown таблицу."""
        if not results:
            return "No results"
        
        lines = [
            "| Material | PT Name | Supplier | Price | URL | Reasoning |",
            "|----------|---------|----------|-------|-----|-----------|",
        ]
        
        for r in results:
            lines.append(
                f"| {r.material_name} | {r.analysis.pt_name} | "
                f"{r.best_offer.best_supplier} | {r.best_offer.price} | "
                f"{r.best_offer.url} | {r.best_offer.reasoning} |"
            )
        
        return "\n".join(lines)
    
    # ========================================================================
    # GOOGLE SHEETS: Работа с таблицами через естественный язык
    # ========================================================================
    
    def process_sheets_command(self, command: str) -> str:
        """
        Обработать команду для работы с Google Sheets.

        Args:
            command: Команда на естественном языке

        Returns:
            Результат выполнения команды
        """
        SHEETS_COMMAND_REQUESTS.inc()
        with SHEETS_COMMAND_DURATION.time():
            if not self.sheets_ai:
                return "❌ Google Sheets AI not initialized. Set GOOGLE_SHEET_ID in .env"

            return self.sheets_ai.process_command(command)

    def chat_about_sheets(self, message: str, *, reset_history: bool = False) -> str:
        """Отправить сообщение в чат Google Sheets."""
        if not self.sheets_ai:
            return "❌ Google Sheets AI not initialized. Set GOOGLE_SHEET_ID in .env"
        return self.sheets_ai.process_command(message, reset=reset_history)

    def reset_sheets_chat(self) -> str:
        """Сбросить историю чата Google Sheets и вернуть системное сообщение."""
        if not self.sheets_ai:
            return "❌ Google Sheets AI not initialized. Set GOOGLE_SHEET_ID in .env"
        self.sheets_ai.reset_chat()
        system_message = "История чата Google Sheets сброшена."
        self.sheets_ai.append_system_message(system_message)
        return system_message

    def get_sheets_chat_history(self) -> List[Dict[str, str]]:
        """Получить историю чата Google Sheets."""
        if not self.sheets_ai:
            return []
        return self.sheets_ai.get_chat_history()
    
    def read_sheet_data(self, worksheet_name: Optional[str] = None) -> List[List[str]]:
        """Прочитать данные из Google Sheets."""
        if not self.sheets_ai:
            raise RuntimeError("Google Sheets AI not initialized")
        return self.sheets_ai.read_sheet_data(worksheet_name)

    def read_sheet_range(
        self,
        range_a1: str,
        worksheet_name: Optional[str] = None,
    ) -> List[List[str]]:
        """Прочитать данные из указанного диапазона листа."""
        if not self.sheets_ai:
            raise RuntimeError("Google Sheets AI not initialized")
        worksheet = self.sheets_ai._resolve_worksheet(worksheet_name)  # type: ignore[attr-defined]
        if not range_a1:
            return worksheet.get_all_values()
        return worksheet.get(range_a1)
    
    def write_sheet_data(
        self,
        data: List[List[str]],
        title: Optional[str] = None,
        worksheet_name: Optional[str] = None,
    ):
        """Записать данные в Google Sheets."""
        if not self.sheets_ai:
            raise RuntimeError("Google Sheets AI not initialized")
        self.sheets_ai.write_sheet_data(data, title, worksheet_name)

    def append_sheet_data(
        self,
        rows: List[List[str]],
        worksheet_name: Optional[str] = None,
    ):
        """Добавить строки в таблицу."""
        if not self.sheets_ai:
            raise RuntimeError("Google Sheets AI not initialized")
        self.sheets_ai.append_to_sheet(rows, worksheet_name=worksheet_name)

    def clear_sheet(self, worksheet_name: Optional[str] = None):
        """Очистить лист."""
        if not self.sheets_ai:
            raise RuntimeError("Google Sheets AI not initialized")
        self.sheets_ai.clear_sheet(worksheet_name)

    def update_sheet_cell(
        self,
        row: int,
        col: int,
        value: str,
        worksheet_name: Optional[str] = None,
    ):
        """Обновить конкретную ячейку (0-индексация)."""
        if not self.sheets_ai:
            raise RuntimeError("Google Sheets AI not initialized")
        self.sheets_ai.update_cell(row, col, value, worksheet_name)

    def format_sheet_range(
        self,
        range_a1: str,
        format_spec: Dict[str, Any],
        worksheet_name: Optional[str] = None,
    ) -> str:
        """Применить форматирование к диапазону."""
        if not self.sheets_ai:
            raise RuntimeError("Google Sheets AI not initialized")
        return self.sheets_ai.format_range(range_a1, format_spec, worksheet_name)

    def get_sheet_info(self, worksheet_name: Optional[str] = None) -> Dict[str, Any]:
        """Получить информацию о листе."""
        if not self.sheets_ai:
            raise RuntimeError("Google Sheets AI not initialized")
        return self.sheets_ai.get_table_info(worksheet_name)
    
    def setup_estimate_constructor(
        self,
        db_worksheet_name: str = "DB_Works",
        calc_sheet_name: str = "Estimate_Calculator",
    ) -> str:
        """Настроить лист конструктора сметы в Google Sheets."""
        if not self.sheets_ai:
            raise RuntimeError("Google Sheets AI not initialized")
        return self.sheets_ai.setup_estimate_constructor(
            db_worksheet_name=db_worksheet_name,
            calc_sheet_name=calc_sheet_name,
        )
    
    def fetch_web_content(self, url: str) -> str:
        """Загрузить контент с веб-страницы."""
        if not self.sheets_ai:
            raise RuntimeError("Google Sheets AI not initialized")
        return self.sheets_ai.fetch_web_content(url)
    
    def search_web(self, query: str) -> List[Dict[str, str]]:
        """Выполнить веб-поиск через SerpAPI."""
        if not self.sheets_ai:
            raise RuntimeError("Google Sheets AI not initialized")
        return self.sheets_ai.search_web(query)

    def _make_unique_sheet_title(self, base_title: str) -> str:
        """Подобрать уникальное имя листа в пределах таблицы."""
        if not self.sheets_ai:
            raise RuntimeError("Google Sheets is not configured")
        existing = {ws.title for ws in self.sheets_ai.spreadsheet.worksheets()}
        title = base_title
        counter = 1
        while title in existing:
            counter += 1
            title = f"{base_title} ({counter})"
        return title

    def _detect_language(self, *texts: str) -> str:
        """Простейшее определение языка по набору текстов."""
        joined = " ".join([t for t in texts if t]).lower()
        if any("а" <= ch <= "я" or "ё" == ch for ch in joined):
            return "ru"
        if any(token in joined for token in ("azulejo", "portuga", "mapei", "desmontagem", "terraço", "terraÃ§o")):
            return "pt"
        return "en"

    def _sheet_titles_for_language(self, lang: str) -> Dict[str, str]:
        if lang == "ru":
            return {"estimate": "Смета", "summary": "Резюме", "master": "Мастер-лист"}
        if lang == "en":
            return {"estimate": "Estimate", "summary": "Summary", "master": "Master List"}
        return {"estimate": "Orçamento", "summary": "Resumo", "master": "Master List"}

    def _labels_for_language(self, lang: str) -> Dict[str, str]:
        """Заголовки и подписи под язык пользователя."""
        if lang == "ru":
            return {
                "brand": "VIORA BUILD",
                "works_title": "ОПИСАНИЕ РАБОТ",
                "section": "РАБОТЫ ПО ДЕМОНТАЖУ",
                "col_num": "№",
                "col_desc": "ОПИСАНИЕ РАБОТ",
                "col_unit": "Ед.",
                "col_qty": "Кол-во",
                "col_price": "Цена за ед.",
                "col_total": "Итого",
                "notes_header": "Примечания",
                "vat_label": "НДС",
                "total_with_vat_label": "ИТОГО с НДС",
            }
        if lang == "en":
            return {
                "brand": "VIORA BUILD",
                "works_title": "SCOPE OF WORK",
                "section": "DEMOLITION WORKS",
                "col_num": "No.",
                "col_desc": "DESCRIPTION OF WORKS",
                "col_unit": "Unit",
                "col_qty": "Qty",
                "col_price": "Unit Price",
                "col_total": "Total",
                "notes_header": "Notes",
                "vat_label": "VAT",
                "total_with_vat_label": "TOTAL WITH VAT",
            }
        # default Portuguese
        return {
            "brand": "VIORA BUILD",
            "works_title": "DESCRIÇÃO DOS TRABALHOS",
            "section": "TRABALHOS DE DESMONTAGEM",
            "col_num": "№",
            "col_desc": "DESCRIÇÃO DOS TRABALHOS",
            "col_unit": "Unid.",
            "col_qty": "Qtd.",
            "col_price": "Preço por Un",
            "col_total": "Valor Total",
            "notes_header": "Notas",
            "vat_label": "IVA",
            "total_with_vat_label": "TOTAL COM IVA",
        }

    def _to_float(self, value: Any) -> float:
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        try:
            cleaned = (
                str(value)
                .replace("€", "")
                .replace(" ", "")
                .replace("\u00a0", "")
                .replace(",", ".")
            )
            return float(cleaned)
        except Exception:
            return 0.0

    def _format_currency(self, value: float, currency: str = "€") -> str:
        return f"{currency}{value:,.2f}".replace(",", " ").replace(".", ",", 1)

    def _fallback_estimate_table(self, text_input: str, variant_label: str, language: str) -> Dict[str, Any]:
        """Запасной вариант генерации таблицы сметы, когда LLM недоступен."""
        uses_new_tile = "нов" in text_input.lower() or "new" in text_input.lower() or "nova" in text_input.lower()

        if language == "ru":
            items = [
                {"number": "1.1", "description": "Демонтаж существующей плитки (25x40), толщина 2,5 см", "unit": "м2", "quantity": 640, "unit_price": 10},
                {"number": "1.2", "description": "Демонтаж старых опорных ножек для плитки", "unit": "м2", "quantity": 640, "unit_price": 3},
                {"number": "1.3", "description": "Мойка/очистка террасы", "unit": "м2", "quantity": 640, "unit_price": 3},
                {"number": "1.4", "description": "Нанесение гидроизоляции MAPEI (MAPE LASTIC) в 2 слоя", "unit": "м2", "quantity": 700, "unit_price": 12},
                {"number": "1.4.1", "description": "1 слой с армирующей сеткой", "unit": "м2", "quantity": 700, "unit_price": 8.56},
                {"number": "1.4.2", "description": "1 слой без сетки", "unit": "м2", "quantity": 700, "unit_price": 3.44},
                {"number": "1.4.3", "description": "Mapelastic Smart Kit — Mapei", "unit": "м2", "quantity": 700, "unit_price": 8.56},
                {"number": "1.7", "description": "Установка новых регулируемых опор для плитки", "unit": "м2", "quantity": 640, "unit_price": 25},
                {"number": "1.8", "description": "Регулируемые опоры (средняя цена рынка)", "unit": "шт", "quantity": 6400, "unit_price": 3.85},
            ]
            notes = [
                "По расчёту 10 опор/м² нужно 6 400 опор. Финальная цена уточняется у поставщика.",
                "Все доп. работы и материалы считаются отдельно. НДС не включён.",
            ]
            if uses_new_tile:
                items.append({"number": "1.9", "description": "Утилизация старой плитки", "unit": "м2", "quantity": 640, "unit_price": 4})
                items.append({"number": "1.10", "description": "Поставка и укладка новой плитки", "unit": "м2", "quantity": 640, "unit_price": 35})
                notes.append("Вариант с новой плиткой: учтена утилизация старой плитки и укладка новой.")
        else:
            items = [
                {"number": "1.1", "description": "Desmontagem do azulejo existente (25x40), espessura 2,5cm", "unit": "m2", "quantity": 640, "unit_price": 10},
                {"number": "1.2", "description": "Desmontagem dos pés de apoio antigos para azulejo", "unit": "m2", "quantity": 640, "unit_price": 3},
                {"number": "1.3", "description": "Limpeza do terraço", "unit": "m2", "quantity": 640, "unit_price": 3},
                {"number": "1.4", "description": "Aplicação de impermeabilização MAPEI (MAPE LASTIC) em 2 camadas", "unit": "m2", "quantity": 700, "unit_price": 12},
                {"number": "1.4.1", "description": "1 camada com malha", "unit": "m2", "quantity": 700, "unit_price": 8.56},
                {"number": "1.4.2", "description": "1 camada sem malha", "unit": "m2", "quantity": 700, "unit_price": 3.44},
                {"number": "1.4.3", "description": "Mapelastic Smart Kit - Mapei", "unit": "m2", "quantity": 700, "unit_price": 8.56},
                {"number": "1.7", "description": "Colocação dos novos pés para azulejo", "unit": "m2", "quantity": 640, "unit_price": 25},
                {"number": "1.8", "description": "Novos pés telescópicos auto-nivelantes (média de mercado)", "unit": "un", "quantity": 6400, "unit_price": 3.85},
            ]
            notes = [
                "Novos pés. De acordo com o consumo de 10 unidades/m², a quantidade necessária é de 6.400 unidades. Preço será confirmado com o fornecedor.",
                "Todos os trabalhos e materiais adicionados serão calculados отдельно. IVA não включен.",
            ]
            if uses_new_tile:
                items.append({"number": "1.9", "description": "Remoção e eliminação do azulejo antigo", "unit": "m2", "quantity": 640, "unit_price": 4})
                items.append({"number": "1.10", "description": "Fornecimento e colocação de novo azulejo", "unit": "m2", "quantity": 640, "unit_price": 35})
                notes.append("Segundo variante: inclui eliminação do azulejo antigo e colocação de novo.")

        currency = "€"
        for item in items:
            qty = self._to_float(item.get("quantity"))
            price = self._to_float(item.get("unit_price"))
            item["total"] = qty * price

            labels = self._labels_for_language(language)
            return {
                "section_title": labels["section"],
                "currency": currency,
                "iva_percent": 23,
                "items": items,
                "notes": notes,
                "variant": variant_label,
                "labels": labels,
            }

    def _generate_estimate_table_data(
        self,
        name: str,
        description: str,
        client_name: str,
        text_input: str,
        variant_label: str,
    ) -> Dict[str, Any]:
        """Использовать LLM, чтобы превратить текст задания в структурированную таблицу сметы."""
        language = self._detect_language(name, description, client_name, text_input)
        reference_hint = self._load_reference_snippets()
        if not self.openai_client:
            return self._fallback_estimate_table(text_input, variant_label, language)

        system_prompt = """
Ты сметчик (PT/EN/ES допускается), оформляешь коммерческое предложение для укладки плитки на террасе.
Нужно вернуть JSON без пояснений вида:
{
  "section_title": "TRABALHOS DE DESMONTAGEM",
  "currency": "€",
  "iva_percent": 23,
  "items": [
     {"number": "1.1", "description": "...", "unit": "m2", "quantity": 640, "unit_price": 10},
     ...
  ],
  "notes": ["краткие примечания"]
}
Требования:
- Нумерация в стиле 1 / 1.1 / 1.2 (как на строительных сметах).
- Единицы: m2 для площадей, un для штук.
- Заполни quantity и unit_price (евро) осознанно; округляй до 2 знаков.
- total не обязателен — если не укажешь, мы посчитаем qty*price.
- Добавь 1–3 примечания (notes) про доп. работы, исключения, подтверждение цен у поставщика.
- Используй тот же язык, что и вход (не переводить таблицу).
"""
        user_prompt = f"""
Смета: {name}
Клиент: {client_name or '—'}
Описание: {description or '—'}
Вариант: {variant_label}
Язык: тот же, что в тексте ниже (детектируй автоматически)
Задача/вход: {text_input}

Сделай структуру, похожую на пример Viora Build: колонок 6 (№, DESCRIÇÃO, Unid., Qtd., Preço por Un, Valor Total).
Референсы (ориентируйся на их стиль и нумерацию, но не копируй цены):
{reference_hint}
"""
        try:
            completion = self.openai_client.chat.completions.create(
                model=self.config.llm_model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = completion.choices[0].message.content
            data = json.loads(content)
            if not isinstance(data, dict) or "items" not in data:
                raise ValueError("Unexpected LLM response")
            data["labels"] = self._labels_for_language(language)
            return data
        except Exception as exc:
            logger.warning("LLM estimate generation failed, using fallback: %s", exc)
            return self._fallback_estimate_table(text_input, variant_label, language)

    def _load_reference_snippets(self, max_chars: int = 4000) -> str:
        """Считать текстовые референсы для стиля смет."""
        if not REFERENCE_DIR.exists():
            return ""
        snippets: List[str] = []
        remaining = max_chars
        for path in sorted(REFERENCE_DIR.glob("*")):
            if remaining <= 0:
                break
            if path.suffix.lower() in {".txt", ".md", ".csv", ".tsv"}:
                try:
                    text = path.read_text(errors="ignore")
                    text = text.strip()
                except Exception:
                    continue
            elif path.suffix.lower() in {".xlsx", ".xls"}:
                try:
                    import pandas as pd  # type: ignore

                    df = pd.read_excel(path)
                    text = df.to_csv(index=False)
                except Exception:
                    continue
            else:
                continue
            if not text:
                continue
            slice_text = text[:remaining]
            snippets.append(f"[{path.name}]\n{slice_text}")
            remaining -= len(slice_text)
        return "\n\n".join(snippets)

    def _prepare_worksheet(self, sheets_ai: GoogleSheetsAI, title: str, rows: int, cols: int) -> gspread.Worksheet:
        """Получить/создать лист и очистить его."""
        try:
            worksheet = sheets_ai.spreadsheet.worksheet(title)
            worksheet.clear()
        except WorksheetNotFound:
            worksheet = sheets_ai.spreadsheet.add_worksheet(title=title, rows=rows, cols=cols)
        return worksheet

    def _build_estimate_rows(
        self,
        items: List[Dict[str, Any]],
        labels: Dict[str, str],
        iva_percent: float,
        section_title: str,
        notes: Optional[List[str]] = None,
    ) -> tuple[List[List[Any]], int, int, int]:
        """Сформировать строки таблицы сметы в стандартном формате."""
        rows: List[List[Any]] = []
        rows.append(["", "", "", "", "", ""])
        rows.append([labels["brand"], "", "", "", "", ""])
        rows.append([labels["works_title"], "", "", "", "", ""])
        rows.append([labels["col_num"], labels["col_desc"], labels["col_unit"], labels["col_qty"], labels["col_price"], labels["col_total"]])

        rows.append(["1", section_title, "", "", "", ""])

        first_item_row = len(rows) + 1
        current_row_index = first_item_row
        for item in items:
            qty = self._to_float(item.get("quantity"))
            price = self._to_float(item.get("unit_price"))
            rows.append([
                item.get("number") or "",
                item.get("description") or item.get("name") or "",
                item.get("unit") or "",
                qty if qty else "",
                price if price else "",
                f"=D{current_row_index}*E{current_row_index}",
            ])
            current_row_index += 1

        rows.append(["", "", "", "", "", ""])
        subtotal_row = len(rows) + 1
        last_item_row = max(first_item_row, current_row_index - 1)
        rows.append(["", "TOTAL", "", "", "", f"=SUM(F{first_item_row}:F{last_item_row})"])
        vat_row = len(rows) + 1
        rows.append(["", f"{labels.get('vat_label', 'IVA')} {iva_percent:.0f}%", "", "", "", f"=F{subtotal_row}*{iva_percent/100:.4f}"])
        total_with_vat_row = len(rows) + 1
        rows.append(["", labels.get("total_with_vat_label", "TOTAL COM IVA"), "", "", "", f"=F{subtotal_row}+F{vat_row}"])

        if notes:
            rows.append(["", "", "", "", "", ""])
            for note in notes:
                rows.append(["", note, "", "", "", ""])

        return rows, first_item_row, last_item_row, total_with_vat_row

    def _apply_estimate_sheet_formatting(
        self,
        sheets_ai: GoogleSheetsAI,
        worksheet: gspread.Worksheet,
        items_count: int,
        rows_len: int,
    ) -> None:
        """Применить фирменное форматирование к листу сметы."""
        try:
            header_color = {"red": 0.043, "green": 0.188, "blue": 0.18}
            sheets_ai.spreadsheet.batch_update({
                "requests": [
                    {
                        "mergeCells": {
                            "range": {
                                "sheetId": worksheet.id,
                                "startRowIndex": 1,
                                "endRowIndex": 2,
                                "startColumnIndex": 0,
                                "endColumnIndex": 6,
                            },
                            "mergeType": "MERGE_ALL"
                        }
                    },
                    {
                        "mergeCells": {
                            "range": {
                                "sheetId": worksheet.id,
                                "startRowIndex": 2,
                                "endRowIndex": 3,
                                "startColumnIndex": 0,
                                "endColumnIndex": 6,
                            },
                            "mergeType": "MERGE_ALL"
                        }
                    },
                ]
            })
            sheets_ai.format_range(
                "A2:F3",
                {"backgroundColor": header_color, "textColor": {"red": 1, "green": 1, "blue": 1}, "bold": True, "alignment": "CENTER", "fontFamily": "Calibri"},
                worksheet_name=worksheet.title,
            )
            sheets_ai.format_range(
                "A4:F4",
                {"backgroundColor": header_color, "textColor": {"red": 1, "green": 1, "blue": 1}, "bold": True, "alignment": "CENTER", "fontFamily": "Calibri"},
                worksheet_name=worksheet.title,
            )
            sheets_ai.format_range(
                "A5:F5",
                {"backgroundColor": header_color, "textColor": {"red": 1, "green": 1, "blue": 1}, "bold": True, "fontFamily": "Calibri"},
                worksheet_name=worksheet.title,
            )
            total_start = items_count + 7
            sheets_ai.format_range(
                f"E{total_start}:F{total_start+2}",
                {"bold": True, "fontFamily": "Calibri"},
                worksheet_name=worksheet.title,
            )
            sheets_ai.format_range(
                "B:B",
                {"wrapStrategy": "WRAP", "fontFamily": "Calibri"},
                worksheet_name=worksheet.title,
            )
            sheets_ai.format_range(
                f"A1:F{rows_len}",
                {"fontFamily": "Calibri"},
                worksheet_name=worksheet.title,
            )
            sheets_ai.format_range(
                f"A5:F{rows_len}",
                {"borders": {"style": "SOLID", "color": {"red": 0, "green": 0, "blue": 0}}},
                worksheet_name=worksheet.title,
            )
        except Exception as exc:
            logger.warning("Failed to format estimate sheet: %s", exc)

    def _create_estimate_sheet(
        self,
        sheet_title: str,
        name: str,
        description: str,
        client_name: str,
        text_input: str,
        sheets_ai: GoogleSheetsAI,
    ) -> Dict[str, Any]:
        """Создать лист Google Sheets с оформленной сметой."""
        if not sheets_ai:
            raise RuntimeError("Google Sheets AI is not configured for this spreadsheet")

        language = self._detect_language(name, description, client_name, text_input)
        table_data = self._generate_estimate_table_data(
            name=name,
            description=description,
            client_name=client_name,
            text_input=text_input,
            variant_label=sheet_title,
        )
        labels = table_data.get("labels") or self._labels_for_language(language)
        items: List[Dict[str, Any]] = table_data.get("items") or []
        notes: List[str] = table_data.get("notes") or []
        iva_percent = self._to_float(table_data.get("iva_percent") or 23)

        section_title = table_data.get("section_title") or labels["section"]
        rows, first_item_row, last_item_row, total_with_vat_row = self._build_estimate_rows(
            items=items,
            labels=labels,
            iva_percent=iva_percent,
            section_title=section_title,
            notes=notes,
        )

        row_count = max(40, len(rows) + 15)
        worksheet = self._prepare_worksheet(sheets_ai, sheet_title, row_count, 8)
        worksheet.update("A1", rows, value_input_option="USER_ENTERED")

        self._apply_estimate_sheet_formatting(
            sheets_ai=sheets_ai,
            worksheet=worksheet,
            items_count=len(items),
            rows_len=len(rows),
        )

        gid = getattr(worksheet, "id", None) or worksheet._properties.get("sheetId")
        sheet_url = f"https://docs.google.com/spreadsheets/d/{sheets_ai.spreadsheet.id}/edit#gid={gid}"

        return {
            "title": sheet_title,
            "url": sheet_url,
            "rows": len(rows),
            "subtotal_formula": f"=SUM(F{first_item_row}:F{last_item_row})",
            "total_with_iva_cell": f"F{total_with_vat_row}",
            "sheet": sheet_title,
        }

    def _create_estimate_sheet_from_items(
        self,
        sheet_title: str,
        estimate: "Estimate",
        labels: Dict[str, str],
        sheets_ai: GoogleSheetsAI,
    ) -> Dict[str, Any]:
        """Создать лист сметы по уже готовым позициям (после импорта Excel)."""
        items_data: List[Dict[str, Any]] = []
        for idx, item in enumerate(estimate.items, start=1):
            items_data.append(
                {
                    "number": str(idx),
                    "description": item.description or item.name,
                    "unit": item.unit,
                    "quantity": item.quantity,
                    "unit_price": item.unit_price,
                }
            )

        iva_percent = estimate.metadata.tax_percent or 23
        rows, first_item_row, last_item_row, total_with_vat_row = self._build_estimate_rows(
            items=items_data,
            labels=labels,
            iva_percent=iva_percent,
            section_title=labels["section"],
            notes=[estimate.metadata.notes] if estimate.metadata.notes else None,
        )
        row_count = max(40, len(rows) + 15)
        worksheet = self._prepare_worksheet(sheets_ai, sheet_title, row_count, 8)
        worksheet.update("A1", rows, value_input_option="USER_ENTERED")

        self._apply_estimate_sheet_formatting(
            sheets_ai=sheets_ai,
            worksheet=worksheet,
            items_count=len(items_data),
            rows_len=len(rows),
        )

        gid = getattr(worksheet, "id", None) or worksheet._properties.get("sheetId")
        sheet_url = f"https://docs.google.com/spreadsheets/d/{sheets_ai.spreadsheet.id}/edit#gid={gid}"

        return {
            "title": sheet_title,
            "url": sheet_url,
            "rows": len(rows),
            "subtotal_formula": f"=SUM(F{first_item_row}:F{last_item_row})",
            "total_with_iva_cell": f"F{total_with_vat_row}",
            "sheet": sheet_title,
        }

    def _create_summary_sheet(
        self,
        title: str,
        estimate_sheet_title: str,
        labels: Dict[str, str],
        total_cell: str,
        sheets_ai: GoogleSheetsAI,
    ) -> Dict[str, Any]:
        """Создать лист-резюме с итогами."""
        worksheet = self._prepare_worksheet(sheets_ai, title, 20, 6)
        rows = [
            [labels["brand"], "", "", "", "", ""],
            [labels["works_title"], "", "", "", "", ""],
            ["", labels.get("total_with_vat_label", "TOTAL COM IVA"), "", "", "", f"='{estimate_sheet_title}'!{total_cell}"],
        ]
        worksheet.update("A1", rows, value_input_option="USER_ENTERED")
        try:
            header_color = {"red": 0.043, "green": 0.188, "blue": 0.18}
            sheets_ai.format_range(
                "A1:F2",
                {"backgroundColor": header_color, "textColor": {"red": 1, "green": 1, "blue": 1}, "bold": True, "fontFamily": "Calibri"},
                worksheet_name=worksheet.title,
            )
            sheets_ai.format_range(
                "A1:F20",
                {"fontFamily": "Calibri"},
                worksheet_name=worksheet.title,
            )
        except Exception as exc:
            logger.warning("Failed to format summary sheet: %s", exc)
        gid = getattr(worksheet, "id", None) or worksheet._properties.get("sheetId")
        url = f"https://docs.google.com/spreadsheets/d/{sheets_ai.spreadsheet.id}/edit#gid={gid}"
        return {"title": title, "url": url, "rows": len(rows), "sheet": title}

    def _create_master_sheet(self, title: str, labels: Dict[str, str], sheets_ai: GoogleSheetsAI) -> Dict[str, Any]:
        """Создать мастер-лист с заготовкой."""
        worksheet = self._prepare_worksheet(sheets_ai, title, 100, 6)
        rows = [
            [labels["col_num"], labels["col_desc"], labels["col_unit"], labels["col_qty"], labels["col_price"], labels["col_total"]],
        ]
        worksheet.update("A1", rows, value_input_option="USER_ENTERED")
        try:
            header_color = {"red": 0.043, "green": 0.188, "blue": 0.18}
            sheets_ai.format_range(
                "A1:F1",
                {"backgroundColor": header_color, "textColor": {"red": 1, "green": 1, "blue": 1}, "bold": True, "fontFamily": "Calibri"},
                worksheet_name=worksheet.title,
            )
            sheets_ai.format_range(
                "A1:F100",
                {"fontFamily": "Calibri"},
                worksheet_name=worksheet.title,
            )
        except Exception as exc:
            logger.warning("Failed to format master sheet: %s", exc)
        gid = getattr(worksheet, "id", None) or worksheet._properties.get("sheetId")
        url = f"https://docs.google.com/spreadsheets/d/{sheets_ai.spreadsheet.id}/edit#gid={gid}"
        return {"title": title, "url": url, "rows": 100, "sheet": title}

    # ========================================================================
    # СМЕТА: Создание смет через конструктор
    # ========================================================================

    def create_estimate(
        self,
        name: str,
        description: str = "",
        text_input: str = "",
        client_name: str = "",
        auto_find_prices: bool = True,
        create_sheet: bool = False,
        worksheet_name: Optional[str] = None,
        double_variant: bool = False,
    ) -> Dict[str, Any]:
        """
        Создать новую смету на основе текстового описания.

        Args:
            name: Название сметы (обязательное)
            description: Описание проекта
            text_input: Текст с перечнем работ/материалов
            client_name: Имя клиента
            auto_find_prices: Подтягивать ли цены автоматически
            create_sheet: Создать новый лист Google Sheets с оформленной сметой
            worksheet_name: Базовое название листа (если создаём таблицу)
            double_variant: Создать вторую версию сметы (например, с новой плиткой)

        Returns:
            Словарь с данными созданной сметы
        """
        if not self.estimate_constructor:
            raise RuntimeError("Estimate constructor not available")

        if not name:
            name = f"Estimate {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        if not description:
            description = (text_input[:120] + "...") if text_input else "Смета создана автоматически"
        if not client_name:
            client_name = "Клиент не указан"

        estimate = create_quick_estimate(
            constructor=self.estimate_constructor,
            name=name,
            description=description or "",
            text_input=text_input or "",
            client_name=client_name or "",
            auto_find_prices=auto_find_prices,
        )

        result: Dict[str, Any] = {"estimate": estimate.to_dict()}

        if create_sheet:
            base_title = (worksheet_name or name or "Estimate").strip() or "Estimate"
            language = self._detect_language(name, description, client_name, text_input)
            sheet_titles = self._sheet_titles_for_language(language)
            variants = [("Опция 1: текущая плитка", text_input)]
            if double_variant:
                variants.append((
                    "Опция 2: новая плитка",
                    f"{text_input}\nДобавь утилизацию старой плитки и укладку новой плитки вместо старой.",
                ))

            sheet_links: List[Dict[str, Any]] = []
            gspread_client = self._ensure_gspread_client()

            for suffix, variant_text in variants:
                spreadsheet_title = f"{base_title} — {suffix}"
                try:
                    spreadsheet = gspread_client.create(spreadsheet_title)
                    spreadsheet.share(None, perm_type="anyone", role="reader")
                    temp_ai = GoogleSheetsAI(
                        sheet_id=spreadsheet.id,
                        openai_client=self.openai_client,
                        llm_model=self.config.llm_model,
                    )
                except APIError as exc:
                    logger.warning("Не удалось создать таблицу '%s': %s. Используем текущую таблицу.", spreadsheet_title, exc)
                    if not self.sheets_ai:
                        raise
                    spreadsheet = self.sheets_ai.spreadsheet
                    temp_ai = self.sheets_ai
                estimate_sheet_title = f"{sheet_titles['estimate']} — {suffix}"
                estimate_info = self._create_estimate_sheet(
                    sheet_title=estimate_sheet_title,
                    name=name,
                    description=description,
                    client_name=client_name,
                    text_input=variant_text,
                    sheets_ai=temp_ai,
                )
                summary_sheet_title = f"{sheet_titles['summary']} — {suffix}"
                master_sheet_title = f"{sheet_titles['master']} — {suffix}"
                labels = self._labels_for_language(language)
                summary_info = self._create_summary_sheet(
                    title=summary_sheet_title,
                    estimate_sheet_title=estimate_sheet_title,
                    labels=labels,
                    total_cell=estimate_info["total_with_iva_cell"],
                    sheets_ai=temp_ai,
                )
                master_info = self._create_master_sheet(
                    title=master_sheet_title,
                    labels=labels,
                    sheets_ai=temp_ai,
                )

                spreadsheet_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet.id}"
                for info, info_type in (
                    (summary_info, "summary"),
                    (estimate_info, "estimate"),
                    (master_info, "master"),
                ):
                    info["type"] = info_type
                    info["spreadsheet_id"] = spreadsheet.id
                    info["spreadsheet_title"] = spreadsheet.title
                    info["spreadsheet_url"] = spreadsheet_url

                sheet_links.extend([summary_info, estimate_info, master_info])

            result["sheets"] = sheet_links

        return result

    def import_estimate_from_excel(
        self,
        path: str,
        sheet_name: Optional[str] = None,
        column_map: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
        description: str = "",
        client_name: str = "",
        currency: str = "€",
        skip_rows: int = 0,
        default_item_type: ItemType | str = ItemType.WORK,
        create_sheet: bool = False,
        worksheet_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Импортировать Excel-файл со сметой в наш формат.

        Args:
            path: путь к Excel.
            sheet_name: название листа (если None — первый лист).
            column_map: ручное сопоставление колонок {"description": "Описание работ", ...}.
            name: имя сметы (по умолчанию имя файла).
            description: описание проекта.
            client_name: имя клиента.
            currency: валюта.
            skip_rows: пропустить первые N строк перед чтением заголовков.
            default_item_type: тип позиций (work/material/...).
            create_sheet: создать лист Google Sheets в фирменном формате.
            worksheet_name: базовое название листов при создании в Google Sheets.
        """
        if not self.estimate_importer:
            raise RuntimeError("Estimate importer not available")

        estimate = self.estimate_importer.import_from_excel(
            path=path,
            sheet_name=sheet_name,
            column_map=column_map,
            name=name,
            description=description,
            client_name=client_name,
            currency=currency,
            skip_rows=skip_rows,
            default_item_type=default_item_type,
        )

        result: Dict[str, Any] = {"estimate": estimate.to_dict()}

        if create_sheet:
            if not self.sheets_ai:
                raise RuntimeError("Google Sheets AI is not configured for this spreadsheet")
            language = self._detect_language(
                estimate.metadata.name,
                estimate.metadata.description,
                estimate.metadata.client_name,
                "",
            )
            labels = self._labels_for_language(language)
            sheet_titles = self._sheet_titles_for_language(language)
            base_title = (worksheet_name or estimate.metadata.name or "Estimate").strip() or "Estimate"
            estimate_sheet_title = f"{sheet_titles['estimate']} — {base_title}"
            summary_sheet_title = f"{sheet_titles['summary']} — {base_title}"
            master_sheet_title = f"{sheet_titles['master']} — {base_title}"

            estimate_info = self._create_estimate_sheet_from_items(
                sheet_title=estimate_sheet_title,
                estimate=estimate,
                labels=labels,
                sheets_ai=self.sheets_ai,
            )
            summary_info = self._create_summary_sheet(
                title=summary_sheet_title,
                estimate_sheet_title=estimate_sheet_title,
                labels=labels,
                total_cell=estimate_info["total_with_iva_cell"],
                sheets_ai=self.sheets_ai,
            )
            master_info = self._create_master_sheet(
                title=master_sheet_title,
                labels=labels,
                sheets_ai=self.sheets_ai,
            )

            sheet_links: List[Dict[str, Any]] = []
            spreadsheet = self.sheets_ai.spreadsheet
            spreadsheet_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet.id}"
            for info, info_type in (
                (summary_info, "summary"),
                (estimate_info, "estimate"),
                (master_info, "master"),
            ):
                info["type"] = info_type
                info["spreadsheet_id"] = spreadsheet.id
                info["spreadsheet_title"] = spreadsheet.title
                info["spreadsheet_url"] = spreadsheet_url
                sheet_links.append(info)

            result["sheets"] = sheet_links

        return result

    def estimate_assistant_reply(self, messages: List[Dict[str, str]]) -> str:
        """ИИ-помощник для пошаговой проработки сметы."""
        if not messages:
            raise ValueError("Нет сообщений для ассистента")
        if not self.openai_client:
            raise RuntimeError("LLM недоступен")

        system_prompt = """
Ты опытный сметчик по отделке/плитке. Помогаешь шаг за шагом собрать вводные и предложить структуру сметы в стиле примера:
- Табличные поля: №, DESCRIÇÃO DOS TRABALHOS, Unid., Qtd., Preço por Un, Valor Total.
- Любишь португальские заголовки типа TRABALHOS DE DESMONTAGEM.
- Всегда уточняешь недостающие данные (площадь, количество, марка материалов, нужна ли утилизация старых материалов, НДС).
- Формулируешь ответы кратко, списками; если готов итог, перечисляешь позиции с нумерацией.
"""
        chat_messages = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            role = msg.get("role") or "user"
            content = msg.get("content") or ""
            chat_messages.append({"role": role, "content": content})

        completion = self.openai_client.chat.completions.create(
            model=self.config.llm_model,
            messages=chat_messages,
        )
        return completion.choices[0].message.content
    
    # ========================================================================
    # СМЕТА: Проверка строительных смет
    # ========================================================================
    
    def check_estimate(
        self,
        estimate_sheet: str = None,
        master_sheet: str = "Master List",
        quantity_col: str = "F",
    ) -> str:
        """
        Проверить строительную смету.
        
        Args:
            estimate_sheet: Название листа со сметой
            master_sheet: Название листа с мастер-листом
            quantity_col: Колонка с количеством
        
        Returns:
            Отчет о проверке
        """
        ESTIMATE_CHECK_REQUESTS.inc()
        with ESTIMATE_CHECK_DURATION.time():
            if not self.sheets_ai or not self.sheets_ai.estimate_checker:
                return "❌ Estimate checker not available"

            estimate_sheet = estimate_sheet or self.config.google_worksheet_name

            try:
                estimate_worksheet = self.sheets_ai.spreadsheet.worksheet(estimate_sheet)
                master_worksheet = self.sheets_ai.spreadsheet.worksheet(master_sheet)

                estimate_data = estimate_worksheet.get_all_values()
                master_data = master_worksheet.get_all_values()

                result = self.sheets_ai.estimate_checker.validate_estimate(
                    estimate_data,
                    master_data,
                    quantity_col,
                )

                return self.sheets_ai.estimate_checker.format_validation_report(result)

            except Exception as e:
                return f"❌ Error checking estimate: {e}"
    
    # ========================================================================
    # УНИВЕРСАЛЬНЫЙ ИНТЕРФЕЙС: Обработка любых команд
    # ========================================================================
    
    def process_command(self, command: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Универсальный обработчик команд.
        
        Автоматически определяет тип команды и выбирает подходящий метод:
        - Поиск материалов
        - Работа с Google Sheets
        - Проверка смет
        - Веб-поиск
        - Помощник базы материалов и работ
       
        Args:
            command: Команда на естественном языке
            context: Дополнительный контекст
        
        Returns:
            Результат выполнения
        """
        logger.info("Processing command: %s", command)
        
        # Определение типа команды через LLM
        system_prompt = """Ты - классификатор команд для AI-агента.
Определи тип команды и верни JSON со структурой:
{
  "command_type": "MATERIAL_SEARCH|SHEETS_OPERATION|ESTIMATE_CHECK|WEB_SEARCH|PROJECT_CHAT|MATERIAL_DB_ASSISTANT|GENERAL",
  "parameters": {
    // Для MATERIAL_SEARCH
    "material_name": "строка или null",
    "use_cache": true|false|null,
    "use_scraping": true|false|null,
    "use_advanced_search": true|false|null,
    // Для SHEETS_OPERATION
    "action": "read|write|append|analyze|... или null",
    "worksheet": "название листа или null",
    "range": "диапазон, если указан, или null",
    "title": "новый заголовок или null",
    "data": [[...]] или null,
    // Для ESTIMATE_CHECK
    "estimate_sheet": "лист сметы или null",
    "master_sheet": "лист эталона или null",
    "quantity_col": "буква колонки или null",
    // Для WEB_SEARCH
    "query": "поисковый запрос или null",
    // Для PROJECT_CHAT
    "topic": "тема обсуждения или null",
    // Для MATERIAL_DB_ASSISTANT
    "intent": "lookup|stage_update|stage_insert|stage_delete|confirm_change|cancel_change|list_pending|help|null",
    "record_id": "ID записи или null",
    "db_query": "поисковый запрос или null",
    "confirmation_id": "ID подготовленного изменения или null"
  },
  "explanation": "краткое объяснение выбранного типа"
}

Правила:
- Если информация явно не указана, ставь null.
- Используй PROJECT_CHAT, если пользователь хочет обсудить проект, его архитектуру, файлы или задать общий вопрос по коду.
- Используй MATERIAL_DB_ASSISTANT для запросов, связанных с локальной базой материалов и работ (поиск, подготовка изменений, подтверждение/отмена изменений).
- Не добавляй новые типы команд и не включай комментарии в JSON.
- Всегда возвращай валидный JSON-объект."""
        
        try:
            classification_schema = {
                "type": "object",
                "properties": {
                    "command_type": {"type": "string"},
                    "parameters": {"type": "object", "additionalProperties": True},
                    "explanation": {"type": "string"},
                },
                "required": ["command_type"],
                "additionalProperties": True,
            }

            response = self.openai_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": command},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "command_classification",
                        "schema": classification_schema,
                    },
                },
            )
            
            content = response.choices[0].message.content if response.choices else ""
            classification = self._safe_parse_classification(content)
            command_type = classification.get("command_type", "GENERAL")
            params = classification.get("parameters") or {}
            
            logger.info("Command classified as: %s", command_type)
            
            if command_type == "PROJECT_CHAT":
                return self._handle_project_chat(params, command, classification, context)
            
            handlers = {
                "MATERIAL_SEARCH": self._handle_material_search,
                "SHEETS_OPERATION": self._handle_sheets_operation,
                "ESTIMATE_CHECK": self._handle_estimate_check,
                "WEB_SEARCH": self._handle_web_search,
                "MATERIAL_DB_ASSISTANT": self._handle_materials_db,
                "GENERAL": self._handle_general,
            }
            handler = handlers.get(command_type, self._handle_general)
            return handler(params, command, classification)
        
        except Exception as e:
            logger.error("Error processing command: %s", e)
            return f"❌ Ошибка обработки команды: {e}"
    
    # ========================================================================
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ========================================================================
    
    def _safe_parse_classification(self, content: Optional[str]) -> Dict[str, Any]:
        """Безопасно распарсить JSON классификации команды."""
        default = {"command_type": "GENERAL", "parameters": {}, "explanation": ""}
        if not content:
            logger.warning("Empty classification response from LLM")
            return default
        
        try:
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise ValueError("classification must be a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Failed to parse classification JSON: %s; raw=%s", exc, content)
            return default
        
        if "parameters" not in parsed or not isinstance(parsed["parameters"], dict):
            parsed["parameters"] = {}
        if "command_type" not in parsed or not isinstance(parsed["command_type"], str):
            parsed["command_type"] = default["command_type"]
        if "explanation" not in parsed or not isinstance(parsed["explanation"], str):
            parsed["explanation"] = default["explanation"]
        return parsed
    
    @staticmethod
    def _normalize_bool(value: Any, default: bool) -> bool:
        """Преобразовать значение к булеву типу."""
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            truthy = {"true", "1", "yes", "y", "да", "ok", "on"}
            falsy = {"false", "0", "no", "n", "нет", "off"}
            if normalized in truthy:
                return True
            if normalized in falsy:
                return False
        return default
    
    @staticmethod
    def _normalize_action(action: str) -> str:
        """Привести действие для Google Sheets к каноническому виду."""
        mapping = {
            "прочитай": "READ",
            "прочитать": "READ",
            "читать": "READ",
            "чтение": "READ",
            "читатьлист": "READ",
            "read": "READ",
            "прочти": "READ",
            "запиши": "WRITE",
            "записать": "WRITE",
            "перепиши": "WRITE",
            "overwrite": "WRITE",
            "write": "WRITE",
            "append": "APPEND",
            "добавь": "APPEND",
            "добавить": "APPEND",
            "append_rows": "APPEND",
            "appendrows": "APPEND",
            "очисти": "CLEAR",
            "очистить": "CLEAR",
            "clear": "CLEAR",
            "инфо": "INFO",
            "информация": "INFO",
            "info": "INFO",
            "format_range": "FORMAT_RANGE",
            "формат": "FORMAT_RANGE",
            "format": "FORMAT_RANGE",
            "update_cell": "UPDATE_CELL",
            "обнови": "UPDATE_CELL",
            "обновить": "UPDATE_CELL",
            "update": "UPDATE_CELL",
        }
        key = action.strip().lower()
        if not key:
            return ""
        return mapping.get(key, action.strip().upper())
    
    def _handle_material_search(self, params: Dict[str, Any], command: str, _: Dict[str, Any]) -> str:
        """Обработать команды поиска материалов."""
        material_name = (params.get("material_name") or command or "").strip()
        if not material_name:
            return "❌ Не удалось определить название материала"
        
        use_cache = self._normalize_bool(params.get("use_cache"), True)
        use_scraping = self._normalize_bool(params.get("use_scraping"), False)
        use_advanced = self._normalize_bool(params.get("use_advanced_search"), False)
        
        result = self.find_material_price(
            material_name,
            use_cache=use_cache,
            use_scraping=use_scraping,
            use_advanced_search=use_advanced,
        )
        return self._format_material_result(result)
    
    def _handle_sheets_operation(self, params: Dict[str, Any], command: str, _: Dict[str, Any]) -> str:
        """Обработать команды Google Sheets."""
        if not self.sheets_ai:
            return "❌ Google Sheets AI not initialized. Set GOOGLE_SHEET_ID in .env"
        
        action = self._normalize_action(str(params.get("action") or ""))
        worksheet = params.get("worksheet") or params.get("sheet")
        worksheet_hint = worksheet.strip() if isinstance(worksheet, str) else None
        range_a1_raw = params.get("range") or params.get("a1_range")
        range_a1 = range_a1_raw.strip() if isinstance(range_a1_raw, str) else None
        title = params.get("title")
        data_param = params.get("data")
        
        if action == "READ":
            try:
                if range_a1:
                    data = self.read_sheet_range(range_a1, worksheet_hint)
                    preview = self._format_sheet_preview(data)
                    resolved = self.sheets_ai.worksheet_name
                    return (
                        f"📄 Лист: {resolved}\n"
                        f"Диапазон {range_a1}: {len(data)} строк(и)\n"
                        f"Предпросмотр:\n{preview}"
                    )
                
                data = self.read_sheet_data(worksheet_hint)
                rows = len(data)
                cols = len(data[0]) if data else 0
                preview = self._format_sheet_preview(data)
                name = self.sheets_ai.worksheet_name
                return (
                    f"📄 Лист: {name}\n"
                    f"Строк: {rows}, Колонок: {cols}\n"
                    f"Предпросмотр (до 10 строк):\n{preview}"
                )
            except Exception as exc:
                logger.error("Error reading sheet: %s", exc)
                return f"❌ Ошибка чтения листа: {exc}"
        
        if action == "WRITE":
            if isinstance(data_param, list) and all(isinstance(row, list) for row in data_param):
                try:
                    self.write_sheet_data(data_param, title, worksheet_hint)
                    return f"✅ Данные записаны в лист '{self.sheets_ai.worksheet_name}'"
                except Exception as exc:
                    logger.error("Error writing sheet: %s", exc)
                    return f"❌ Ошибка записи данных: {exc}"
            logger.info("Structured write request missing data payload; falling back to natural command pipeline")
        
        if action == "APPEND":
            if isinstance(data_param, list) and all(isinstance(row, list) for row in data_param):
                try:
                    self.append_sheet_data(data_param, worksheet_hint)
                    return f"✅ Добавлены строки в лист '{self.sheets_ai.worksheet_name}'"
                except Exception as exc:
                    logger.error("Error appending to sheet: %s", exc)
                    return f"❌ Ошибка добавления строк: {exc}"
            logger.info("Structured append request missing data payload; falling back to natural command pipeline")
        
        if action == "CLEAR":
            try:
                self.clear_sheet(worksheet_hint)
                return f"✅ Лист '{self.sheets_ai.worksheet_name}' очищен"
            except Exception as exc:
                logger.error("Error clearing sheet: %s", exc)
                return f"❌ Ошибка очистки листа: {exc}"
        
        if action == "INFO":
            try:
                info = self.get_sheet_info(worksheet_hint)
                lines = [
                    f"📊 Spreadsheet: {info.get('spreadsheet_title')}",
                    f"📄 Лист: {info.get('worksheet')}",
                    f"Строк: {info.get('rows')}, Колонок: {info.get('columns')}",
                ]
                url = info.get("worksheet_url")
                if url:
                    lines.append(f"🔗 {url}")
                return "\n".join(lines)
            except Exception as exc:
                logger.error("Error getting sheet info: %s", exc)
                return f"❌ Ошибка получения информации о листе: {exc}"
        
        if action == "UPDATE_CELL":
            row = params.get("row")
            col = params.get("col")
            value = params.get("value")
            if isinstance(row, (int, float)) and isinstance(col, (int, float)) and value is not None:
                try:
                    self.update_sheet_cell(int(row), int(col), str(value), worksheet_hint)
                    return (
                        f"✅ Ячейка ({int(row)}, {int(col)}) листа '{self.sheets_ai.worksheet_name}' обновлена"
                    )
                except Exception as exc:
                    logger.error("Error updating cell: %s", exc)
                    return f"❌ Ошибка обновления ячейки: {exc}"
            logger.info("Structured update_cell request missing row/col/value; falling back to natural command pipeline")
        
        if action == "FORMAT_RANGE":
            format_spec = params.get("format")
            if not range_a1:
                return "❌ Для форматирования необходимо указать поле 'range'"
            if not isinstance(format_spec, dict):
                return "❌ Для форматирования необходимо передать объект 'format'"
            try:
                message = self.format_sheet_range(range_a1, format_spec, worksheet_hint)
                return f"✅ {message}"
            except Exception as exc:
                logger.error("Error formatting range: %s", exc)
                return f"❌ Ошибка форматирования диапазона: {exc}"
        
        # Для остальных действий полагаемся на универсальный обработчик в GoogleSheetsAI
        return self.process_sheets_command(command)
    
    def _handle_estimate_check(self, params: Dict[str, Any], _: str, __: Dict[str, Any]) -> str:
        """Обработать команды проверки сметы."""
        estimate_sheet = params.get("estimate_sheet") or params.get("worksheet") or params.get("sheet")
        master_sheet = params.get("master_sheet") or "Master List"
        quantity_col = params.get("quantity_col") or "F"
        return self.check_estimate(estimate_sheet, master_sheet, quantity_col)
        
    def _handle_web_search(self, params: Dict[str, Any], command: str, _: Dict[str, Any]) -> str:
        """Обработать команды веб-поиска."""
        if not self.sheets_ai:
            return "❌ Web search not available: Google Sheets AI is not configured"
        
        query = (params.get("query") or command or "").strip()
        if not query:
            return "❌ Не удалось определить поисковый запрос"
        
        try:
            results = self.search_web(query)
            return self._format_search_results(results)
        except Exception as exc:
            logger.error("Error during web search: %s", exc)
            return f"❌ Ошибка веб-поиска: {exc}"

    def _handle_materials_db(self, params: Dict[str, Any], command: str, classification: Dict[str, Any]) -> str:
        """Обработать команды помощника базы материалов и работ."""
        if not self.materials_db_assistant:
            return "❌ Помощник базы материалов недоступен. Проверьте LOCAL_MATERIALS_CSV_PATH."

        hints: Dict[str, Any] = {}
        if isinstance(params, dict):
            for key, target in [
                ("intent", "intent"),
                ("record_id", "record_id"),
                ("db_query", "query"),
                ("query", "query"),
                ("material_name", "query"),
                ("confirmation_id", "confirmation_id"),
                ("worksheet", "worksheet"),
                ("sheet", "worksheet"),
            ]:
                value = params.get(key)
                if isinstance(value, str) and value.strip():
                    hints[target] = value.strip()
        explanation = classification.get("explanation")
        if isinstance(explanation, str) and explanation.strip():
            hints.setdefault("classification_explanation", explanation.strip())

        try:
            return self.materials_db_assistant.handle_request(command, hints=hints or None)
        except Exception as exc:
            logger.error("Error in materials DB assistant: %s", exc)
            return f"❌ Ошибка помощника базы материалов: {exc}"
    
    def _handle_project_chat(
        self,
        params: Dict[str, Any],
        command: str,
        _: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Обработать запросы, связанные с обсуждением проекта."""
        if not self.project_chat_agent:
            return "❌ Project chat недоступен: функция отключена или не инициализирована"
        
        extra_context_parts: List[str] = []
        topic = params.get("topic")
        if isinstance(topic, str) and topic.strip():
            extra_context_parts.append(f"Запрошенная тема: {topic.strip()}")
        
        if self.materials_db_assistant:
            lookup_seed = None
            if isinstance(topic, str) and topic.strip():
                lookup_seed = topic.strip()
            elif isinstance(command, str) and command.strip():
                lookup_seed = command.strip()
            if lookup_seed:
                try:
                    db_summary = self.materials_db_assistant.quick_lookup(lookup_seed, limit=5)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Materials DB quick lookup failed: %s", exc)
                    db_summary = None
                if db_summary:
                    snippet_lines = db_summary.splitlines()
                    snippet = "\n".join(snippet_lines[:8])
                    extra_context_parts.append(
                        "Материалы / работы (Google Sheets):\n" + snippet
                    )
        
        reset_flag = False
        if context:
            if context.get("reset_project_chat"):
                reset_flag = True
            ctx_text = context.get("project_context") or context.get("extra_context")
            if isinstance(ctx_text, str) and ctx_text.strip():
                extra_context_parts.append(ctx_text.strip())
        
        extra_context = "\n\n".join(extra_context_parts) if extra_context_parts else None
        
        try:
            return self.chat_about_project(
                command,
                extra_context=extra_context,
                reset_history=reset_flag,
            )
        except Exception as exc:
            logger.error("Error during project chat: %s", exc)
            return f"❌ Ошибка проектного чата: {exc}"
    
    def _handle_general(self, params: Dict[str, Any], _: str, classification: Dict[str, Any]) -> str:
        """Обработать неопознанные команды."""
        explanation = classification.get("explanation") or ""
        hint = f"\n{explanation}" if explanation else ""
        return f"ℹ️ Команда распознана как общий запрос. Попробуйте уточнить.{hint}"
    
    @staticmethod
    def _format_sheet_preview(data: List[List[str]], limit: int = 10) -> str:
        """Сформировать текстовый предпросмотр листа."""
        if not data:
            return "(пусто)"
        
        preview_rows = data[:limit]
        lines = []
        for row in preview_rows:
            formatted = " | ".join(
                (
                    cell.strip()
                    if isinstance(cell, str)
                    else ("" if cell is None else str(cell))
                )
                for cell in row
            )
            lines.append(formatted)
        if len(data) > limit:
            lines.append("...")
        return "\n".join(lines)
    
    def _reconstruct_material_result(self, material_name: str, cached: Dict) -> MaterialResult:
        """Восстановить MaterialResult из кэша."""
        analysis = MaterialQueryAnalysis(
            pt_name=cached["pt_name"],
            search_queries=cached["search_queries"],
            key_specs=cached["key_specs"],
        )
        
        quotes = [SupplierQuote(**q) for q in cached["suppliers"]]
        
        best_offer = BestOffer(
            best_supplier=cached["best_supplier"],
            price=cached["price"],
            url=cached["url"],
            reasoning=cached["reasoning"],
        )
        
        return MaterialResult(
            material_name=material_name,
            analysis=analysis,
            quotes=quotes,
            best_offer=best_offer,
        )
    
    def _cache_material_result(self, material_name: str, result: MaterialResult):
        """Сохранить результат в кэш."""
        self.cache.set_material(
            material_name=material_name,
            pt_name=result.analysis.pt_name,
            search_queries=result.analysis.search_queries,
            key_specs=result.analysis.key_specs,
            best_supplier=result.best_offer.best_supplier,
            price=result.best_offer.price,
            url=result.best_offer.url,
            reasoning=result.best_offer.reasoning,
            suppliers=[q.model_dump() for q in result.quotes],
        )
    
    def _convert_advanced_result(self, advanced_result: "MaterialSearchResult") -> MaterialResult:
        """Конвертировать результат продвинутого агента в стандартный формат."""
        analysis = MaterialQueryAnalysis(
            pt_name=advanced_result.pt_name,
            search_queries=[],
            key_specs=[],
        )
        
        quotes = [
            SupplierQuote(
                supplier=s.get("supplier", "Unknown"),
                price=s.get("price", "N/A"),
                url=s.get("url", ""),
                notes=s.get("notes"),
            )
            for s in advanced_result.suppliers
        ]
        
        best_offer = BestOffer(
            best_supplier=advanced_result.best_supplier,
            price=advanced_result.best_price,
            url=advanced_result.best_url,
            reasoning=advanced_result.reasoning,
        )
        
        return MaterialResult(
            material_name=advanced_result.material_name,
            analysis=analysis,
            quotes=quotes,
            best_offer=best_offer,
        )
    
    def _format_material_result(self, result: MaterialResult) -> str:
        """Форматировать результат поиска материала."""
        lines = [
            f"🔍 **Материал:** {result.material_name}",
            f"🇵🇹 **Португальское название:** {result.analysis.pt_name}",
            f"",
            f"✅ **Лучшее предложение:**",
            f"- **Поставщик:** {result.best_offer.best_supplier}",
            f"- **Цена:** {result.best_offer.price}",
            f"- **Ссылка:** {result.best_offer.url}",
            f"- **Обоснование:** {result.best_offer.reasoning}",
        ]
        
        if result.quotes:
            lines.append(f"\n📋 **Найдено предложений:** {len(result.quotes)}")
        
        return "\n".join(lines)
    
    def _format_search_results(self, results: List[Dict[str, str]]) -> str:
        """Форматировать результаты веб-поиска."""
        if not results:
            return "Результаты не найдены"
        
        lines = ["🔍 **Результаты поиска:**\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. **{r['title']}**")
            lines.append(f"   {r['snippet']}")
            lines.append(f"   🔗 {r['link']}\n")
        
        return "\n".join(lines)
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику работы агента."""
        cache_stats = self.cache.get_stats()
        sheet_stats: Optional[Dict[str, Any]] = None

        if self.sheets_ai:
            sheet_stats = self.sheets_ai.get_table_info()
            sheet_stats['worksheet_name'] = self.sheets_ai.worksheet_name
            try:
                sheet_stats['available_worksheets'] = [
                    ws.title for ws in self.sheets_ai.spreadsheet.worksheets()
                ]
            except Exception:
                sheet_stats['available_worksheets'] = []

        return {
            "cache": cache_stats,
            "config": {
                "model": self.config.llm_model,
                "cache_enabled": True,
                "sheets_enabled": self.sheets_ai is not None,
                "advanced_search_enabled": self.advanced_agent is not None,
                "local_llm_enabled": self.config.enable_local_llm,
            },
            "sheet": sheet_stats,
            "materials_db": {
                "assistant_enabled": self.materials_db_assistant is not None,
                "pending_changes": (
                    self.materials_db_assistant.pending_count()
                    if self.materials_db_assistant
                    else 0
                ),
                "csv_path": str(self.config.local_materials_csv_path or ""),
                "sheet_id": self.config.materials_db_sheet_id or "",
                "worksheet": (
                    self.materials_db_assistant.worksheet_name
                    if self.materials_db_assistant
                    else (self.config.materials_db_worksheet or "")
                ),
                "sheet_enabled": bool(self.config.materials_db_sheet_id),
                "sync_csv": bool(
                    self.config.materials_db_sync_csv and self.config.local_materials_csv_path
                ),
                "worksheets": (
                    self.materials_db_assistant.worksheet_names
                    if self.materials_db_assistant
                    else [
                        item.strip()
                        for item in (self.config.materials_db_worksheet or "").split(",")
                        if item.strip()
                    ]
                ),
            },
        }


# ---------------------------------------------------------------------------
# Обратная совместимость
# ---------------------------------------------------------------------------
UnifiedAgentConfig = ConstructionAIAgentConfig
UnifiedAgent = ConstructionAIAgent


def main():
    """CLI интерфейс для единого агента."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Construction AI Agent")
    parser.add_argument("command", nargs="*", help="Command to execute")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    
    args = parser.parse_args()
    
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    # Инициализация агента
    try:
        agent = ConstructionAIAgent()
    except ValueError as exc:
        message = str(exc)
        print(f"❌ Ошибка инициализации агента: {message}")
        if "OPENAI_API_KEY" in message:
            print("Подсказка: установите переменную окружения OPENAI_API_KEY перед запуском.")
        sys.exit(1)
    except Exception as exc:
        print(f"❌ Непредвиденная ошибка инициализации агента: {exc}")
        sys.exit(1)
    
    if args.stats:
        stats = agent.get_stats()
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return
    
    if args.interactive:
        print("🤖 Construction AI Agent - Interactive Mode")
        print("=" * 60)
        print("Доступные команды:")
        print("- Найди цену на [материал]")
        print("- Прочитай таблицу")
        print("- Проверь смету")
        print("- Найди в интернете [запрос]")
        print("- exit - выход\n")
        
        while True:
            try:
                command = input("💬 Команда: ").strip()
                if command.lower() in ["exit", "quit", "выход"]:
                    print("👋 До свидания!")
                    break
                
                if not command:
                    continue
                
                print("\n⏳ Обработка...\n")
                result = agent.process_command(command)
                print(result)
                print("\n" + "-" * 60 + "\n")
            
            except KeyboardInterrupt:
                print("\n👋 До свидания!")
                break
            except Exception as e:
                print(f"❌ Ошибка: {e}\n")
    
    elif args.command:
        command = " ".join(args.command)
        result = agent.process_command(command)
        print(result)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
