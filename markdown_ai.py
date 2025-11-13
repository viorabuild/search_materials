import os
import re
import json
import difflib
import logging
import threading
import time
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urlparse

import gspread
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from gspread.exceptions import WorksheetNotFound, APIError
from gspread.utils import a1_range_to_grid_range
from openai import OpenAI
from tabulate import tabulate
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class GoogleAPIRateLimitError(RuntimeError):
    """Ошибка ограничения частоты обращений к Google API."""

    def __init__(self, message: str, code: str = "GOOGLE_RATE_LIMIT_EXCEEDED"):
        super().__init__(message)
        self.code = code
        self.message = message


def _column_letter(index: int) -> str:
    """Преобразование 0-индексированного номера столбца в буквенное обозначение."""
    index += 1  # переводим в 1-индекс
    letters = []
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        letters.append(chr(65 + remainder))
    return ''.join(reversed(letters)) or 'A'


def _cell_label(row: int, col: int) -> str:
    """Преобразование координат (0-индекс) в обозначение A1."""
    return f"{_column_letter(col)}{row + 1}"


def _normalize_key(key: str) -> str:
    """Нормализация ключей формата (нижний регистр, без неалфанумерических символов)."""
    return ''.join(ch for ch in str(key).lower() if ch.isalnum())


_COLOR_KEYWORDS = {
    'red': '#ff0000',
    'green': '#00ff00',
    'blue': '#0000ff',
    'black': '#000000',
    'white': '#ffffff',
    'yellow': '#ffff00',
    'orange': '#ffa500',
    'purple': '#800080',
    'violet': '#800080',
    'pink': '#ffc0cb',
    'gray': '#808080',
    'grey': '#808080',
    'teal': '#008080',
    'cyan': '#00ffff',
    'magenta': '#ff00ff',
    'brown': '#8b4513',
}


def _parse_color(value: Any) -> Optional[Dict[str, float]]:
    """Преобразование цвета в формат Google Sheets (значения 0..1)."""
    if value is None:
        return None

    if isinstance(value, str):
        color_str = value.strip().lower()
        if not color_str:
            return None
        color_str = _COLOR_KEYWORDS.get(color_str, color_str)
        if color_str.startswith('#'):
            hex_value = color_str.lstrip('#')
            if len(hex_value) == 3:
                hex_value = ''.join(ch * 2 for ch in hex_value)
            if len(hex_value) == 6:
                try:
                    red = int(hex_value[0:2], 16) / 255.0
                    green = int(hex_value[2:4], 16) / 255.0
                    blue = int(hex_value[4:6], 16) / 255.0
                    return {'red': red, 'green': green, 'blue': blue}
                except ValueError:
                    return None
        return None

    if isinstance(value, dict):
        components = {}
        for comp in ('red', 'green', 'blue', 'alpha'):
            if comp in value:
                comp_value = value[comp]
                try:
                    float_value = float(comp_value)
                    if float_value > 1:
                        float_value /= 255.0
                    components[comp] = max(0.0, min(1.0, float_value))
                except (TypeError, ValueError):
                    continue
        if components:
            return components

    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            red, green, blue = (float(value[0]), float(value[1]), float(value[2]))
            if red > 1 or green > 1 or blue > 1:
                red /= 255.0
                green /= 255.0
                blue /= 255.0
            return {
                'red': max(0.0, min(1.0, red)),
                'green': max(0.0, min(1.0, green)),
                'blue': max(0.0, min(1.0, blue)),
            }
        except (TypeError, ValueError):
            return None

    return None


def _normalize_side(value: str) -> Optional[str]:
    """Нормализация названия стороны границы."""
    if not value:
        return None
    value = value.strip().lower()
    if value.startswith('top'):
        return 'top'
    if value.startswith('bottom'):
        return 'bottom'
    if value.startswith('left'):
        return 'left'
    if value.startswith('right'):
        return 'right'
    if value in {'outer', 'outline', 'all', 'border'}:
        return 'all'
    return None

# Import estimate checker module
try:
    from estimate_checker import EstimateChecker, create_estimate_system_prompt
    ESTIMATE_CHECKER_AVAILABLE = True
except ImportError:
    ESTIMATE_CHECKER_AVAILABLE = False


class GoogleSheetsAI:
    def __init__(
        self,
        sheet_id: Optional[str] = None,
        worksheet_name: Optional[str] = None,
        openai_client: Optional[Any] = None,
        llm_model: Optional[str] = None,
    ):
        """Инициализация клиентов OpenAI и Google Sheets"""
        env_api_key = os.getenv('OPENAI_API_KEY')
        if openai_client is None and not env_api_key:
            raise ValueError("OPENAI_API_KEY is required when openai_client is not provided.")
        self.openai_client = openai_client or OpenAI(api_key=env_api_key)
        self.llm_model = llm_model or os.getenv('GOOGLE_SHEETS_LLM_MODEL') or "gpt-4o-mini"

        self.sheet_id = sheet_id or os.getenv('GOOGLE_SHEET_ID')
        if not self.sheet_id:
            raise ValueError("Не задан идентификатор таблицы. Укажите GOOGLE_SHEET_ID в окружении.")

        env_worksheet = os.getenv('GOOGLE_SHEET_WORKSHEET')
        if env_worksheet is not None:
            env_worksheet = env_worksheet.strip() or None

        self.worksheet_name = worksheet_name or env_worksheet

        self.allowed_domains = self._load_allowed_domains()
        self.fetch_timeout = int(os.getenv('FETCH_TIMEOUT_SECONDS', '10'))
        self.fetch_max_chars = int(os.getenv('FETCH_MAX_CONTENT_LENGTH', '8000'))
        self.fetch_max_rounds = int(os.getenv('FETCH_MAX_REQUESTS', '3'))

        self.serpapi_key = os.getenv('SERPAPI_API_KEY')
        self.search_locale = os.getenv('SERPAPI_LOCALE', 'ru')
        self.search_max_rounds = int(os.getenv('SEARCH_MAX_REQUESTS', '2'))
        self.search_max_results = int(os.getenv('SEARCH_MAX_RESULTS', '5'))

        self._requests_retryer = self._build_requests_retryer()
        self._google_retryer = self._build_google_retryer()
        self._init_google_rate_limiter()

        self.gspread_client = self._create_gspread_client()
        self.spreadsheet = self.gspread_client.open_by_key(self.sheet_id)

        desired_worksheet = (self.worksheet_name or "").strip()
        if desired_worksheet:
            self.worksheet = self._get_or_create_worksheet(self.spreadsheet, desired_worksheet)
        else:
            # Используем первый лист, если имя не указано
            self.worksheet = self.spreadsheet.sheet1

        self.worksheet_name = self.worksheet.title

        # Initialize estimate checker if available
        self.estimate_checker = EstimateChecker(self) if ESTIMATE_CHECKER_AVAILABLE else None

        # Conversation history for chat-style interactions
        self._chat_history: List[Dict[str, str]] = []

    # ------------------------------------------------------------------
    # Chat helpers
    # ------------------------------------------------------------------

    def reset_chat(self) -> None:
        """Сбросить историю диалога с ассистентом Google Sheets."""
        self._chat_history.clear()

    def get_chat_history(self) -> List[Dict[str, str]]:
        """Вернуть текущую историю диалога (без модификации)."""
        return list(self._chat_history)

    def append_system_message(self, content: str) -> None:
        """Добавить служебное сообщение в историю (без запроса к LLM)."""
        if content:
            self._chat_history.append({"role": "system", "content": content})

    def _build_requests_retryer(self) -> Retrying:
        attempts = max(1, int(os.getenv('NETWORK_RETRY_ATTEMPTS', '3')))
        multiplier = float(os.getenv('NETWORK_RETRY_BACKOFF_MULTIPLIER', '1'))
        max_wait = max(1.0, float(os.getenv('NETWORK_RETRY_MAX_WAIT_SECONDS', '10')))
        return Retrying(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(multiplier=multiplier, min=1, max=max_wait),
            retry=retry_if_exception_type(requests.exceptions.RequestException),
            reraise=True,
        )

    def _build_google_retryer(self) -> Retrying:
        attempts = max(1, int(os.getenv('GOOGLE_API_RETRY_ATTEMPTS', '3')))
        multiplier = float(os.getenv('GOOGLE_API_RETRY_BACKOFF_MULTIPLIER', '1'))
        max_wait = max(1.0, float(os.getenv('GOOGLE_API_RETRY_MAX_WAIT_SECONDS', '10')))
        return Retrying(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(multiplier=multiplier, min=1, max=max_wait),
            retry=retry_if_exception_type(APIError),
            reraise=True,
        )

    def _init_google_rate_limiter(self) -> None:
        tokens_raw = os.getenv('GOOGLE_API_RATE_TOKENS', '10')
        interval_raw = os.getenv('GOOGLE_API_RATE_INTERVAL_SECONDS', '60')
        error_code = os.getenv('GOOGLE_API_RATE_ERROR_CODE', 'GOOGLE_RATE_LIMIT_EXCEEDED')

        try:
            capacity = int(tokens_raw)
        except (TypeError, ValueError):
            capacity = 10
        self._google_api_capacity = max(1, capacity)

        try:
            interval = float(interval_raw)
        except (TypeError, ValueError):
            interval = 60.0
        self._google_api_interval = max(0.1, interval)

        self._google_api_rate_error_code = error_code or 'GOOGLE_RATE_LIMIT_EXCEEDED'
        self._google_api_tokens = self._google_api_capacity
        self._google_api_last_refill = time.monotonic()
        self._google_api_lock = threading.Lock()

    def _refill_google_api_tokens(self) -> None:
        now = time.monotonic()
        elapsed = now - self._google_api_last_refill
        if elapsed >= self._google_api_interval:
            self._google_api_tokens = self._google_api_capacity
            self._google_api_last_refill = now

    def _acquire_google_api_token(self) -> None:
        with self._google_api_lock:
            self._refill_google_api_tokens()
            if self._google_api_tokens <= 0:
                logger.warning(
                    "Google API rate limit exceeded (capacity=%s, interval=%s)",
                    self._google_api_capacity,
                    self._google_api_interval,
                )
                raise GoogleAPIRateLimitError(
                    (
                        "Превышено количество запросов к Google API: "
                        f"доступно {self._google_api_capacity} запросов за {self._google_api_interval} с"
                    ),
                    code=self._google_api_rate_error_code,
                )
            self._google_api_tokens -= 1

    def _call_with_requests_retry(self, func, *args, **kwargs):
        for attempt in self._requests_retryer:
            with attempt:
                return func(*args, **kwargs)

    def _call_google_api(self, func, *args, **kwargs):
        self._acquire_google_api_token()
        for attempt in self._google_retryer:
            with attempt:
                return func(*args, **kwargs)

    def _create_gspread_client(self) -> gspread.Client:
        """Создание gspread клиента на основе сервисного аккаунта"""
        json_credentials = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON')
        json_file = os.getenv('GOOGLE_SERVICE_ACCOUNT_FILE')

        if json_credentials:
            try:
                credentials_dict = json.loads(json_credentials)
            except json.JSONDecodeError as exc:
                raise ValueError("GOOGLE_SERVICE_ACCOUNT_JSON содержит некорректный JSON") from exc
            return gspread.service_account_from_dict(credentials_dict)

        if json_file:
            if not os.path.exists(json_file):
                raise FileNotFoundError(f"Файл сервисного аккаунта не найден: {json_file}")
            return gspread.service_account(filename=json_file)

        raise ValueError(
            "Не найден сервисный аккаунт Google. Укажите GOOGLE_SERVICE_ACCOUNT_JSON "
            "или GOOGLE_SERVICE_ACCOUNT_FILE в окружении."
        )

    def _resolve_worksheet(self, worksheet_name: Optional[str] = None) -> gspread.Worksheet:
        """
        Получение листа по имени с поддержкой похожих названий.

        Если worksheet_name не указан, возвращается текущий лист.
        При успешном поиске текущий лист обновляется.
        """
        if not worksheet_name:
            return self.worksheet

        target_name = worksheet_name.strip()
        if not target_name:
            return self.worksheet

        if target_name.lower() in {self.worksheet.title.lower(), self.worksheet_name.lower()}:
            return self.worksheet

        try:
            worksheet = self.spreadsheet.worksheet(target_name)
        except WorksheetNotFound:
            all_sheets = self.spreadsheet.worksheets()
            titles = [ws.title for ws in all_sheets]
            matches = difflib.get_close_matches(target_name, titles, n=1, cutoff=0.6)
            if not matches:
                raise WorksheetNotFound(f"Worksheet '{worksheet_name}' not found")
            worksheet = self.spreadsheet.worksheet(matches[0])

        # Обновляем текущий лист
        self.worksheet = worksheet
        self.worksheet_name = worksheet.title
        return worksheet

    def _get_or_create_worksheet(self, spreadsheet: gspread.Spreadsheet, worksheet_name: str) -> gspread.Worksheet:
        """Получение существующего листа или создание нового"""
        try:
            return spreadsheet.worksheet(worksheet_name)
        except WorksheetNotFound:
            default_rows = int(os.getenv('GOOGLE_SHEET_DEFAULT_ROWS', '100'))
            default_columns = int(os.getenv('GOOGLE_SHEET_DEFAULT_COLUMNS', '20'))
            return spreadsheet.add_worksheet(
                title=worksheet_name,
                rows=str(default_rows),
                cols=str(default_columns)
            )

    @staticmethod
    def _load_allowed_domains() -> List[str]:
        """Получение списка разрешенных доменов из окружения"""
        raw = os.getenv('ALLOWED_WEB_DOMAINS', '')
        domains = []
        for item in raw.split(','):
            domain = item.strip().lower()
            if domain:
                domains.append(domain)
        return domains

    def _ensure_domain_allowed(self, url: str):
        """Проверка, что домен разрешен политикой безопасности"""
        parsed = urlparse(url)
        hostname = (parsed.hostname or '').lower()
        if not hostname:
            raise ValueError(f"Некорректный URL: {url}")

        if not self.allowed_domains:
            return

        for allowed in self.allowed_domains:
            if hostname == allowed or hostname.endswith('.' + allowed):
                return
        raise ValueError(f"Домен {hostname} не входит в список разрешенных: {', '.join(self.allowed_domains)}")

    @staticmethod
    def _html_to_text(html: str) -> str:
        """Преобразование HTML в текст"""
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(" ", strip=True)
        return re.sub(r'\s+', ' ', text)

    def fetch_web_content(self, url: str) -> str:
        """Загрузка текста со страницы с учетом ограничений"""
        normalized_url = url.strip()
        if not normalized_url:
            raise ValueError("URL не указан")
        if not normalized_url.startswith(('http://', 'https://')):
            normalized_url = f"https://{normalized_url}"

        self._ensure_domain_allowed(normalized_url)

        response = self._call_with_requests_retry(
            requests.get,
            normalized_url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; GoogleSheetsAI/1.0)"},
            timeout=self.fetch_timeout,
        )
        response.raise_for_status()

        if 'text/html' in response.headers.get('Content-Type', ''):
            text = self._html_to_text(response.text)
        else:
            text = response.text

        if len(text) > self.fetch_max_chars:
            return text[: self.fetch_max_chars]
        return text

    def search_web(self, query: str) -> List[Dict[str, str]]:
        """Поиск информации через SerpAPI"""
        if not self.serpapi_key:
            raise ValueError("SERPAPI_API_KEY не задан")

        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Пустой поисковый запрос")

        params = {
            "api_key": self.serpapi_key,
            "engine": "google",
            "q": normalized_query,
            "hl": self.search_locale,
            "gl": self.search_locale.split('-')[-1] if '-' in self.search_locale else self.search_locale,
            "num": self.search_max_results,
        }

        response = self._call_with_requests_retry(
            requests.get,
            "https://serpapi.com/search.json",
            params=params,
            timeout=self.fetch_timeout,
        )
        response.raise_for_status()

        data = response.json()
        organic_results = data.get("organic_results", [])[: self.search_max_results]

        sanitized_results: List[Dict[str, str]] = []
        for item in organic_results:
            title = (item.get("title") or "").strip()
            link = (item.get("link") or "").strip()
            snippet = (item.get("snippet") or item.get("snippet_highlighted_words") or "")
            if isinstance(snippet, list):
                snippet = " ".join(snippet)
            snippet = str(snippet).strip()
            sanitized_results.append({
                "title": title or "(без заголовка)",
                "link": link,
                "snippet": snippet[:400],
            })

        return sanitized_results

    def read_sheet_data(self, worksheet_name: Optional[str] = None) -> List[List[str]]:
        """Чтение данных с листа"""
        worksheet = self._resolve_worksheet(worksheet_name)
        return worksheet.get_all_values()

    def write_sheet_data(
        self,
        data: List[List[str]],
        title: Optional[str] = None,
        worksheet_name: Optional[str] = None,
    ):
        """Полная перезапись листа данными"""
        if not data:
            return

        worksheet = self._resolve_worksheet(worksheet_name)
        self._call_google_api(worksheet.clear)
        self._call_google_api(
            worksheet.update,
            'A1',
            data,
            value_input_option="USER_ENTERED",
        )

        if title:
            try:
                self.spreadsheet.update_title(title)
            except Exception:
                # Название таблицы менять необязательно, поэтому игнорируем ошибки
                pass

    def append_to_sheet(self, rows: List[List[str]], worksheet_name: Optional[str] = None):
        """Добавление строк в конец листа"""
        if not rows:
            return

        worksheet = self._resolve_worksheet(worksheet_name)
        existing_data = worksheet.get_all_values()

        if not existing_data:
            self.write_sheet_data(rows, worksheet_name=worksheet.title)
            return

        rows_to_append = rows
        if rows and existing_data:
            header = existing_data[0]
            if rows[0] == header:
                rows_to_append = rows[1:]

        if rows_to_append:
            self._call_google_api(
                worksheet.append_rows,
                rows_to_append,
                value_input_option="USER_ENTERED",
            )

    def update_cell(self, row: int, col: int, value: str, worksheet_name: Optional[str] = None):
        """Обновление конкретной ячейки (0-indexed)"""
        worksheet = self._resolve_worksheet(worksheet_name)
        self._call_google_api(worksheet.update_cell, row + 1, col + 1, value)

    def format_range(self, range_a1: str, format_spec: Dict[str, Any], worksheet_name: Optional[str] = None) -> str:
        """Применение форматирования к диапазону."""
        if not range_a1 or not range_a1.strip():
            raise ValueError("Не указан диапазон (range)")
        if not isinstance(format_spec, dict) or not format_spec:
            raise ValueError("Параметр 'format' должен быть непустым объектом")

        normalized = {}
        for key, value in format_spec.items():
            normalized[_normalize_key(key)] = value

        def get_value(*keys: str) -> Any:
            for key in keys:
                norm = _normalize_key(key)
                if norm in normalized:
                    return normalized[norm]
            return None

        user_format: Dict[str, Any] = {}
        fields: Set[str] = set()

        def mark(field: str):
            fields.add(f"userEnteredFormat.{field}")

        # Background color
        background = _parse_color(get_value('backgroundcolor', 'background', 'fillcolor', 'fill'))
        if background:
            user_format['backgroundColor'] = background
            mark('backgroundColor')

        # Text format
        text_format: Dict[str, Any] = {}
        text_color = _parse_color(get_value('textcolor', 'fontcolor', 'fontcolour', 'foregroundcolor'))
        if text_color:
            text_format['foregroundColor'] = text_color
            mark('textFormat.foregroundColor')

        for attr in ('bold', 'italic', 'underline', 'strikethrough'):
            value = get_value(attr, f'{attr}text')
            if value is not None:
                text_format[attr] = bool(value)
                mark(f'textFormat.{attr}')

        strike = get_value('strike', 'strikeout')
        if strike is not None:
            text_format['strikethrough'] = bool(strike)
            mark('textFormat.strikethrough')

        font_size = get_value('fontsize', 'size')
        if font_size is not None:
            try:
                text_format['fontSize'] = int(font_size)
                mark('textFormat.fontSize')
            except (TypeError, ValueError):
                pass

        font_family = get_value('fontfamily', 'font')
        if font_family:
            text_format['fontFamily'] = str(font_family)
            mark('textFormat.fontFamily')

        if text_format:
            user_format['textFormat'] = text_format

        # Alignment
        horizontal = get_value('horizontalalignment', 'halign', 'alignment', 'align')
        if horizontal:
            value = str(horizontal).strip().upper().replace(' ', '_')
            allowed = {'LEFT', 'CENTER', 'RIGHT', 'JUSTIFY'}
            if value in allowed:
                user_format['horizontalAlignment'] = value
                mark('horizontalAlignment')

        vertical = get_value('verticalalignment', 'valign')
        if vertical:
            value = str(vertical).strip().upper().replace(' ', '_')
            allowed = {'TOP', 'MIDDLE', 'BOTTOM'}
            if value in allowed:
                user_format['verticalAlignment'] = value
                mark('verticalAlignment')

        # Wrap strategy
        wrap = get_value('wrapstrategy', 'wrap', 'wraptext')
        if wrap:
            value = str(wrap).strip().upper().replace(' ', '_')
            allowed = {'WRAP', 'OVERFLOW_CELL', 'CLIP'}
            if value in allowed:
                user_format['wrapStrategy'] = value
                mark('wrapStrategy')

        # Borders
        borders_spec = get_value('borders', 'border')
        borders: Dict[str, Any] = {}
        sides_all = ['top', 'bottom', 'left', 'right']

        def build_border(border_info: Dict[str, Any]) -> Dict[str, Any]:
            border: Dict[str, Any] = {}
            style_raw = border_info.get('style', 'SOLID')
            border['style'] = str(style_raw).upper()
            width = border_info.get('width')
            if width is not None:
                try:
                    border['width'] = float(width)
                except (TypeError, ValueError):
                    pass
            color_value = border_info.get('color') or border_info.get('colour') or border_info.get('foreground')
            color = _parse_color(color_value) if color_value is not None else None
            if color is None:
                color = {'red': 0, 'green': 0, 'blue': 0}
            border['color'] = color
            return border

        if isinstance(borders_spec, str):
            border = build_border({'style': borders_spec})
            for side in sides_all:
                borders[side] = dict(border)
        elif isinstance(borders_spec, dict):
            normalized_borders = {_normalize_key(k): v for k, v in borders_spec.items()}
            if any(key in normalized_borders for key in ('style', 'color', 'colour', 'width', 'sides')):
                base_info = {
                    'style': normalized_borders.get('style', 'SOLID'),
                    'width': normalized_borders.get('width'),
                    'color': normalized_borders.get('color') or normalized_borders.get('colour'),
                }
                sides_value = normalized_borders.get('sides')
                if isinstance(sides_value, str):
                    sides = [sides_value]
                elif isinstance(sides_value, (list, tuple, set)):
                    sides = list(sides_value)
                else:
                    sides = sides_all
                resolved_sides = []
                for side in sides:
                    normalized_side = _normalize_side(str(side))
                    if normalized_side == 'all':
                        resolved_sides = sides_all
                        break
                    if normalized_side in sides_all:
                        resolved_sides.append(normalized_side)
                if not resolved_sides:
                    resolved_sides = sides_all
                border = build_border(base_info)
                for side in resolved_sides:
                    borders[side] = dict(border)
            else:
                for side_key, side_value in borders_spec.items():
                    normalized_side = _normalize_side(str(side_key))
                    if normalized_side in sides_all and isinstance(side_value, dict):
                        borders[normalized_side] = build_border(side_value)

        if borders:
            user_format['borders'] = borders
            for side in borders:
                mark(f'borders.{side}')

        if not user_format or not fields:
            raise ValueError("Не удалось распознать параметры форматирования")

        worksheet = self._resolve_worksheet(worksheet_name)
        sheet_id = getattr(worksheet, "id", None)
        if sheet_id is None:
            sheet_id = worksheet._properties.get("sheetId")
        if isinstance(sheet_id, str) and sheet_id.isdigit():
            sheet_id = int(sheet_id)
        grid_range = a1_range_to_grid_range(range_a1, sheet_id)
        self.spreadsheet.batch_update({
            "requests": [
                {
                    "repeatCell": {
                        "range": grid_range,
                        "cell": {
                            "userEnteredFormat": user_format
                        },
                        "fields": ','.join(sorted(fields))
                    }
                }
            ]
        })

        return f"Форматирование применено к диапазону {range_a1}"

    def clear_sheet(self, worksheet_name: Optional[str] = None):
        """Очистка листа"""
        worksheet = self._resolve_worksheet(worksheet_name)
        worksheet.clear()

    def get_table_info(self, worksheet_name: Optional[str] = None) -> Dict[str, Any]:
        """Получение информации о листе"""
        worksheet = self._resolve_worksheet(worksheet_name)
        data = worksheet.get_all_values()
        info = {
            'spreadsheet_id': self.sheet_id,
            'spreadsheet_title': self.spreadsheet.title,
            'worksheet': worksheet.title,
            'rows': len(data),
            'columns': len(data[0]) if data else 0,
            'has_data': len(data) > 0
        }
        worksheet_id = getattr(worksheet, "id", None)
        if worksheet_id is None:
            worksheet_id = worksheet._properties.get("sheetId")
        if worksheet_id is not None:
            info['worksheet_id'] = worksheet_id
            info['worksheet_url'] = (
                f"https://docs.google.com/spreadsheets/d/{self.sheet_id}/edit#gid={worksheet_id}"
            )
        return info

    def process_command(self, command: str, *, reset: bool = False) -> str:
        """Обработка команды через OpenAI для выполнения действий с таблицей.

        Args:
            command: Команда пользователя на естественном языке.
            reset: Если True, история переписки сбрасывается перед обработкой.
        """

        if reset:
            self.reset_chat()

        normalized_command = (command or "").strip()
        if not normalized_command:
            if reset:
                return "🧹 История диалога очищена. Чем помочь?"
            return "❌ Команда пуста. Опишите, что нужно сделать в Google Sheets."

        table_info = self.get_table_info()
        available_sheets = [ws.title for ws in self.spreadsheet.worksheets()]
        current_data = self.read_sheet_data()
        data_preview = current_data[:10] if current_data else []

        allowed_domains_text = ', '.join(self.allowed_domains) if self.allowed_domains else 'без ограничений'

        system_prompt = f"""Ты - помощник для работы с Google Sheets.
Информация о таблице:
- Spreadsheet ID: {table_info['spreadsheet_id']}
- Название документа: {table_info['spreadsheet_title']}
- Лист: {table_info['worksheet']}
- Доступные листы: {', '.join(available_sheets)}
- Строк: {table_info['rows']}
- Столбцов: {table_info['columns']}
- Текущие данные (первые 10 строк): {json.dumps(data_preview, ensure_ascii=False)}

Твоя задача - понять команду пользователя и вернуть JSON с действием.

Доступные действия:
1. READ - прочитать данные
2. WRITE - записать данные (перезаписывает лист)
3. APPEND - добавить строки в конец листа
4. UPDATE_CELL - обновить конкретную ячейку
5. CLEAR - очистить лист
6. INFO - показать информацию о листе
7. FETCH_URL - запросить текст (до {self.fetch_max_chars} символов) по ссылке из разрешенного списка доменов
8. WEB_SEARCH - выполнить поисковый запрос через SerpAPI и получить список релевантных ссылок
9. CHECK_ESTIMATE - проверить строительную смету и мастер-лист (требует два листа: смета и 'Master List')
10. FORMAT_RANGE - применить форматирование (цвет, текст, границы) к диапазону

Формат ответа (JSON):
{{
    "action": "READ|WRITE|APPEND|UPDATE_CELL|CLEAR|INFO|FETCH_URL|WEB_SEARCH|CHECK_ESTIMATE|FORMAT_RANGE",
    "data": [["header1", "header2"], ["row1col1", "row1col2"]],
    "title": "Название таблицы",
    "row": 0,
    "col": 0,
    "value": "новое значение",
    "url": "https://...",
    "query": "поисковый запрос",
    "estimate_sheet": "название листа со сметой",
    "master_sheet": "Master List",
    "quantity_col": "F",
    "worksheet": "Название листа (пример: \"Orcamento RU Viora Build\")",
    "range": "A1:B2",
    "format": {{"backgroundColor": "#FF0000", "textColor": "#ffffff", "bold": true, "borders": {{"style": "SOLID", "color": "#000000"}}}},
    "explanation": "Объяснение что будет сделано"
}}

Поля:
- data: массив строк для WRITE и APPEND (первая строка - заголовки)
- title: новое название документа (опционально, используется в WRITE)
- row, col, value: для UPDATE_CELL (индексы с 0)
- url: ссылка для FETCH_URL
- query: строка запроса для WEB_SEARCH
- estimate_sheet: название листа со сметой для CHECK_ESTIMATE (по умолчанию текущий лист)
- master_sheet: название листа с мастер-листом для CHECK_ESTIMATE (по умолчанию 'Master List')
- quantity_col: буква колонки с количеством для CHECK_ESTIMATE (по умолчанию 'F')
- worksheet: название листа, над которым нужно выполнить действие (по умолчанию текущий лист). Можно указывать приблизительное название — выбери наиболее подходящее.
- range: диапазон в A1-нотации для FORMAT_RANGE (например, "A1:B5")
- format: параметры форматирования для FORMAT_RANGE. Поддерживаются:
  - backgroundColor / fillColor: цвет фона (формат #RRGGBB или имя цвета, например "red")
  - textColor / fontColor: цвет текста
  - bold, italic, underline, strikethrough: булевы значения
  - fontSize, fontFamily: настройки шрифта
  - horizontalAlignment, verticalAlignment: выравнивание (LEFT/CENTER/RIGHT, TOP/MIDDLE/BOTTOM)
  - wrapStrategy: WRAP, OVERFLOW_CELL или CLIP
  - borders: {{ "style": "SOLID", "color": "#000000", "sides": ["top","bottom","left","right"] }} или отдельные стороны (top, bottom, left, right)
- explanation: всегда обязательно

FETCH_URL можно использовать несколько раз: после получения контента ты получишь сообщение пользователя с текстом страницы и сможешь принять конечное решение.
Если домен не входит в разрешенный список ({allowed_domains_text}), выбери другое действие.
WEB_SEARCH возвращает краткий список ссылок и описаний (до {self.search_max_results} результатов) и может вызываться ограниченное количество раз.
Для READ, CLEAR, INFO поля data, row, col, value не нужны.

Важно:
- Если запрос связан с анализом данных или форматирования текущего листа (например, «проанализируй первые строки», «покажи форматирование», «напомни что в смете»), используй действия READ, INFO, CHECK_ESTIMATE или FORMAT_RANGE. Не запускай WEB_SEARCH, если информация уже есть в таблице.
- Если пользователь упоминает название листа, даже приблизительно (например, «orcamento» вместо полного названия), подбери наиболее похожий лист и используй его в поле worksheet.
- WEB_SEARCH применяй только когда пользователь явно просит найти информацию в интернете, не связанную с текущей таблицей.
"""

        try:
            messages: List[Dict[str, str]] = [
                {"role": "system", "content": system_prompt}
            ]

            if self._chat_history:
                messages.extend(self._chat_history)

            messages.append({"role": "user", "content": normalized_command})

            fetch_rounds = 0

            while True:
                response = self.openai_client.chat.completions.create(
                    model=self.llm_model,
                    messages=messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "markdown_ai_action",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "action": {"type": "string"},
                                    "explanation": {"type": "string"},
                                },
                                "required": ["action"],
                                "additionalProperties": True,
                            },
                        },
                    },
                )

                raw_content = response.choices[0].message.content
                messages.append({"role": "assistant", "content": raw_content})

                try:
                    action_data = json.loads(raw_content)
                except json.JSONDecodeError:
                    final_response = f"❌ Модель вернула невалидный JSON: {raw_content}"
                    self._chat_history = [msg for msg in messages if msg["role"] != "system"]
                    self._chat_history.append({"role": "assistant", "content": final_response})
                    return final_response

                action = action_data.get('action', '').upper()

                if action == 'FETCH_URL':
                    if fetch_rounds >= self.fetch_max_rounds:
                        final_response = "❌ Превышено количество запросов FETCH_URL за одну команду"
                        self._chat_history = [msg for msg in messages if msg["role"] != "system"]
                        self._chat_history.append({"role": "assistant", "content": final_response})
                        return final_response

                    fetch_rounds += 1
                    url = action_data.get('url', '')
                    if not url:
                        final_response = "❌ Для FETCH_URL необходимо указать поле 'url'"
                        self._chat_history = [msg for msg in messages if msg["role"] != "system"]
                        self._chat_history.append({"role": "assistant", "content": final_response})
                        return final_response

                    try:
                        content = self.fetch_web_content(url)
                        messages.append({
                            "role": "user",
                            "content": (
                                f"Контент по ссылке {url} (символов: {len(content)}):\n{content}"
                            )
                        })
                    except Exception as fetch_error:
                        messages.append({
                            "role": "user",
                            "content": f"Ошибка при загрузке {url}: {fetch_error}"
                        })

                    continue

                result = self._execute_action(action_data)
                explanation = action_data.get('explanation', 'Выполнено')

                if not result:
                    final_response = f"✅ {explanation}"
                else:
                    normalized_result = str(result).strip()
                    if normalized_result.startswith("✅") or normalized_result.startswith("❌"):
                        final_response = normalized_result
                    else:
                        final_response = f"✅ {explanation}\n{normalized_result}"

                self._chat_history = [msg for msg in messages if msg["role"] != "system"]
                self._chat_history.append({"role": "assistant", "content": final_response})
                return final_response

        except GoogleAPIRateLimitError as rate_error:
            return f"❌ Ошибка {rate_error.code}: {rate_error.message}"
        except Exception as e:
            error_message = f"❌ Ошибка: {str(e)}"
            # Включаем текущую переписку (если есть) и последнюю реплику
            self._chat_history = [msg for msg in messages if msg["role"] != "system"] if 'messages' in locals() else []
            self._chat_history.append({"role": "assistant", "content": error_message})
            return error_message

    def _execute_action(self, action_data: Dict[str, Any]) -> str:
        """Выполнение действия на основе данных от OpenAI"""
        worksheet_hint = (
            action_data.get('worksheet')
            or action_data.get('sheet')
            or action_data.get('sheet_name')
        )
        action = action_data.get('action', '')
        if isinstance(action, str):
            action = action.strip().upper()
        else:
            action = str(action).strip().upper()

        if action == 'READ':
            data = self.read_sheet_data(worksheet_hint)
            if not data:
                return "Лист пустой"

            worksheet_title = self.worksheet.title
            header = data[0]
            body = data[1:]
            total_rows = len(data)
            data_rows = sum(1 for row in body if any(cell.strip() for cell in row))
            total_columns = len(header) if header else 0

            first_col_samples: List[str] = []
            seen_samples = set()
            for row in body:
                if not row:
                    continue
                value = row[0].strip()
                if value and value not in seen_samples:
                    first_col_samples.append(value)
                    seen_samples.add(value)
                if len(first_col_samples) >= 5:
                    break

            preview_body = [
                row for row in body
                if any(cell.strip() for cell in row)
            ][:min(15, len(body))]
            preview_data = [header] + preview_body if header else preview_body
            preview_table = tabulate(
                preview_data,
                headers="firstrow" if header else (),
                tablefmt="github"
            ) if preview_data else "Нет заполненных строк"

            summary_lines = [
                f"📈 **Анализ листа {worksheet_title}**",
                f"- Строк всего (включая заголовок): {total_rows}",
                f"- Строк с данными: {data_rows}",
                f"- Столбцов: {total_columns}",
            ]
            if first_col_samples:
                summary_lines.append(
                    "- Примеры значений первой колонки: "
                    + ", ".join(first_col_samples)
                )

            if preview_body:
                summary_lines.append("")
                summary_lines.append(
                    f"Предпросмотр (первые {len(preview_body)} строк с данными):"
                )
                summary_lines.append(preview_table)
            else:
                summary_lines.append("")
                summary_lines.append("Заполненные строки отсутствуют.")

            action_data['explanation'] = action_data.get(
                'explanation',
                f"Анализ листа {worksheet_title}"
            )
            return "\n".join(summary_lines)

        if action == 'WRITE':
            data = action_data.get('data', [])
            title = action_data.get('title')
            self.write_sheet_data(data, title=title, worksheet_name=worksheet_hint)
            return f"Лист перезаписан, строк: {len(data)}"

        if action == 'APPEND':
            data = action_data.get('data', [])
            self.append_to_sheet(data, worksheet_name=worksheet_hint)
            return f"Добавлено строк: {len(data)}"

        if action == 'UPDATE_CELL':
            row = action_data.get('row', 0)
            col = action_data.get('col', 0)
            value = action_data.get('value', '')
            self.update_cell(row, col, value, worksheet_name=worksheet_hint)
            cell_name = _cell_label(row, col)
            action_data['explanation'] = f"Изменение ячейки {cell_name}"
            return f"Ячейка {cell_name} обновлена значением '{value}'"

        if action == 'FORMAT_RANGE':
            range_a1 = (
                action_data.get('range')
                or action_data.get('range_a1')
                or action_data.get('a1')
                or action_data.get('target')
            )
            format_spec = action_data.get('format') or {}
            applied_range = self.format_range(range_a1, format_spec, worksheet_name=worksheet_hint)
            action_data['explanation'] = action_data.get(
                'explanation',
                f"Форматирование диапазона {range_a1}"
            )
            return applied_range

        if action == 'WEB_SEARCH':
            query = action_data.get('query') or action_data.get('search') or action_data.get('text')
            if not query or not str(query).strip():
                return "Не указан поисковый запрос"
            try:
                results = self.search_web(str(query))
            except Exception as exc:
                return f"Ошибка веб-поиска: {exc}"

            action_data['explanation'] = action_data.get(
                'explanation',
                f"Поиск информации: {query}"
            )

            if not results:
                return "Результаты не найдены"

            lines = ["🔍 Результаты поиска:"]
            for idx, item in enumerate(results, 1):
                title = item.get("title") or "(без заголовка)"
                link = item.get("link") or ""
                snippet = item.get("snippet") or ""
                lines.append(f"{idx}. {title}")
                if snippet:
                    lines.append(f"   {snippet}")
                if link:
                    lines.append(f"   🔗 {link}")
            return "\n".join(lines)

        if action == 'CLEAR':
            self.clear_sheet(worksheet_name=worksheet_hint)
            return "Лист очищен"

        if action == 'INFO':
            info = self.get_table_info(worksheet_hint)
            lines = [
                "📊 **Информация о таблице**",
                f"- Документ: {info['spreadsheet_title']} ({info['spreadsheet_id']})",
                f"- Текущий лист: {info['worksheet']}",
                f"- Строк: {info['rows']} • Столбцов: {info['columns']}",
                f"- Данные: {'есть' if info['has_data'] else 'нет'}",
            ]

            worksheet_url = info.get('worksheet_url')
            if worksheet_url:
                lines.append(f"🔗 <a href=\"{worksheet_url}\" target=\"_blank\">Открыть в Google Sheets</a>")

            try:
                other_sheets = [
                    ws.title
                    for ws in self.spreadsheet.worksheets()
                    if ws.title != self.worksheet_name
                ]
            except Exception:
                other_sheets = []

            if other_sheets:
                lines.append(f"📑 Другие листы: {', '.join(other_sheets)}")

            data = self.read_sheet_data()
            preview_limit = max(int(os.getenv('GOOGLE_SHEET_INFO_PREVIEW_ROWS', '10')), 1)

            if data:
                preview = data[:preview_limit]
                body_rows = max(len(preview) - 1, 0)
                if body_rows > 0:
                    lines.append(f"\n👀 Предпросмотр (первые {body_rows} строк + заголовок):")
                else:
                    lines.append("\n👀 Предпросмотр (только заголовок):")
                lines.append(tabulate(preview, headers="firstrow", tablefmt="github"))
            else:
                lines.append("\nЛист пустой")

            return "\n".join(lines)
        
        if action == 'CHECK_ESTIMATE':
            if not self.estimate_checker:
                return "❌ Модуль проверки смет недоступен. Убедитесь, что файл estimate_checker.py находится в директории проекта."
            
            # Получаем параметры
            estimate_sheet_name = action_data.get('estimate_sheet', self.worksheet_name)
            master_sheet_name = action_data.get('master_sheet', 'Master List')
            quantity_col = action_data.get('quantity_col', 'F')
            
            try:
                # Читаем данные из обоих листов
                estimate_worksheet = self.spreadsheet.worksheet(estimate_sheet_name)
                master_worksheet = self.spreadsheet.worksheet(master_sheet_name)
                
                estimate_data = estimate_worksheet.get_all_values()
                master_data = master_worksheet.get_all_values()
                
                # Выполняем проверку
                result = self.estimate_checker.validate_estimate(
                    estimate_data, 
                    master_data, 
                    quantity_col
                )
                
                # Форматируем отчет
                report = self.estimate_checker.format_validation_report(result)
                
                # Добавляем формулы
                formulas = self.estimate_checker.create_validation_formulas(quantity_col)
                report += "\n\n📋 ФОРМУЛЫ ДЛЯ ПРОВЕРКИ:\n\n"
                report += f"Статус проверки (добавьте в новую колонку):\n{formulas['status_check']}\n\n"
                report += f"Следующий код M-SHF:\n{formulas['next_shf_code']}\n\n"
                report += f"Следующий код M-WIN:\n{formulas['next_win_code']}\n\n"
                report += f"Проверка дубликатов:\n{formulas['duplicate_check']}\n\n"
                
                return report
                
            except Exception as e:
                return f"❌ Ошибка при проверке сметы: {str(e)}\n\nУбедитесь, что листы '{estimate_sheet_name}' и '{master_sheet_name}' существуют."

        return f"Неизвестное действие: {action}"


def main():
    """Основная функция для интерактивной работы"""
    print("🤖 Google Sheets AI - Управление таблицей через OpenAI")
    print("=" * 60)

    try:
        sheets_ai = GoogleSheetsAI()
        print(f"✅ Работаем с документом: {sheets_ai.spreadsheet.title} / {sheets_ai.worksheet_name}\n")

        print("Примеры команд:")
        print("- Создай таблицу с колонками: Имя, Возраст, Город")
        print("- Добавь строку: Иван, 25, Москва")
        print("- Прочитай таблицу")
        print("- Измени ячейку в первой строке второго столбца на 30")
        print("- Очисти таблицу")
        print("- Покажи информацию о таблице")
        print("- Используй сайт https://docs.python.org для заполнения данных")
        print("\nВведите 'exit' для выхода\n")

        while True:
            command = input("💬 Ваша команда: ").strip()

            if command.lower() in ['exit', 'quit', 'выход']:
                print("👋 До свидания!")
                break

            if not command:
                continue

            print("\n⏳ Обработка команды...\n")
            result = sheets_ai.process_command(command)
            print(result)
            print("\n" + "-" * 60 + "\n")

    except Exception as e:
        print(f"❌ Ошибка инициализации: {str(e)}")
        print("\nПроверьте:")
        print("1. Файл .env с OPENAI_API_KEY и доступом к Google Sheets")
        print("2. Путь к сервисному аккаунту (GOOGLE_SERVICE_ACCOUNT_FILE или GOOGLE_SERVICE_ACCOUNT_JSON)")
        print("3. Правильность идентификатора таблицы (GOOGLE_SHEET_ID)")


if __name__ == "__main__":
    main()
