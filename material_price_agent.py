"""Core logic for an AI-assisted material price procurement agent.

This module follows the architecture proposed in the project brief:
1. Pull material names/specifications from Google Sheets
2. Use OpenAI to understand each material and craft search intents
3. Gather pricing information (via LLM-assisted search or direct scraping)
4. Select the best supplier and write the result back to Google Sheets
"""

from __future__ import annotations

import json
import logging
import time
from urllib.parse import quote_plus
from dataclasses import dataclass
from textwrap import dedent
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import requests
from bs4 import BeautifulSoup
from google.oauth2.service_account import Credentials
import gspread
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

try:
    from site_scrapers import MultiSiteScraper, ProductInfo
    SCRAPERS_AVAILABLE = True
except ImportError:
    SCRAPERS_AVAILABLE = False
    logger.warning("site_scrapers module not available, real scraping disabled")

try:
    from gpt_scraper import MultiSiteGPTScraper
    GPT_SCRAPER_AVAILABLE = True
except ImportError:
    GPT_SCRAPER_AVAILABLE = False
    logger.warning("gpt_scraper module not available, GPT scraping disabled")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# URL validation helpers
# ---------------------------------------------------------------------------


def _validate_url(url: str, timeout: int = 5) -> bool:
    """Check if URL is accessible and returns 200 status."""
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        return response.status_code == 200
    except requests.RequestException:
        # Try GET if HEAD fails
        try:
            response = requests.get(url, timeout=timeout, allow_redirects=True)
            return response.status_code == 200
        except requests.RequestException:
            return False


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class MaterialQueryAnalysis(BaseModel):
    pt_name: str = Field(..., description="Material name translated into Portuguese.")
    search_queries: List[str] = Field(
        default_factory=list, description="Search queries to find supplier prices."
    )
    key_specs: List[str] = Field(
        default_factory=list, description="Key specifications to match against search results."
    )


class SupplierQuote(BaseModel):
    supplier: str
    price: str
    url: str
    notes: Optional[str] = None


class BestOffer(BaseModel):
    best_supplier: str
    price: str
    url: str
    reasoning: str


class MaterialResult(BaseModel):
    material_name: str
    analysis: MaterialQueryAnalysis
    quotes: List[SupplierQuote]
    best_offer: BestOffer


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _strip_code_fences(text: str) -> str:
    """Remove Markdown code fences if the LLM wrapped the payload."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        fence_end = cleaned.rfind("```")
        if fence_end != -1:
            cleaned = cleaned.split("\n", 1)[-1]
            cleaned = cleaned[:fence_end].strip()
    return cleaned


def _extract_json_from_text(text: str) -> str:
    """Extract JSON object or array from text that may contain extra content."""
    text = text.strip()
    
    # Try to find JSON object
    if '{' in text:
        start = text.find('{')
        # Find matching closing brace
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
    
    # Try to find JSON array
    if '[' in text:
        start = text.find('[')
        # Find matching closing bracket
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '[':
                depth += 1
            elif text[i] == ']':
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
    
    return text


def _parse_json_payload(payload: str) -> Any:
    cleaned = _strip_code_fences(payload)
    # Extract JSON from potential extra text
    json_str = _extract_json_from_text(cleaned)
    return json.loads(json_str)


class JsonParseError(RuntimeError):
    pass


class LLMValidationError(RuntimeError):
    pass


class SheetUpdateError(RuntimeError):
    pass


class SearchTimeoutError(RuntimeError):
    """Raised when material search exceeds allotted time budget."""
    pass


# ---------------------------------------------------------------------------
# Core agent implementation
# ---------------------------------------------------------------------------


@dataclass
class MaterialPriceAgent:
    """High-level orchestrator for fetching and evaluating supplier prices."""

    openai_api_key: Optional[str] = None
    openai_client: Optional[Any] = None
    google_service_account_path: Optional[str] = None
    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.2
    request_delay_seconds: float = 1.0
    enable_known_sites: bool = True
    known_sites_only: bool = False

    def __post_init__(self) -> None:
        logger.debug("Initializing OpenAI client with model %s", self.llm_model)
        if not self.openai_client and not self.openai_api_key:
            raise ValueError("Either openai_client or openai_api_key must be provided.")

        if self.openai_client is not None:
            self._client = self.openai_client
        else:
            self._client = OpenAI(api_key=self.openai_api_key)

        if self.known_sites_only and not self.enable_known_sites:
            logger.info(
                "Known-sites-only mode requested; enabling catalog search automatically."
            )
            self.enable_known_sites = True

        self._sheets = None
        if self.google_service_account_path:
            logger.debug(
                "Authorizing Google Sheets via service account %s",
                self.google_service_account_path,
            )
            creds = Credentials.from_service_account_file(
                self.google_service_account_path,
                scopes=["https://www.googleapis.com/auth/spreadsheets"],
            )
            self._sheets = gspread.authorize(creds)
        else:
            logger.info("Google Sheets integration disabled (no service account path provided)")

    # ------------------------------- LLM helpers --------------------------- #

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(JsonParseError),
        reraise=True,
    )
    def _structured_chat(self, system_prompt: str, user_prompt: str) -> Any:
        """Call the LLM and parse the response as JSON."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        logger.debug("Sending prompt to OpenAI model %s", self.llm_model)
        completion = self._client.chat.completions.create(
            model=self.llm_model,
            temperature=self.temperature,
            messages=messages,
        )
        raw_content = completion.choices[0].message.content
        logger.debug("Raw LLM response: %s", raw_content)

        try:
            payload = _parse_json_payload(raw_content or "")
        except json.JSONDecodeError as exc:
            raise JsonParseError(f"Failed to parse JSON from LLM response: {exc}") from exc

        return payload

    def _fetch_quotes(
        self,
        search_queries: Sequence[str],
        use_scraping: bool,
        use_real_scraping: bool = False,
        use_gpt_scraping: bool = False,
        deadline: Optional[float] = None,
    ) -> List[SupplierQuote]:
        """Fetch quotes using the configured strategy."""
        if use_gpt_scraping:
            return self.fetch_quotes_with_gpt(search_queries, deadline)
        if use_real_scraping:
            return self.fetch_quotes_from_real_sites(search_queries, deadline)
        if use_scraping:
            return self.fetch_quotes_via_scraping(search_queries, deadline)
        return self.fetch_quotes_via_llm(search_queries, deadline)

    def _collect_quotes(
        self,
        material_name: str,
        analysis: MaterialQueryAnalysis,
        use_scraping: bool,
        enable_known_sites: bool,
        known_sites_only: bool,
        use_real_scraping: bool = False,
        use_gpt_scraping: bool = False,
        deadline: Optional[float] = None,
    ) -> List[SupplierQuote]:
        """Try sequential strategies to collect supplier quotes."""
        if deadline and time.time() >= deadline:
            raise SearchTimeoutError(
                f"Time limit exceeded before collecting quotes for '{material_name}'"
            )

        if known_sites_only:
            if enable_known_sites:
                logger.info(
                    "Known-sites-only mode enabled for '%s'.",
                    material_name,
                )
                return self._search_known_site_quotes(material_name, analysis, deadline)
            logger.info(
                "Known-sites-only mode requested but catalog search disabled for '%s'.",
                material_name,
            )
            return []

        strategies: List[Tuple[str, Sequence[str]]] = [
            ("primary", analysis.search_queries),
            ("portuguese", self._build_portuguese_queries(material_name, analysis)),
            ("europe", self._build_european_queries(material_name, analysis)),
        ]

        for label, queries in strategies:
            if deadline and time.time() >= deadline:
                raise SearchTimeoutError(
                    f"Time limit exceeded during '{label}' search for '{material_name}'"
                )
            cleaned = [query.strip() for query in queries if query and query.strip()]
            if not cleaned:
                continue

            quotes = self._fetch_quotes(
                cleaned,
                use_scraping,
                use_real_scraping,
                use_gpt_scraping,
                deadline,
            )
            if quotes:
                if label != "primary":
                    logger.info(
                        "Quotes for '%s' found using %s strategy.",
                        material_name,
                        label,
                    )
                return quotes

        logger.info(
            "No direct quotes found for '%s' after all strategies.",
            material_name,
        )

        if enable_known_sites:
            if deadline and time.time() >= deadline:
                raise SearchTimeoutError(
                    f"Time limit exceeded before known-site search for '{material_name}'"
                )
            known_site_quotes = self._search_known_site_quotes(material_name, analysis, deadline)
            if known_site_quotes:
                logger.info(
                    "Quotes for '%s' found via known supplier catalogs.",
                    material_name,
                )
                return known_site_quotes

        return []

    def _build_portuguese_queries(
        self,
        material_name: str,
        analysis: MaterialQueryAnalysis,
    ) -> List[str]:
        """Generate fallback queries focused on Portuguese market."""
        base = analysis.pt_name or material_name
        return [
            f"{base} preço Portugal",
            f"comprar {base} fornecedor Portugal",
            f"{base} materiais construção preço em Portugal",
        ]

    def _build_european_queries(
        self,
        material_name: str,
        analysis: MaterialQueryAnalysis,
    ) -> List[str]:
        """Generate fallback queries for European market."""
        base_pt = analysis.pt_name or material_name
        return [
            f"{base_pt} preço Europa",
            f"{material_name} supplier Europe price",
            f"comprar {base_pt} online Europa",
        ]

    def _search_known_site_quotes(
        self,
        material_name: str,
        analysis: MaterialQueryAnalysis,
        deadline: Optional[float] = None,
    ) -> List[SupplierQuote]:
        """Search trusted supplier directories directly."""
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
            )
        }

        queries = [
            analysis.pt_name or "",
            material_name,
        ]
        results: List[SupplierQuote] = []
        seen_urls: set[str] = set()

        for query in queries:
            if deadline and time.time() >= deadline:
                raise SearchTimeoutError(
                    f"Time limit exceeded during known-site search for '{material_name}'"
                )
            if not query:
                continue
            for quote in self._scrape_europages(query, headers):
                if quote.url in seen_urls:
                    continue
                seen_urls.add(quote.url)
                results.append(quote)
            for quote in self._scrape_leroy_merlin(query, headers):
                if quote.url in seen_urls:
                    continue
                seen_urls.add(quote.url)
                results.append(quote)
            if results:
                break

        return results

    def _scrape_europages(self, query: str, headers: dict[str, str]) -> List[SupplierQuote]:
        """Scrape Europages listings for the given query."""
        quotes: List[SupplierQuote] = []
        slug = quote_plus((query or "").replace("/", " "))
        search_url = f"https://www.europages.pt/companies/Portugal/{slug}.html"
        try:
            response = requests.get(search_url, headers=headers, timeout=15)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.debug("Europages request failed for %s: %s", search_url, exc)
            return quotes

        soup = BeautifulSoup(response.text, "html.parser")
        cards = soup.select("article[data-company-id], li.result, div.company-card")
        if not cards:
            cards = soup.select("section.company-result, div[itemtype='http://schema.org/Organization']")

        for card in cards[:3]:
            name_el = card.select_one("h2, h3, span.company-card__name, p.company-name")
            link_el = card.find("a", href=True)
            snippet_el = card.select_one("p, span.company-card__description, div.result-summary")
            if not name_el or not link_el:
                continue

            supplier_name = name_el.get_text(strip=True)
            url = link_el["href"]
            if url.startswith("/"):
                url = f"https://www.europages.pt{url}"

            notes = snippet_el.get_text(strip=True) if snippet_el else "Listing from Europages"

            quotes.append(
                SupplierQuote(
                    supplier=supplier_name or "Europages",
                    price="По запросу",
                    url=url,
                    notes=notes,
                )
            )

        return quotes

    def _scrape_leroy_merlin(self, query: str, headers: dict[str, str]) -> List[SupplierQuote]:
        """Scrape Leroy Merlin Portugal listings for the given query."""
        quotes: List[SupplierQuote] = []
        if not query:
            return quotes

        search_url = f"https://www.leroymerlin.pt/search?q={quote_plus(query)}"
        try:
            response = requests.get(search_url, headers=headers, timeout=15)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.debug("Leroy Merlin request failed for %s: %s", search_url, exc)
            return quotes

        soup = BeautifulSoup(response.text, "html.parser")
        cards = soup.select("article[data-product], div.product-card, li.product-item")
        if not cards:
            cards = soup.select("div[data-testid='product-card'], div.card-product")

        for card in cards[:5]:
            name_el = card.select_one("h2, h3, p.product-name, span[data-testid='product-name']")
            link_el = card.find("a", href=True)
            price_el = card.select_one("span.price, span[data-testid='product-price']")
            if not name_el or not link_el:
                continue

            name = name_el.get_text(strip=True)
            url = link_el["href"]
            if url.startswith("/"):
                url = f"https://www.leroymerlin.pt{url}"
            price_text = price_el.get_text(strip=True) if price_el else "По запросу"

            quotes.append(
                SupplierQuote(
                    supplier="Leroy Merlin",
                    price=price_text,
                    url=url,
                    notes=name,
                )
            )

        return quotes

    def _generate_alternative_offer(
        self,
        material_name: str,
        analysis: MaterialQueryAnalysis,
        deadline: Optional[float] = None,
    ) -> Optional[Tuple[List[SupplierQuote], BestOffer]]:
        """Generate an alternative supplier suggestion when no direct quotes found."""
        system_prompt = (
            "Ты закупщик строительных материалов в Европе. Если прямой поставщик не найден, "
            "предложи доступную альтернативу с ценой и ссылкой. Отвечай только JSON."
        )
        user_prompt = dedent(
            f"""
            Исходный материал: "{material_name}".
            Перевод на португальский: "{analysis.pt_name}".
            Важные характеристики: {json.dumps(analysis.key_specs, ensure_ascii=False)}.

            Найди ближайший доступный товар у проверенного европейского поставщика.
            Требования:
            - Укажи название альтернативного материала (если отличается).
            - Приведи цену с валютой и ссылку на страницу товара.
            - Кратко обоснуй выбор.

            Формат ответа (JSON):
            {{
              "alternative_material": "...",
              "supplier": "...",
              "price": "...",
              "url": "...",
              "reasoning": "...",
              "notes": "..."
            }}
            """
        ).strip()

        if deadline and time.time() >= deadline:
            raise SearchTimeoutError(
                f"Time limit exceeded before generating alternative for '{material_name}'"
            )

        try:
            payload = self._structured_chat(system_prompt, user_prompt)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to generate alternative for '%s': %s", material_name, exc)
            return None

        supplier = payload.get("supplier") if isinstance(payload, dict) else None
        price = payload.get("price") if isinstance(payload, dict) else None
        url = payload.get("url") if isinstance(payload, dict) else None

        if not supplier or not price or not url:
            logger.warning(
                "Alternative suggestion for '%s' missing required fields: %s",
                material_name,
                payload,
            )
            return None

        alternative_material = payload.get("alternative_material") or material_name
        notes = payload.get("notes") or ""
        note_text = notes.strip()
        if alternative_material and alternative_material.lower() != material_name.lower():
            alt_note = f"Alternative material: {alternative_material}"
            note_text = f"{note_text} {alt_note}".strip()

        quote = SupplierQuote(
            supplier=supplier,
            price=price,
            url=url,
            notes=note_text or None,
        )
        best_offer = BestOffer(
            best_supplier=supplier,
            price=price,
            url=url,
            reasoning=payload.get("reasoning") or "Предложена альтернативная позиция.",
        )

        logger.info("Using alternative supplier '%s' for material '%s'", supplier, material_name)
        return [quote], best_offer

    def _resolve_material(
        self,
        material_name: str,
        analysis: MaterialQueryAnalysis,
        use_scraping: bool,
        enable_known_sites: Optional[bool] = None,
        known_sites_only: Optional[bool] = None,
        use_real_scraping: bool = False,
        use_gpt_scraping: bool = False,
        deadline: Optional[float] = None,
    ) -> MaterialResult:
        """Resolve material by applying fallback strategies and selecting best offer."""
        known_flag = self.enable_known_sites if enable_known_sites is None else enable_known_sites
        known_only_flag = self.known_sites_only if known_sites_only is None else known_sites_only
        quotes = self._collect_quotes(
            material_name,
            analysis,
            use_scraping,
            known_flag,
            known_only_flag,
            use_real_scraping,
            use_gpt_scraping,
            deadline,
        )

        if quotes:
            best_offer = self.select_best_offer(quotes, analysis.key_specs, deadline)
        else:
            if known_only_flag:
                logger.info(
                    "Known-sites-only mode: no listings found for '%s'.",
                    material_name,
                )
                best_offer = BestOffer(
                    best_supplier="N/A",
                    price="N/A",
                    url="N/A",
                    reasoning="Каталоги поставщиков не содержат подходящих предложений.",
                )
                quotes = []
            else:
                if deadline and time.time() >= deadline:
                    raise SearchTimeoutError(
                        f"Time limit exceeded before generating alternative for '{material_name}'"
                    )
                alternative = self._generate_alternative_offer(material_name, analysis, deadline)
                if alternative:
                    quotes, best_offer = alternative
                else:
                    best_offer = BestOffer(
                        best_supplier="N/A",
                        price="N/A",
                        url="N/A",
                        reasoning="Подходящие предложения не найдены даже после расширенного поиска.",
                    )
                    quotes = []

        return MaterialResult(
            material_name=material_name,
            analysis=analysis,
            quotes=quotes,
            best_offer=best_offer,
        )

    # --------------------------- Google Sheets ---------------------------- #

    def _fetch_materials(self, sheet_id: str) -> List[str]:
        if not self._sheets:
            raise RuntimeError("Google Sheets integration is disabled.")
        sheet = self._sheets.open_by_key(sheet_id).sheet1
        materials = sheet.col_values(1)
        logger.info("Fetched %d rows from sheet", len(materials))
        return [m.strip() for m in materials[1:] if m.strip()]

    def _update_sheet_row(self, sheet_id: str, row_index: int, offer: BestOffer) -> None:
        if not self._sheets:
            raise RuntimeError("Google Sheets integration is disabled.")
        sheet = self._sheets.open_by_key(sheet_id).sheet1
        try:
            sheet.update_cell(row_index, 3, offer.price)
            sheet.update_cell(row_index, 4, offer.best_supplier)
            sheet.update_cell(row_index, 5, offer.url)
            sheet.update_cell(row_index, 6, offer.reasoning)
        except Exception as exc:  # noqa: BLE001
            raise SheetUpdateError(f"Failed to update sheet row {row_index}: {exc}") from exc

    # ----------------------------- LLM workflows -------------------------- #

    def analyze_material(self, material_name: str) -> MaterialQueryAnalysis:
        system_prompt = (
            "Ты эксперт по строительным материалам и закупкам в Португалии. "
            "Отвечай только валидным JSON, без пояснений."
        )
        user_prompt = dedent(
            f"""
            Проанализируй материал "{material_name}".
            Требования:
            1. Переведи название на португальский язык.
            2. Сформируй 3-5 поисковых запросов, ориентированных на поставщиков в Португалии.
            3. Перечисли ключевые характеристики, которые важно учитывать при сравнении предложений.

            Формат JSON:
            {{
              "pt_name": "...",
              "search_queries": ["..."],
              "key_specs": ["..."]
            }}
            """
        ).strip()

        payload = self._structured_chat(system_prompt, user_prompt)
        try:
            return MaterialQueryAnalysis.model_validate(payload)
        except ValidationError as exc:
            raise LLMValidationError(str(exc)) from exc

    def fetch_quotes_via_llm(
        self,
        search_queries: Sequence[str],
        deadline: Optional[float] = None,
    ) -> List[SupplierQuote]:
        quotes: List[SupplierQuote] = []
        system_prompt = (
            "Ты участвуешь в поиске поставщиков строительных материалов в Португалии. "
            "Отвечай только JSON."
        )

        for query in search_queries:
            if deadline and time.time() >= deadline:
                raise SearchTimeoutError(
                    f"Time limit exceeded during LLM quote search for query '{query}'"
                )
            user_prompt = dedent(
                f"""
                Найди актуальные цены в Португалии по запросу "{query}".
                Требуется минимум 3 предложения. Форматируй ответ как массив JSON-объектов:
                [
                  {{
                    "supplier": "...",
                    "price": "...",
                    "url": "...",
                    "notes": "..."
                  }}
                ]
                """
            ).strip()
            try:
                payload = self._structured_chat(system_prompt, user_prompt)
            except JsonParseError as exc:
                logger.warning("Failed to parse quotes for query '%s': %s", query, exc)
                continue

            entries = payload if isinstance(payload, list) else [payload]
            for entry in entries:
                try:
                    quotes.append(SupplierQuote.model_validate(entry))
                except ValidationError as exc:
                    logger.debug("Discarding invalid quote payload: %s", exc)
                    continue
            if not entries:
                logger.warning("LLM returned no quotes for query '%s'", query)
            if deadline and time.time() >= deadline:
                raise SearchTimeoutError(
                    f"Time limit exceeded after processing LLM quotes for query '{query}'"
                )
            time.sleep(self.request_delay_seconds)
            if deadline and time.time() >= deadline:
                raise SearchTimeoutError(
                    f"Time limit exceeded after delay when searching quotes for query '{query}'"
                )

        # Deduplicate by supplier + url combination
        unique = {}
        for quote in quotes:
            key = (quote.supplier.lower(), quote.url)
            unique[key] = quote
        return list(unique.values())

    def fetch_quotes_via_scraping(
        self,
        search_queries: Sequence[str],
        deadline: Optional[float] = None,
    ) -> List[SupplierQuote]:
        suppliers = [
            "https://www.leroymerlin.pt/",
            "https://www.aki.pt/",
            "https://www.bricomarche.pt/",
        ]
        special_suppliers = {
            "https://www.europages.pt/": self._scrape_europages,
            "https://www.leroymerlin.pt/": self._scrape_leroy_merlin,
        }
        quotes: List[SupplierQuote] = []

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
            )
        }

        for query in search_queries:
            if deadline and time.time() >= deadline:
                raise SearchTimeoutError(
                    f"Time limit exceeded during scraping for query '{query}'"
                )
            for supplier_url in suppliers:
                search_url = f"{supplier_url.rstrip('/')}/search?q={quote_plus(query)}"
                try:
                    response = requests.get(search_url, headers=headers, timeout=15)
                    response.raise_for_status()
                except requests.RequestException as exc:
                    logger.debug("Request failed for %s: %s", search_url, exc)
                    continue

                soup = BeautifulSoup(response.text, "html.parser")
                products = soup.select("div.product")
                for product in products[:3]:
                    name_el = product.select_one(".product-name")
                    price_el = product.select_one(".price")
                    link_el = product.find("a", href=True)
                    if not name_el or not link_el:
                        continue
                    price_text = price_el.get_text(strip=True) if price_el else "По запросу"
                    quotes.append(
                        SupplierQuote(
                            supplier=supplier_url,
                            price=price_text,
                            url=link_el["href"],
                            notes=name_el.get_text(strip=True),
                        )
                    )

            for base_url, scraper in special_suppliers.items():
                try:
                    quotes.extend(scraper(query, headers))
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "Special scraper '%s' failed for query '%s': %s",
                        base_url,
                        query,
                        exc,
                    )
            if deadline and time.time() >= deadline:
                raise SearchTimeoutError(
                    f"Time limit exceeded after scraping query '{query}'"
                )
            time.sleep(self.request_delay_seconds)
            if deadline and time.time() >= deadline:
                raise SearchTimeoutError(
                    f"Time limit exceeded after delay when scraping query '{query}'"
                )
        return quotes

    def fetch_quotes_from_real_sites(
        self,
        search_queries: Sequence[str],
        deadline: Optional[float] = None,
    ) -> List[SupplierQuote]:
        """Fetch quotes by scraping real supplier websites."""
        if not SCRAPERS_AVAILABLE:
            logger.warning("Site scrapers not available, falling back to LLM search")
            return self.fetch_quotes_via_llm(search_queries, deadline)
        
        quotes: List[SupplierQuote] = []
        scraper = MultiSiteScraper()
        
        for query in search_queries:
            if deadline and time.time() >= deadline:
                raise SearchTimeoutError(
                    f"Time limit exceeded during real-site scraping for query '{query}'"
                )
            logger.info("Scraping real sites for query: %s", query)
            try:
                products = scraper.search_all(query)
                
                for product in products:
                    # Validate URL before adding
                    if _validate_url(product.url):
                        quotes.append(SupplierQuote(
                            supplier=product.supplier,
                            price=product.price,
                            url=product.url,
                            notes=product.description or product.name,
                        ))
                    else:
                        logger.debug("Skipping invalid URL: %s", product.url)
                
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to scrape sites for query '%s': %s", query, exc)
                continue
            
            if deadline and time.time() >= deadline:
                raise SearchTimeoutError(
                    f"Time limit exceeded after real-site scraping for query '{query}'"
                )
            time.sleep(self.request_delay_seconds)
            if deadline and time.time() >= deadline:
                raise SearchTimeoutError(
                    f"Time limit exceeded after delay when scraping real sites for query '{query}'"
                )
        
        # Deduplicate
        unique = {}
        for quote in quotes:
            key = (quote.supplier.lower(), quote.url)
            unique[key] = quote
        
        logger.info("Found %d unique quotes from real sites", len(unique))
        if not unique:
            logger.warning("No quotes found via real-site scraping; falling back to LLM search")
            return self.fetch_quotes_via_llm(search_queries, deadline)
        return list(unique.values())

    def fetch_quotes_with_gpt(
        self,
        search_queries: Sequence[str],
        deadline: Optional[float] = None,
    ) -> List[SupplierQuote]:
        """Fetch quotes using GPT-powered web scraping."""
        if not GPT_SCRAPER_AVAILABLE:
            logger.warning("GPT scraper not available, falling back to LLM search")
            return self.fetch_quotes_via_llm(search_queries, deadline)
        
        quotes: List[SupplierQuote] = []
        scraper = MultiSiteGPTScraper(
            openai_api_key=self.openai_api_key,
            openai_client=self._client,
            model=self.llm_model,
        )
        
        for query in search_queries:
            if deadline and time.time() >= deadline:
                raise SearchTimeoutError(
                    f"Time limit exceeded during GPT scraping for query '{query}'"
                )
            logger.info("Using GPT to scrape sites for query: %s", query)
            try:
                products = scraper.search_all(query)
                
                for product in products:
                    # Validate URL before adding
                    if _validate_url(product.url):
                        quotes.append(SupplierQuote(
                            supplier=product.supplier,
                            price=product.price,
                            url=product.url,
                            notes=product.description or product.name,
                        ))
                    else:
                        logger.debug("Skipping invalid URL: %s", product.url)
                
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to GPT-scrape for query '%s': %s", query, exc)
                continue
            
            if deadline and time.time() >= deadline:
                raise SearchTimeoutError(
                    f"Time limit exceeded after GPT scraping for query '{query}'"
                )
            time.sleep(self.request_delay_seconds)
            if deadline and time.time() >= deadline:
                raise SearchTimeoutError(
                    f"Time limit exceeded after delay when GPT-scraping for query '{query}'"
                )
        
        # Deduplicate
        unique = {}
        for quote in quotes:
            key = (quote.supplier.lower(), quote.url)
            unique[key] = quote
        
        logger.info("Found %d unique quotes via GPT scraping", len(unique))
        if not unique:
            logger.warning("No quotes found via GPT scraping; falling back to LLM search")
            return self.fetch_quotes_via_llm(search_queries, deadline)
        return list(unique.values())

    def select_best_offer(
        self,
        quotes: Sequence[SupplierQuote],
        specs: Sequence[str],
        deadline: Optional[float] = None,
    ) -> BestOffer:
        # Filter quotes with valid URLs
        valid_quotes = []
        for quote in quotes:
            if quote.url and quote.url != "N/A":
                logger.debug("Validating URL: %s", quote.url)
                if _validate_url(quote.url):
                    valid_quotes.append(quote)
                    logger.debug("URL is valid: %s", quote.url)
                else:
                    logger.warning("URL is not accessible, skipping: %s", quote.url)
            else:
                valid_quotes.append(quote)
        
        # If no valid quotes, return all quotes
        if not valid_quotes:
            logger.warning("No quotes with valid URLs found, using all quotes")
            valid_quotes = list(quotes)
        
        system_prompt = (
            "Ты закупщик материалов. Сравни предложения и выбери лучшее. "
            "Ответ только в JSON формате."
        )
        user_prompt = dedent(
            f"""
            Предложения:
            {json.dumps([quote.model_dump() for quote in valid_quotes], ensure_ascii=False)}

            Требования:
            {json.dumps(list(specs), ensure_ascii=False)}

            Возьми во внимание цену, репутацию поставщика и соответствие характеристикам.
            Формат JSON:
            {{
              "best_supplier": "...",
              "price": "...",
              "url": "...",
              "reasoning": "..."
            }}
            """
        ).strip()

        if deadline and time.time() >= deadline:
            raise SearchTimeoutError("Time limit exceeded before selecting best offer")

        payload = self._structured_chat(system_prompt, user_prompt)
        try:
            return BestOffer.model_validate(payload)
        except ValidationError as exc:
            raise LLMValidationError(str(exc)) from exc

    # -------------------------- Public interface -------------------------- #

    def process_materials(
        self,
        materials: Sequence[str],
        use_scraping: bool = False,
        enable_known_sites: Optional[bool] = None,
        known_sites_only: Optional[bool] = None,
        use_real_scraping: bool = False,
        use_gpt_scraping: bool = False,
    ) -> Iterable[MaterialResult]:
        """
        Process in-memory list of materials. Useful for chat/UI flows without Google Sheets.
        """
        known_flag = self.enable_known_sites if enable_known_sites is None else enable_known_sites
        known_only_flag = self.known_sites_only if known_sites_only is None else known_sites_only
        for material in materials:
            logger.info("Processing material '%s'", material)
            analysis = self.analyze_material(material)
            yield self._resolve_material(
                material,
                analysis,
                use_scraping,
                known_flag,
                known_only_flag,
                use_real_scraping,
                use_gpt_scraping,
            )

    def process_sheet(
        self,
        sheet_id: str,
        use_scraping: bool = False,
        enable_known_sites: Optional[bool] = None,
        known_sites_only: Optional[bool] = None,
        use_real_scraping: bool = False,
        use_gpt_scraping: bool = False,
    ) -> Iterable[MaterialResult]:
        materials = self._fetch_materials(sheet_id)
        logger.info("Processing %d materials", len(materials))
        known_flag = self.enable_known_sites if enable_known_sites is None else enable_known_sites
        known_only_flag = self.known_sites_only if known_sites_only is None else known_sites_only

        for row_offset, material in enumerate(materials, start=2):
            logger.info("Processing material '%s' (row %d)", material, row_offset)
            analysis = self.analyze_material(material)
            result = self._resolve_material(
                material,
                analysis,
                use_scraping,
                known_flag,
                known_only_flag,
                use_real_scraping,
                use_gpt_scraping,
            )

            if result.best_offer.best_supplier != "N/A":
                self._update_sheet_row(sheet_id, row_offset, result.best_offer)

            if not result.quotes:
                logger.warning("No viable offers for '%s' even after fallbacks", material)
                continue

            yield result


def results_to_markdown_table(results: Sequence[MaterialResult]) -> str:
    """Render material results as Markdown table."""
    if not results:
        return "Нет данных для отображения."

    header = (
        "| Материал | Название на португальском | Лучшая цена | Поставщик | Ссылка | "
        "Обоснование |\n"
        "|---|---|---|---|---|---|"
    )
    rows = [
        f"| {_escape_markdown_cell(result.material_name)}"
        f" | {_escape_markdown_cell(result.analysis.pt_name)}"
        f" | {_escape_markdown_cell(result.best_offer.price)}"
        f" | {_escape_markdown_cell(result.best_offer.best_supplier)}"
        f" | {_format_url_cell(result.best_offer.url)}"
        f" | {_escape_markdown_cell(result.best_offer.reasoning)} |"
        for result in results
    ]
    return "\n".join([header, *rows])


def _format_url_cell(url: str) -> str:
    """Format URL for Markdown table."""
    if not url or url == "N/A":
        return "-"
    return f"[ссылка]({url})"


def _escape_markdown_cell(value: Optional[str]) -> str:
    """Escape Markdown table cell value."""
    if value is None:
        return "-"
    return value.replace("|", "\\|").replace("\n", " ")
