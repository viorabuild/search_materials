"""Web scrapers for specific supplier websites in Portugal.

This module contains specialized scrapers for known construction material suppliers.
Each scraper extracts real prices and product information from the website.
"""

from __future__ import annotations

import logging
import re
import time
from typing import List, Optional
from urllib.parse import quote_plus, urljoin

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

# Try to import known suppliers database
try:
    from known_suppliers import find_material as find_known_material
    KNOWN_DB_AVAILABLE = True
except ImportError:
    KNOWN_DB_AVAILABLE = False
    logger.warning("known_suppliers module not available")


class ProductInfo(BaseModel):
    """Product information scraped from a website."""
    name: str
    price: str
    url: str
    supplier: str
    description: Optional[str] = None
    in_stock: bool = True


class SiteScraper:
    """Base class for site-specific scrapers."""
    
    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "pt-PT,pt;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Cache-Control": "max-age=0",
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def search(self, query: str) -> List[ProductInfo]:
        """Search for products matching the query."""
        raise NotImplementedError
    
    def _make_request(self, url: str, retries: int = 3) -> Optional[requests.Response]:
        """Make HTTP request with retry logic."""
        import time
        
        for attempt in range(retries):
            try:
                response = self.session.get(
                    url, 
                    timeout=self.timeout,
                    allow_redirects=True,
                    verify=False  # Disable SSL verification for problematic sites
                )
                response.raise_for_status()
                return response
            except requests.exceptions.SSLError as exc:
                logger.warning("SSL error on attempt %d/%d for %s: %s", attempt + 1, retries, url, exc)
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue
            except requests.exceptions.HTTPError as exc:
                if exc.response.status_code == 403:
                    logger.warning("403 Forbidden - site may be blocking scrapers: %s", url)
                    return None
                logger.warning("HTTP error on attempt %d/%d: %s", attempt + 1, retries, exc)
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                continue
            except requests.RequestException as exc:
                logger.warning("Request failed on attempt %d/%d: %s", attempt + 1, retries, exc)
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                continue
        
        return None


class LeroyMerlinScraper(SiteScraper):
    """Scraper for Leroy Merlin Portugal."""
    
    BASE_URL = "https://www.leroymerlin.pt"
    
    def search(self, query: str) -> List[ProductInfo]:
        """Search Leroy Merlin for products."""
        products = []
        search_url = f"{self.BASE_URL}/search?q={quote_plus(query)}"
        
        response = self._make_request(search_url)
        if not response:
            return products
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Try multiple selectors for product cards
        product_cards = soup.select("article[data-product]")
        if not product_cards:
            product_cards = soup.select("div.product-card, li.product-item")
        
        for card in product_cards[:10]:  # Limit to 10 products
            try:
                # Extract product name
                name_el = card.select_one("h2, h3, p.product-name, span[data-testid='product-name']")
                if not name_el:
                    continue
                name = name_el.get_text(strip=True)
                
                # Extract price
                price_el = card.select_one("span.price, span[data-testid='product-price'], div.price")
                if not price_el:
                    continue
                price_text = price_el.get_text(strip=True)
                
                # Extract URL
                link_el = card.find("a", href=True)
                if not link_el:
                    continue
                url = link_el["href"]
                if url.startswith("/"):
                    url = urljoin(self.BASE_URL, url)
                
                # Extract description
                desc_el = card.select_one("p.description, div.product-description")
                description = desc_el.get_text(strip=True) if desc_el else None
                
                products.append(ProductInfo(
                    name=name,
                    price=price_text,
                    url=url,
                    supplier="Leroy Merlin",
                    description=description,
                ))
                
            except Exception as exc:
                logger.debug("Failed to parse product card: %s", exc)
                continue
        
        logger.info("Found %d products from Leroy Merlin for query '%s'", len(products), query)
        return products


class AKIScraper(SiteScraper):
    """Scraper for AKI Portugal."""
    
    BASE_URL = "https://www.aki.pt"
    
    def search(self, query: str) -> List[ProductInfo]:
        """Search AKI for products."""
        products = []
        search_url = f"{self.BASE_URL}/pesquisa?q={quote_plus(query)}"
        
        response = self._make_request(search_url)
        if not response:
            return products
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Product selectors for AKI
        product_cards = soup.select("div.product-item, article.product, div.product-card")
        
        for card in product_cards[:10]:
            try:
                name_el = card.select_one("h2, h3, div.product-name, span.name")
                if not name_el:
                    continue
                name = name_el.get_text(strip=True)
                
                price_el = card.select_one("span.price, div.price, span.value")
                if not price_el:
                    continue
                price_text = price_el.get_text(strip=True)
                
                link_el = card.find("a", href=True)
                if not link_el:
                    continue
                url = link_el["href"]
                if url.startswith("/"):
                    url = urljoin(self.BASE_URL, url)
                
                products.append(ProductInfo(
                    name=name,
                    price=price_text,
                    url=url,
                    supplier="AKI",
                ))
                
            except Exception as exc:
                logger.debug("Failed to parse AKI product card: %s", exc)
                continue
        
        logger.info("Found %d products from AKI for query '%s'", len(products), query)
        return products


class BricomarcheScraper(SiteScraper):
    """Scraper for Bricomarche Portugal."""
    
    BASE_URL = "https://www.bricomarche.pt"
    
    def search(self, query: str) -> List[ProductInfo]:
        """Search Bricomarche for products."""
        products = []
        search_url = f"{self.BASE_URL}/pesquisa?q={quote_plus(query)}"
        
        response = self._make_request(search_url)
        if not response:
            return products
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        product_cards = soup.select("div.product, article.product-item, div.product-card")
        
        for card in product_cards[:10]:
            try:
                name_el = card.select_one("h2, h3, div.name, span.product-name")
                if not name_el:
                    continue
                name = name_el.get_text(strip=True)
                
                price_el = card.select_one("span.price, div.price")
                if not price_el:
                    continue
                price_text = price_el.get_text(strip=True)
                
                link_el = card.find("a", href=True)
                if not link_el:
                    continue
                url = link_el["href"]
                if url.startswith("/"):
                    url = urljoin(self.BASE_URL, url)
                
                products.append(ProductInfo(
                    name=name,
                    price=price_text,
                    url=url,
                    supplier="Bricomarche",
                ))
                
            except Exception as exc:
                logger.debug("Failed to parse Bricomarche product card: %s", exc)
                continue
        
        logger.info("Found %d products from Bricomarche for query '%s'", len(products), query)
        return products


class EuropagesScraper(SiteScraper):
    """Scraper for Europages Portugal."""
    
    BASE_URL = "https://www.europages.pt"
    
    def search(self, query: str) -> List[ProductInfo]:
        """Search Europages for suppliers."""
        products = []
        slug = quote_plus(query.replace("/", " "))
        search_url = f"{self.BASE_URL}/companies/Portugal/{slug}.html"
        
        response = self._make_request(search_url)
        if not response:
            return products
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Europages company cards
        cards = soup.select("article[data-company-id], li.result, div.company-card")
        if not cards:
            cards = soup.select("section.company-result, div[itemtype='http://schema.org/Organization']")
        
        for card in cards[:5]:  # Limit to 5 suppliers
            try:
                name_el = card.select_one("h2, h3, span.company-card__name, p.company-name")
                link_el = card.find("a", href=True)
                snippet_el = card.select_one("p, span.company-card__description, div.result-summary")
                
                if not name_el or not link_el:
                    continue
                
                supplier_name = name_el.get_text(strip=True)
                url = link_el["href"]
                if url.startswith("/"):
                    url = f"{self.BASE_URL}{url}"
                
                description = snippet_el.get_text(strip=True) if snippet_el else "Supplier from Europages"
                
                products.append(ProductInfo(
                    name=supplier_name,
                    price="По запросу",  # Europages doesn't show prices
                    url=url,
                    supplier="Europages",
                    description=description,
                ))
                
            except Exception as exc:
                logger.debug("Failed to parse Europages card: %s", exc)
                continue
        
        logger.info("Found %d suppliers from Europages for query '%s'", len(products), query)
        return products


class GenericScraper(SiteScraper):
    """Generic scraper that uses DuckDuckGo search to find products."""
    
    def search(self, query: str) -> List[ProductInfo]:
        """Search using DuckDuckGo and extract product info."""
        products = []
        
        # Use DuckDuckGo HTML search
        search_query = f"{query} Portugal preço site:leroymerlin.pt OR site:aki.pt OR site:bricomarche.pt"
        search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(search_query)}"
        
        response = self._make_request(search_url)
        if not response:
            return products
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Parse DuckDuckGo results
        results = soup.select("div.result, div.web-result")
        
        for result in results[:5]:
            try:
                link_el = result.select_one("a.result__a, a[href]")
                title_el = result.select_one("h2, .result__title")
                snippet_el = result.select_one("a.result__snippet, .result__snippet")
                
                if not link_el or not title_el:
                    continue
                
                url = link_el.get("href", "")
                if not url or url.startswith("/"):
                    continue
                
                title = title_el.get_text(strip=True)
                description = snippet_el.get_text(strip=True) if snippet_el else ""
                
                # Determine supplier from URL
                supplier = "Unknown"
                if "leroymerlin" in url:
                    supplier = "Leroy Merlin"
                elif "aki" in url:
                    supplier = "AKI"
                elif "bricomarche" in url:
                    supplier = "Bricomarche"
                
                products.append(ProductInfo(
                    name=title,
                    price="Consultar no site",
                    url=url,
                    supplier=supplier,
                    description=description,
                ))
                
            except Exception as exc:
                logger.debug("Failed to parse search result: %s", exc)
                continue
        
        logger.info("Found %d products via generic search for query '%s'", len(products), query)
        return products


class KnownSuppliersScraper(SiteScraper):
    """Scraper that uses known suppliers database as fallback."""
    
    def search(self, query: str) -> List[ProductInfo]:
        """Search in known suppliers database."""
        products = []
        
        if not KNOWN_DB_AVAILABLE:
            logger.warning("Known suppliers database not available")
            return products
        
        result = find_known_material(query)
        if not result:
            logger.info("Material '%s' not found in known suppliers database", query)
            return products
        
        for supplier in result.get("suppliers", []):
            products.append(ProductInfo(
                name=f"{result['material']} - {supplier['description']}",
                price=supplier['price'],
                url=supplier['url'],
                supplier=supplier['name'],
                description=f"Preço típico: {supplier['price']}. Contacte para confirmação.",
                in_stock=True,
            ))
        
        logger.info("Found %d suppliers from known database for query '%s'", len(products), query)
        return products


class MultiSiteScraper:
    """Scraper that searches multiple sites with rate limiting."""
    
    def __init__(self, delay_between_sites: float = 2.0, use_fallback: bool = True):
        self.scrapers = [
            EuropagesScraper(),  # Try Europages first (more reliable)
            LeroyMerlinScraper(),
            AKIScraper(),
            BricomarcheScraper(),
        ]
        if use_fallback:
            self.scrapers.append(GenericScraper())  # Fallback to generic search
            if KNOWN_DB_AVAILABLE:
                self.scrapers.append(KnownSuppliersScraper())  # Final fallback to known DB
        self.delay_between_sites = delay_between_sites
    
    def search_all(self, query: str) -> List[ProductInfo]:
        """Search all configured sites for products with rate limiting."""
        all_products = []
        
        for i, scraper in enumerate(self.scrapers):
            try:
                logger.info("Searching %s for '%s'...", scraper.__class__.__name__, query)
                products = scraper.search(query)
                all_products.extend(products)
                
                # Rate limiting - wait between sites
                if i < len(self.scrapers) - 1 and products:
                    logger.debug("Waiting %.1fs before next site...", self.delay_between_sites)
                    time.sleep(self.delay_between_sites)
                    
            except Exception as exc:
                logger.warning("Scraper %s failed: %s", scraper.__class__.__name__, exc)
                continue
        
        logger.info("Found total %d products across all sites for query '%s'", len(all_products), query)
        return all_products


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    scraper = MultiSiteScraper()
    products = scraper.search_all("cimento")
    
    for product in products:
        print(f"{product.supplier}: {product.name} - {product.price}")
        print(f"  URL: {product.url}")
