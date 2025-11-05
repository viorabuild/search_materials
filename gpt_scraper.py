"""GPT-powered web scraper that uses AI to extract product information from HTML.

This scraper uses OpenAI GPT to intelligently parse HTML pages and extract
product information, making it resilient to website structure changes.
"""

from __future__ import annotations

import logging
import time
from typing import Any, List, Optional
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup

from openai import OpenAI
from pydantic import BaseModel, Field
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


class ProductInfo(BaseModel):
    """Product information extracted by GPT."""
    name: str = Field(..., description="Product name")
    price: str = Field(..., description="Price with currency")
    url: str = Field(..., description="Product URL")
    supplier: str = Field(..., description="Supplier/store name")
    description: Optional[str] = Field(None, description="Product description")
    in_stock: bool = Field(True, description="Whether product is in stock")


class GPTScraper:
    """Web scraper that uses GPT to extract product information from HTML."""
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        timeout: int = 15,
        max_html_length: int = 8000,
        openai_client: Optional[Any] = None,
    ):
        if openai_client is None and not openai_api_key:
            raise ValueError("Either openai_client or openai_api_key must be provided.")
        self.client = openai_client or OpenAI(api_key=openai_api_key)
        self.model = model
        self.timeout = timeout
        self.max_html_length = max_html_length
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "pt-PT,pt;q=0.9,en-US;q=0.8,en;q=0.7",
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def search_site(self, base_url: str, query: str, supplier_name: str) -> List[ProductInfo]:
        """Search a website and extract products using GPT."""
        products = []
        
        # Construct search URL
        search_url = f"{base_url}/search?q={quote_plus(query)}"
        
        # Fetch HTML
        html_content = self._fetch_html(search_url)
        if not html_content:
            return products
        
        # Clean and truncate HTML
        cleaned_html = self._clean_html(html_content)
        
        # Use GPT to extract products
        products = self._extract_products_with_gpt(
            cleaned_html, 
            query, 
            supplier_name,
            base_url
        )
        
        return products
    
    def _fetch_html(self, url: str, retries: int = 3) -> Optional[str]:
        """Fetch HTML content with retry logic."""
        for attempt in range(retries):
            try:
                response = self.session.get(
                    url,
                    timeout=self.timeout,
                    allow_redirects=True,
                    verify=False
                )
                response.raise_for_status()
                return response.text
            except requests.exceptions.HTTPError as exc:
                if exc.response.status_code == 403:
                    logger.warning("403 Forbidden - trying with different headers")
                    # Try with minimal headers
                    try:
                        response = requests.get(url, timeout=self.timeout, verify=False)
                        if response.status_code == 200:
                            return response.text
                    except:
                        pass
                logger.warning("HTTP error on attempt %d/%d: %s", attempt + 1, retries, exc)
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
            except requests.RequestException as exc:
                logger.warning("Request failed on attempt %d/%d: %s", attempt + 1, retries, exc)
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        
        return None
    
    def _clean_html(self, html: str) -> str:
        """Clean HTML and extract relevant content."""
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style", "noscript", "iframe"]):
            script.decompose()
        
        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, str) and text.strip().startswith("<!--")):
            comment.extract()
        
        # Get text content with some structure
        text = soup.get_text(separator="\n", strip=True)
        
        # Truncate if too long
        if len(text) > self.max_html_length:
            text = text[:self.max_html_length] + "\n... (truncated)"
        
        return text
    
    def _extract_products_with_gpt(
        self,
        html_content: str,
        query: str,
        supplier_name: str,
        base_url: str
    ) -> List[ProductInfo]:
        """Use GPT to extract product information from HTML."""
        system_prompt = """You are a web scraping assistant. Extract product information from HTML content.
Focus on finding products that match the search query.
Return a JSON array of products with: name, price, url, description, in_stock.
If you can't find exact prices, use "Consultar no site" or "По запросу".
Make sure URLs are complete (add base_url if needed)."""

        user_prompt = f"""Extract product information from this webpage.

Search query: {query}
Supplier: {supplier_name}
Base URL: {base_url}

HTML Content:
{html_content}

Return a JSON array of products found. Example format:
[
  {{
    "name": "Product name",
    "price": "10.50 EUR",
    "url": "https://example.com/product",
    "description": "Product description",
    "in_stock": true
  }}
]

If no products found, return empty array: []
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"} if "gpt-4" in self.model else None
            )
            
            content = response.choices[0].message.content
            logger.debug("GPT response: %s", content)
            
            # Parse JSON response
            import json
            data = json.loads(content)
            
            # Handle both array and object with products key
            if isinstance(data, list):
                products_data = data
            elif isinstance(data, dict) and "products" in data:
                products_data = data["products"]
            elif isinstance(data, dict) and len(data) > 0:
                # Try to find array in the dict
                for value in data.values():
                    if isinstance(value, list):
                        products_data = value
                        break
                else:
                    products_data = []
            else:
                products_data = []
            
            # Convert to ProductInfo objects
            products = []
            for item in products_data:
                try:
                    # Ensure URL is complete
                    url = item.get("url", "")
                    if url and not url.startswith("http"):
                        url = base_url.rstrip("/") + "/" + url.lstrip("/")
                    
                    product = ProductInfo(
                        name=item.get("name", "Unknown"),
                        price=item.get("price", "Consultar no site"),
                        url=url or base_url,
                        supplier=supplier_name,
                        description=item.get("description"),
                        in_stock=item.get("in_stock", True)
                    )
                    products.append(product)
                except Exception as exc:
                    logger.warning("Failed to parse product: %s", exc)
                    continue
            
            logger.info("GPT extracted %d products from %s", len(products), supplier_name)
            return products
            
        except Exception as exc:
            logger.error("GPT extraction failed: %s", exc)
            return []


class MultiSiteGPTScraper:
    """Scraper that uses GPT to search multiple sites."""
    
    SITES = {
        "Leroy Merlin": "https://www.leroymerlin.pt",
        "AKI": "https://www.aki.pt",
        "Bricomarche": "https://www.bricomarche.pt",
        "Maxmat": "https://www.maxmat.pt",
    }
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        delay_between_sites: float = 3.0,
        openai_client: Optional[Any] = None,
        model: str = "gpt-4o-mini",
    ):
        self.scraper = GPTScraper(
            openai_api_key=openai_api_key,
            openai_client=openai_client,
            model=model,
        )
        self.delay_between_sites = delay_between_sites
    
    def search_all(self, query: str) -> List[ProductInfo]:
        """Search all sites for products."""
        all_products = []
        
        for i, (supplier_name, base_url) in enumerate(self.SITES.items()):
            logger.info("Searching %s with GPT for '%s'...", supplier_name, query)
            
            try:
                products = self.scraper.search_site(base_url, query, supplier_name)
                all_products.extend(products)
                
                # Rate limiting
                if i < len(self.SITES) - 1 and products:
                    logger.debug("Waiting %.1fs before next site...", self.delay_between_sites)
                    time.sleep(self.delay_between_sites)
                    
            except Exception as exc:
                logger.warning("Failed to scrape %s: %s", supplier_name, exc)
                continue
        
        logger.info("Found total %d products across all sites", len(all_products))
        return all_products


# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        exit(1)
    
    scraper = MultiSiteGPTScraper(openai_api_key=api_key)
    products = scraper.search_all("cimento")
    
    print(f"\nFound {len(products)} products:")
    for product in products:
        print(f"\n{product.supplier}: {product.name}")
        print(f"  Price: {product.price}")
        print(f"  URL: {product.url}")
        if product.description:
            print(f"  Description: {product.description[:100]}...")
