"""Advanced AI-assisted material price agent with LangChain and web search integration.

This module extends the basic agent with:
1. LangChain integration for more sophisticated agent workflows
2. Web search capabilities (DuckDuckGo, Bing API support)
3. Better error handling and retry logic
4. Results caching
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_base_url(base_url: Optional[str]) -> Optional[str]:
    """Ensure custom base URLs end with /v1 as required by OpenAI-compatible APIs."""
    if not base_url:
        return None
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        return None
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class MaterialSearchResult(BaseModel):
    """Structured result from material price search."""
    
    material_name: str
    pt_name: str
    suppliers: List[Dict[str, Any]]
    best_supplier: str
    best_price: str
    best_url: str
    reasoning: str
    search_timestamp: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Advanced Agent with LangChain
# ---------------------------------------------------------------------------


@dataclass
class AdvancedMaterialAgent:
    """Enhanced agent using LangChain for orchestration and web search."""
    
    openai_api_key: Optional[str] = None
    openai_client: Optional[Any] = None
    model_name: str = "gpt-5-nano"
    temperature: float = 1.0
    use_bing: bool = False
    bing_api_key: Optional[str] = None
    use_chatgpt_search: bool = False
    local_llm_enabled: bool = False
    local_llm_base_url: Optional[str] = None
    local_llm_model: Optional[str] = None
    local_llm_api_key: str = "lm-studio"

    def __post_init__(self) -> None:
        """Initialize LangChain components."""
        logger.info("Initializing AdvancedMaterialAgent with OpenAI (model %s)", self.model_name)
        
        if self.model_name.startswith("gpt-5-nano") and self.temperature != 1.0:
            logger.warning(
                "gpt-5-nano поддерживает только temperature=1.0. Значение %.2f будет переопределено.",
                self.temperature,
            )
            self.temperature = 1.0

        self._openai_client = self._resolve_openai_client()
        self._using_local_llm = False
        
        # Initialize search tool
        if self.use_chatgpt_search:
            self.search_tool = self._create_chatgpt_search_tool()
        elif self.use_bing and self.bing_api_key:
            self.search_tool = self._create_bing_search_tool()
        else:
            self.search_tool = self._create_duckduckgo_search_tool()
        
        # Create agent tools
        self.tools = [
            self.search_tool,
            Tool(
                name="PriceExtractor",
                func=self._extract_price_info,
                description="Extract structured price information from search results. Input should be raw search results text."
            ),
            Tool(
                name="SupplierComparer",
                func=self._compare_suppliers,
                description="Compare multiple supplier quotes and select the best one. Input should be JSON array of supplier quotes."
            ),
        ]
        
        # Create React agent with preferred LLM, falling back to local if needed
        self._init_agent_with_preference()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _resolve_openai_client(self) -> Any:
        if self.openai_client is not None:
            return self.openai_client

        if self.openai_api_key:
            return OpenAI(api_key=self.openai_api_key)

        if self.local_llm_enabled and self.local_llm_base_url:
            base_url = _normalize_base_url(self.local_llm_base_url)
            return OpenAI(
                api_key=self.local_llm_api_key,
                base_url=base_url,
            )

        raise ValueError("OpenAI credentials or local LLM configuration must be provided.")

    def _init_agent_with_preference(self) -> None:
        """Initialise LangChain agent preferring primary LLM with fallback to local."""
        has_primary = False
        if hasattr(self._openai_client, "has_primary"):
            has_primary = bool(getattr(self._openai_client, "has_primary"))
        elif self.openai_api_key:
            has_primary = True

        if has_primary:
            try:
                self._init_langchain_llm(use_local=False)
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("Primary LLM unavailable for advanced agent: %s", exc)
                self._using_local_llm = False

        if self.local_llm_enabled:
            self._init_langchain_llm(use_local=True)
        else:
            raise RuntimeError("Advanced agent cannot initialize: no usable LLM configured.")

    def _init_langchain_llm(self, *, use_local: bool) -> None:
        """Create ChatOpenAI instance and rebuild the LangChain agent."""
        if use_local:
            base_url = _normalize_base_url(self.local_llm_base_url)
            if not base_url:
                raise ValueError("Local LLM base URL is required for fallback initialization.")
            model_name = self.local_llm_model or self.model_name
            self.llm = ChatOpenAI(
                temperature=self.temperature,
                model=model_name,
                openai_api_key=self.local_llm_api_key,
                openai_api_base=base_url,
            )
            self._using_local_llm = True
        else:
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required to initialize primary LLM.")
            self.llm = ChatOpenAI(
                temperature=self.temperature,
                model=self.model_name,
                openai_api_key=self.openai_api_key,
            )
            self._using_local_llm = False

        self.agent = self._create_agent()

    def _maybe_switch_to_local(self) -> bool:
        """Try to switch LangChain agent to local LLM; return True if successful."""
        if not self.local_llm_enabled or self._using_local_llm:
            return False
        try:
            self._init_langchain_llm(use_local=True)
            logger.info("Advanced agent switched to local LLM at %s", self.local_llm_base_url)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to initialize local fallback LLM: %s", exc)
            return False
    
    def _create_duckduckgo_search_tool(self) -> Tool:
        """Create DuckDuckGo search tool."""
        search = DuckDuckGoSearchRun()
        return Tool(
            name="WebSearch",
            func=lambda query: self._search_portugal_suppliers(query, search),
            description="Search for material prices from Portuguese suppliers. Input should be the material name or search query in Portuguese."
        )
    
    def _create_bing_search_tool(self) -> Tool:
        """Create Bing search tool (if API key is available)."""
        def bing_search(query: str) -> str:
            """Execute Bing search with Portugal filter."""
            import requests
            
            headers = {"Ocp-Apim-Subscription-Key": self.bing_api_key}
            params = {
                "q": f"{query} Portugal preço",
                "mkt": "pt-PT",
                "count": 10,
            }
            
            try:
                response = requests.get(
                    "https://api.bing.microsoft.com/v7.0/search",
                    headers=headers,
                    params=params,
                    timeout=10,
                )
                response.raise_for_status()
                results = response.json()
                
                # Extract relevant snippets
                snippets = []
                for page in results.get("webPages", {}).get("value", [])[:5]:
                    snippets.append(f"{page['name']}: {page['snippet']} (URL: {page['url']})")
                
                return "\n\n".join(snippets) if snippets else "No results found"
            
            except Exception as e:
                logger.error("Bing search failed: %s", e)
                return f"Search error: {e}"
        
        return Tool(
            name="WebSearch",
            func=bing_search,
            description="Search for material prices using Bing. Input should be the material name in Portuguese."
        )

    def _create_chatgpt_search_tool(self) -> Tool:
        """Create ChatGPT Search tool using the OpenAI Responses API."""

        def chatgpt_search(query: str) -> str:
            system_prompt = (
                "You are a procurement analyst helping estimate material costs in Portugal. "
                "Search the web and summarise up-to-date supplier offers with EUR prices and URLs."
            )
            user_prompt = (
                "Find current Portuguese suppliers that sell '{query}' and provide concise bullet points with "
                "supplier name, price, currency, availability notes, and the direct URL."
            )
            try:
                response = self._openai_client.responses.create(
                    model=self.model_name,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt.format(query=query)},
                    ],
                    tools=[{"type": "web_search"}],
                    tool_choice="auto",
                    max_output_tokens=800,
                )
                return self._response_to_text(response)
            except Exception as e:
                logger.error("ChatGPT search failed for '%s': %s", query, e)
                return f"Search error: {e}"

        return Tool(
            name="WebSearch",
            func=chatgpt_search,
            description="Search for material prices using ChatGPT Search with live web results. Input should be the material name in Portuguese."
        )

    def _response_to_text(self, response: Any) -> str:
        """Extract plain text from OpenAI Responses output."""
        text_chunks: List[str] = []

        output = getattr(response, "output", None)
        if output:
            for item in output:
                contents = getattr(item, "content", [])
                for content in contents:
                    if getattr(content, "type", None) == "output_text":
                        text_obj = getattr(content, "text", None)
                        if hasattr(text_obj, "value") and text_obj.value:
                            text_chunks.append(text_obj.value)
                        elif isinstance(text_obj, str):
                            text_chunks.append(text_obj)

        if text_chunks:
            return "\n".join(text_chunks).strip()

        if hasattr(response, "model_dump"):
            data = response.model_dump()
            for item in data.get("output", []):
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        text_value = content.get("text", {}).get("value")
                        if text_value:
                            text_chunks.append(text_value)

        if text_chunks:
            return "\n".join(text_chunks).strip()

        return str(response)
    
    def _search_portugal_suppliers(self, query: str, search_func) -> str:
        """Execute search with Portugal-specific filters."""
        enhanced_query = f"{query} Portugal preço comprar"
        try:
            results = search_func.run(enhanced_query)
            return results
        except Exception as e:
            logger.error("Search failed for query '%s': %s", query, e)
            return f"Search failed: {e}"
    
    def _extract_price_info(self, search_results: str) -> str:
        """Extract structured price information from search results using LLM."""
        prompt = f"""
        Extract price information from these search results about building materials in Portugal:
        
        {search_results}
        
        Return a JSON array with the following structure:
        [
          {{
            "supplier": "supplier name",
            "price": "price with currency",
            "url": "product URL",
            "material": "material description"
          }}
        ]
        
        Only include entries where price information is clearly available.
        """
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error("Price extraction failed: %s", e)
            return "[]"
    
    def _compare_suppliers(self, suppliers_json: str) -> str:
        """Compare suppliers and select best offer."""
        prompt = f"""
        Compare these supplier quotes for building materials and select the best one:
        
        {suppliers_json}
        
        Consider:
        1. Price (lower is better)
        2. Supplier reputation (Leroy Merlin, AKI, Bricomarche are trusted)
        3. Product specifications match
        
        Return JSON:
        {{
          "best_supplier": "supplier name",
          "price": "price",
          "url": "URL",
          "reasoning": "why this is the best choice"
        }}
        """
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error("Supplier comparison failed: %s", e)
            return "{}"
    
    def _create_agent(self) -> AgentExecutor:
        """Create the React agent executor."""
        template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

        prompt = PromptTemplate.from_template(template)
        agent = create_react_agent(self.llm, self.tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
        )
    
    def find_best_price(self, material_name: str) -> MaterialSearchResult:
        """
        Find the best price for a material using the agent workflow.
        
        Args:
            material_name: Name of the material (can be in any language)
        
        Returns:
            MaterialSearchResult with best supplier and price
        """
        logger.info("Starting search for material: %s", material_name)
        
        query = f"""
        Find the best price for the building material "{material_name}" in Portugal.
        
        Steps:
        1. Translate the material name to Portuguese if needed
        2. Search for prices from Portuguese suppliers
        3. Extract price information from search results
        4. Compare suppliers and select the best offer
        5. Return the result with supplier name, price, URL, and reasoning
        """
        
        try:
            return self._invoke_agent(query, material_name)
        except Exception as exc:  # noqa: BLE001
            logger.error("Agent execution failed for '%s': %s", material_name, exc)
            if self._maybe_switch_to_local():
                try:
                    return self._invoke_agent(query, material_name)
                except Exception as local_exc:  # noqa: BLE001
                    logger.error(
                        "Local fallback agent also failed for '%s': %s",
                        material_name,
                        local_exc,
                    )
                    exc = local_exc

            return MaterialSearchResult(
                material_name=material_name,
                pt_name=material_name,
                suppliers=[],
                best_supplier="Error",
                best_price="N/A",
                best_url="N/A",
                reasoning=f"Search failed: {exc}",
            )
    
    def _invoke_agent(self, query: str, material_name: str) -> MaterialSearchResult:
        """Invoke the LangChain agent and parse its response."""
        result = self.agent.invoke({"input": query})
        final_answer = result.get("output", "{}")

        try:
            parsed = json.loads(final_answer)
            return MaterialSearchResult(
                material_name=material_name,
                pt_name=parsed.get("pt_name", material_name),
                suppliers=parsed.get("all_suppliers", []),
                best_supplier=parsed.get("best_supplier", "N/A"),
                best_price=parsed.get("price", "N/A"),
                best_url=parsed.get("url", "N/A"),
                reasoning=parsed.get("reasoning", "No reasoning provided"),
            )
        except json.JSONDecodeError:
            logger.warning("Could not parse agent result as JSON: %s", final_answer)
            return MaterialSearchResult(
                material_name=material_name,
                pt_name=material_name,
                suppliers=[],
                best_supplier="Error",
                best_price="N/A",
                best_url="N/A",
                reasoning=f"Agent result could not be parsed: {final_answer}",
            )
    
    def batch_search(self, materials: List[str]) -> List[MaterialSearchResult]:
        """Search for multiple materials with rate limiting."""
        results = []
        for material in materials:
            result = self.find_best_price(material)
            results.append(result)
            time.sleep(2)  # Rate limiting
        return results
