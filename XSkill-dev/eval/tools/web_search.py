"""
WebSearch tool backed by Bocha Web Search.

The project uses China-accessible services by default. Legacy overseas search
API fallbacks are intentionally disabled for the XSkillRL workflow.
"""

import os
import json
import requests
from tools.base import BaseTool
from tools.tool_registry import register_tool


@register_tool("web_search")
class WebSearch(BaseTool):
    name = "web_search"
    description = "Search the web for information online. Use when you need to find information, facts, or current events. Returns web search results with titles, URLs, and text snippets. You only have limited search times, so please use it wisely."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query string"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 10)",
                "default": 10
            }
        },
        "required": ["query"]
    }
    
    def __init__(self, config=None):
        super().__init__(config)
        config = config or {}
        provider = (
            config.get("provider")
            or os.getenv("WEB_SEARCH_PROVIDER")
            or os.getenv("SEARCH_API_PROVIDER")
            or "bocha"
        )
        self.provider = provider.lower()

        if self.provider != "bocha":
            raise ValueError(
                "Only SEARCH_API_PROVIDER=bocha is supported in the domestic XSkillRL workflow."
            )

        self.api_key = config.get("api_key") or os.getenv("BOCHA_API_KEY")
        self.api_endpoint = (
            config.get("api_endpoint")
            or os.getenv("BOCHA_SEARCH_ENDPOINT")
            or "https://api.bochaai.com/v1/web-search"
        )
        if not self.api_key:
            raise ValueError(
                "Bocha API key not found. Please set BOCHA_API_KEY "
                "or provide api_key in config."
            )

        self.max_results_default = config.get('max_results', 10) if config else 10
        self.timeout = config.get('timeout', 30) if config else 30
        
        # Proxy configuration (read from environment variable)
        self.proxies = None
        http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
        https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
        if http_proxy or https_proxy:
            self.proxies = {
                "http": http_proxy,
                "https": https_proxy or http_proxy
            }
            print(f"[WebSearch] Using proxy: {self.proxies}")
    
    def call(self, params, **kwargs):
        """
        Execute web search
        
        Args:
            params: Dictionary containing query and optional max_results
            **kwargs: Additional parameters (not used)
            
        Returns:
            Formatted search result string
        """
        # Parse parameters
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                # If parsing fails, treat string as query
                params = {"query": params}
        
        query = params.get("query", "")
        max_results = params.get("max_results", self.max_results_default)
        
        if not query:
            return "Error: No search query provided"
        
        # Retry mechanism
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                print(f"[WebSearch:{self.provider}] Searching for: {query} (attempt {attempt + 1}/{max_retries})")

                headers, payload = self._build_request(query, max_results)
                
                # Send request (supports proxy)
                response = requests.post(
                    self.api_endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                    proxies=self.proxies
                )
                
                # Check response status
                if response.status_code != 200:
                    error_msg = f"API request failed with status {response.status_code}: {response.text[:200]}"
                    print(f"[WebSearch] {error_msg}")
                    if attempt < max_retries - 1:
                        print(f"[WebSearch] Retrying in {retry_delay} seconds...")
                        import time
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    return f"Error: {error_msg}"
                
                # Parse response
                try:
                    result_data = response.json()
                except json.JSONDecodeError as e:
                    error_msg = f"Failed to parse JSON response: {str(e)}. Response: {response.text[:200]}"
                    print(f"[WebSearch] {error_msg}")
                    if attempt < max_retries - 1:
                        print(f"[WebSearch] Retrying in {retry_delay} seconds...")
                        import time
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    return f"Error: {error_msg}"
                
                organic_results = self._extract_results(result_data)
                
                if not organic_results:
                    # Check if there is error information
                    error_info = result_data.get("error", "")
                    if error_info:
                        return f"Error: {error_info}"
                    return f"No results found for query: '{query}'"
                
                # Limit result number
                organic_results = organic_results[:max_results]
                
                # Format results
                formatted_results = []
                for i, result in enumerate(organic_results, 1):
                    title = result.get('title', 'No title')
                    link = result.get('link') or result.get('url') or 'No URL'
                    snippet = result.get('snippet') or result.get('summary') or result.get('content') or 'No description'
                    
                    formatted_results.append(
                        f"{i}. [{title}]({link})\n"
                        f"   {snippet}"
                    )
                
                output = f"Search results for '{query}':\n\n" + "\n\n".join(formatted_results)
                print(f"[WebSearch] Found {len(organic_results)} results")
                return output
                
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(f"[WebSearch] Request timed out, retrying in {retry_delay} seconds...")
                    import time
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                error_msg = "Request timed out after all retries"
                print(f"[WebSearch] {error_msg}")
                return f"Error: {error_msg}"
            except requests.exceptions.ConnectionError as e:
                if attempt < max_retries - 1:
                    print(f"[WebSearch] Connection error: {str(e)}, retrying in {retry_delay} seconds...")
                    import time
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                error_msg = f"Connection failed: {str(e)}. Please check network connection and DNS settings."
                print(f"[WebSearch] {error_msg}")
                return f"Error: {error_msg}"
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[WebSearch] Unexpected error: {str(e)}, retrying in {retry_delay} seconds...")
                    import time
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                error_msg = f"Unexpected error during web search: {str(e)}"
                print(f"[WebSearch] {error_msg}")
                import traceback
                traceback.print_exc()
                return f"Error: {error_msg}"
        
        # If all retries fail, return error
        return "Error: All retry attempts failed"

    def _build_request(self, query, max_results):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "query": query,
            "freshness": os.getenv("BOCHA_SEARCH_FRESHNESS", "noLimit"),
            "summary": os.getenv("BOCHA_SEARCH_SUMMARY", "true").lower() not in ("0", "false", "no"),
            "count": min(int(max_results), 50),
        }
        return headers, payload

    def _extract_results(self, result_data):
        data = result_data.get("data", {}) if isinstance(result_data, dict) else {}
        web_pages = data.get("webPages", {}) if isinstance(data, dict) else {}
        value = web_pages.get("value", []) if isinstance(web_pages, dict) else []
        if value:
            return [
                {
                    "title": item.get("name") or item.get("title"),
                    "link": item.get("url"),
                    "snippet": item.get("summary") or item.get("snippet"),
                }
                for item in value
            ]

        # Keep a permissive fallback for compatible Bocha proxy variants.
        for key in ("results", "webPages"):
            candidates = data.get(key) if isinstance(data, dict) else result_data.get(key)
            if isinstance(candidates, list):
                return candidates
        return []
