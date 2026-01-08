import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Set, Union
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


class Scraper:
    """
    Robust scraper skeleton for extracting tickets from a website.
    
    Designed for building a ticket-derived cache layer for RAG systems.
    Supports both HTML scraping (BeautifulSoup) and JSON API endpoints.
    
    This is a skeleton class - site-specific selectors and endpoints
    must be implemented in the marked TODO sections.
    """

    def __init__(
        self,
        base_url: str,
        output_dir: str = "data",
        cookies: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        login_callback: Optional[Callable[[requests.Session], bool]] = None,
        page_size: int = 50,
        rate_limit: float = 1.0,
        timeout: int = 30,
        max_retries: int = 3,
        retry_backoff_base: float = 2.0,
    ):
        """
        Initialize the scraper with configuration.
        
        No network calls are made during initialization.
        
        Args:
            base_url: Base URL of the website to scrape
            output_dir: Directory to save scraped tickets
            cookies: Optional dict of cookies for authentication
            headers: Optional dict of custom headers
            username: Optional username for authentication (placeholder)
            password: Optional password for authentication (placeholder)
            login_callback: Optional function(session) -> bool to perform custom login
            page_size: Number of tickets per page (for pagination)
            rate_limit: Seconds to sleep between requests
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            retry_backoff_base: Base multiplier for exponential backoff
        """
        self.base_url = base_url.rstrip('/')
        self.output_dir = Path(output_dir)
        self.page_size = page_size
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff_base = retry_backoff_base
        
        # Setup output directories
        self.tickets_dir = self.output_dir / "tickets_raw"
        self.logs_dir = self.output_dir / "logs"
        self.tickets_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize session
        self.session = requests.Session()
        self._setup_session(cookies, headers, username, password, login_callback)
        
        self.logger.info(f"Scraper initialized for {self.base_url}")
        self.logger.info(f"Output directory: {self.output_dir.absolute()}")
    
    def _setup_logging(self) -> None:
        """Setup logging to both file and console."""
        log_file = self.logs_dir / "scraper.log"
        
        # Create logger
        self.logger = logging.getLogger("Scraper")
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _setup_session(
        self,
        cookies: Optional[Dict[str, str]],
        headers: Optional[Dict[str, str]],
        username: Optional[str],
        password: Optional[str],
        login_callback: Optional[Callable[[requests.Session], bool]],
    ) -> None:
        """Setup requests session with authentication and default headers."""
        # Default headers
        default_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/json,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate',
        }
        
        if headers:
            default_headers.update(headers)
        
        self.session.headers.update(default_headers)
        
        # Set cookies if provided
        if cookies:
            self.session.cookies.update(cookies)
        
        # Perform login if callback provided
        if login_callback:
            self.logger.info("Attempting login via callback...")
            if login_callback(self.session):
                self.logger.info("Login successful")
            else:
                self.logger.warning("Login callback returned False")
        elif username and password:
            # TODO: Implement username/password login if needed
            self.logger.warning("Username/password login not implemented - use login_callback")
    
    def _make_request(
        self,
        url: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> requests.Response:
        """
        Make an HTTP request with retry logic and exponential backoff.
        
        Args:
            url: URL to request
            method: HTTP method (GET, POST, etc.)
            params: Query parameters
            json_data: JSON payload for POST requests
            **kwargs: Additional arguments to pass to requests
            
        Returns:
            Response object
            
        Raises:
            requests.RequestException: If all retries fail
        """
        full_url = urljoin(self.base_url, url) if not url.startswith('http') else url
        
        for attempt in range(self.max_retries):
            try:
                if method.upper() == "GET":
                    response = self.session.get(
                        full_url,
                        params=params,
                        timeout=self.timeout,
                        **kwargs
                    )
                elif method.upper() == "POST":
                    response = self.session.post(
                        full_url,
                        params=params,
                        json=json_data,
                        timeout=self.timeout,
                        **kwargs
                    )
                else:
                    response = self.session.request(
                        method,
                        full_url,
                        params=params,
                        json=json_data,
                        timeout=self.timeout,
                        **kwargs
                    )
                
                # Check for rate limiting or server errors
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', self.rate_limit * (2 ** attempt)))
                    self.logger.warning(f"Rate limited. Waiting {retry_after}s before retry {attempt + 1}/{self.max_retries}")
                    time.sleep(retry_after)
                    continue
                
                if 500 <= response.status_code < 600:
                    if attempt < self.max_retries - 1:
                        backoff = self.retry_backoff_base ** attempt
                        self.logger.warning(
                            f"Server error {response.status_code}. "
                            f"Retrying in {backoff}s (attempt {attempt + 1}/{self.max_retries})"
                        )
                        time.sleep(backoff)
                        continue
                
                # Success or non-retryable error
                response.raise_for_status()
                return response
                
            except requests.Timeout:
                if attempt < self.max_retries - 1:
                    backoff = self.retry_backoff_base ** attempt
                    self.logger.warning(f"Request timeout. Retrying in {backoff}s (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(backoff)
                    continue
                raise
            
            except requests.RequestException as e:
                if attempt < self.max_retries - 1:
                    backoff = self.retry_backoff_base ** attempt
                    self.logger.warning(f"Request failed: {e}. Retrying in {backoff}s (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(backoff)
                    continue
                raise
        
        raise requests.RequestException(f"Failed after {self.max_retries} attempts")
    
    def run(self, max_tickets: Optional[int] = None) -> list[str]:
        """
        Orchestrate the scraping process.
        
        Args:
            max_tickets: Maximum number of tickets to scrape (None = all)
            
        Returns:
            List of scraped ticket IDs
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting ticket scraping")
        self.logger.info(f"Base URL: {self.base_url}")
        self.logger.info(f"Max tickets: {max_tickets or 'unlimited'}")
        self.logger.info("=" * 60)
        
        scraped_ids = []
        existing_ids = self.load_existing_ticket_ids()
        self.logger.info(f"Found {len(existing_ids)} existing tickets to skip")
        
        try:
            for ticket_ref in self.iter_ticket_refs():
                if max_tickets and len(scraped_ids) >= max_tickets:
                    self.logger.info(f"Reached max_tickets limit ({max_tickets})")
                    break
                
                ticket_id = ticket_ref.get("ticket_id")
                if not ticket_id:
                    self.logger.warning(f"Skipping ticket_ref without ticket_id: {ticket_ref}")
                    continue
                
                if ticket_id in existing_ids:
                    self.logger.debug(f"Skipping existing ticket: {ticket_id}")
                    continue
                
                try:
                    # Fetch ticket
                    raw_ticket = self.fetch_ticket(ticket_ref)
                    
                    # Parse to JSON structure
                    ticket_json = self.parse_ticket_to_json(raw_ticket)
                    
                    # Save ticket
                    saved_path = self.save_ticket(ticket_id, ticket_json)
                    scraped_ids.append(ticket_id)
                    
                    self.logger.info(f"Saved ticket {ticket_id} -> {saved_path}")
                    
                    # Rate limiting
                    if self.rate_limit > 0:
                        time.sleep(self.rate_limit)
                
                except Exception as e:
                    self.logger.error(f"Error processing ticket {ticket_id}: {e}", exc_info=True)
                    continue
        
        except Exception as e:
            self.logger.error(f"Fatal error during scraping: {e}", exc_info=True)
            raise
        
        self.logger.info("=" * 60)
        self.logger.info(f"Scraping complete. Saved {len(scraped_ids)} tickets")
        self.logger.info("=" * 60)
        
        return scraped_ids
    
    def iter_ticket_refs(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate over ticket references from list pages.
        
        Yields dictionaries with at least:
            - "ticket_id": str
            - "url": str
            - (optional metadata)
        
        TODO: Implement site-specific logic to:
            - Determine list page URL pattern
            - Extract ticket references from list pages
            - Handle pagination (page=1, page=2, ...)
            - Stop when no more results
        """
        page = 1
        consecutive_empty_pages = 0
        max_empty_pages = 3
        
        while True:
            self.logger.info(f"Fetching ticket list page {page}...")
            
            # TODO: Build the list page URL
            # Example: list_url = f"{self.base_url}/tickets?page={page}"
            list_url = f"{self.base_url}/tickets?page={page}"  # PLACEHOLDER - REPLACE
            
            try:
                response = self._make_request(list_url)
                
                # TODO: Determine if response is JSON or HTML
                # If JSON: parse JSON and extract ticket refs
                # If HTML: use BeautifulSoup to extract ticket refs
                
                # PLACEHOLDER: Try JSON first, fallback to HTML
                try:
                    data = response.json()
                    # TODO: Extract ticket refs from JSON structure
                    # Example: ticket_refs = [{"ticket_id": t["id"], "url": t["url"]} for t in data.get("tickets", [])]
                    ticket_refs = []  # PLACEHOLDER
                except ValueError:
                    # Not JSON, try HTML
                    soup = BeautifulSoup(response.text, 'html.parser')
                    # TODO: Extract ticket refs using BeautifulSoup selectors
                    # Example: ticket_refs = [{"ticket_id": el.get("data-id"), "url": el.get("href")} for el in soup.select(".ticket-link")]
                    ticket_refs = []  # PLACEHOLDER
                
                if not ticket_refs:
                    consecutive_empty_pages += 1
                    if consecutive_empty_pages >= max_empty_pages:
                        self.logger.info(f"No more tickets found after {max_empty_pages} empty pages")
                        break
                else:
                    consecutive_empty_pages = 0
                    for ref in ticket_refs:
                        # Normalize URLs
                        if "url" in ref and not ref["url"].startswith("http"):
                            ref["url"] = urljoin(self.base_url, ref["url"])
                        yield ref
                
                page += 1
                
                # Safety limit
                if page > 1000:
                    self.logger.warning("Reached page limit (1000), stopping")
                    break
            
            except requests.RequestException as e:
                self.logger.error(f"Error fetching page {page}: {e}")
                break
    
    def fetch_ticket(self, ref: Dict[str, Any]) -> Dict[str, Any]:
        """
        Download a single ticket and return raw data.
        
        Args:
            ref: Ticket reference dict with at least "ticket_id" and "url"
            
        Returns:
            Dict with:
                - "ticket_id": str
                - "fetched_at": ISO timestamp
                - "source_url": str
                - "raw": str (HTML) or dict (JSON)
                - "format": "json" or "html"
        """
        ticket_id = ref.get("ticket_id")
        ticket_url = ref.get("url", f"{self.base_url}/tickets/{ticket_id}")
        
        self.logger.debug(f"Fetching ticket {ticket_id} from {ticket_url}")
        
        response = self._make_request(ticket_url)
        
        # Determine format
        content_type = response.headers.get('Content-Type', '').lower()
        is_json = 'application/json' in content_type
        
        if is_json:
            try:
                raw_data = response.json()
                format_type = "json"
            except ValueError:
                raw_data = response.text
                format_type = "html"
        else:
            raw_data = response.text
            format_type = "html"
        
        from datetime import datetime
        return {
            "ticket_id": ticket_id,
            "fetched_at": datetime.utcnow().isoformat(),
            "source_url": ticket_url,
            "raw": raw_data,
            "format": format_type,
        }
    
    def parse_ticket_to_json(self, raw_ticket: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse raw ticket data (HTML or JSON) into a normalized JSON structure.
        
        If raw is HTML, attempts to extract a minimal JSON structure.
        If not possible, wraps HTML in a JSON structure.
        
        Args:
            raw_ticket: Raw ticket dict from fetch_ticket()
            
        Returns:
            Normalized JSON dict with at least:
                - "ticket_id": str
                - "messages": list (or "html": str if extraction fails)
                - "metadata": dict
        """
        ticket_id = raw_ticket["ticket_id"]
        raw_data = raw_ticket["raw"]
        format_type = raw_ticket["format"]
        
        if format_type == "json":
            # Already JSON - extract messages if possible
            # TODO: Map site-specific JSON structure to normalized format
            # Example: return {"ticket_id": ticket_id, "messages": raw_data.get("conversation", []), "metadata": {...}}
            return {
                "ticket_id": ticket_id,
                "messages": raw_data.get("messages", raw_data.get("conversation", [])),
                "metadata": {
                    "source_url": raw_ticket["source_url"],
                    "fetched_at": raw_ticket["fetched_at"],
                    "format": "json",
                }
            }
        
        else:  # HTML
            # TODO: Extract messages from HTML using BeautifulSoup
            # Example:
            #   soup = BeautifulSoup(raw_data, 'html.parser')
            #   messages = []
            #   for msg_el in soup.select(".message"):
            #       messages.append({
            #           "body": msg_el.get_text(),
            #           "author": msg_el.get("data-author"),
            #           "timestamp": msg_el.get("data-timestamp"),
            #       })
            
            # For now, return HTML wrapped in JSON
            soup = BeautifulSoup(raw_data, 'html.parser')
            messages = []  # PLACEHOLDER - implement extraction
            
            if messages:
                return {
                    "ticket_id": ticket_id,
                    "messages": messages,
                    "metadata": {
                        "source_url": raw_ticket["source_url"],
                        "fetched_at": raw_ticket["fetched_at"],
                        "format": "html_extracted",
                    }
                }
            else:
                # Fallback: store HTML
                return {
                    "ticket_id": ticket_id,
                    "html": raw_data,
                    "metadata": {
                        "source_url": raw_ticket["source_url"],
                        "fetched_at": raw_ticket["fetched_at"],
                        "format": "html",
                    }
                }
    
    def save_ticket(self, ticket_id: str, payload: Dict[str, Any]) -> str:
        """
        Save ticket data to disk.
        
        Args:
            ticket_id: Unique ticket identifier
            payload: Ticket data dict to save
            
        Returns:
            Path to saved file
        """
        # Sanitize ticket_id for filesystem
        safe_id = "".join(c for c in ticket_id if c.isalnum() or c in ('-', '_', '.'))
        file_path = self.tickets_dir / f"{safe_id}.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        
        return str(file_path)
    
    def load_existing_ticket_ids(self) -> Set[str]:
        """
        Load set of ticket IDs that have already been scraped.
        
        Returns:
            Set of existing ticket IDs
        """
        existing = set()
        
        if not self.tickets_dir.exists():
            return existing
        
        for file_path in self.tickets_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    ticket_id = data.get("ticket_id")
                    if ticket_id:
                        existing.add(ticket_id)
            except (json.JSONDecodeError, KeyError) as e:
                self.logger.warning(f"Error reading {file_path}: {e}")
                continue
        
        return existing
    
    def get_ticket_count(self) -> int:
        """
        Get total number of tickets available (best effort).
        
        Returns:
            Number of tickets, or -1 if unknown
            
        TODO: Implement site-specific logic to determine total count
        """
        # TODO: Fetch first page or a count endpoint to determine total
        # Example:
        #   response = self._make_request(f"{self.base_url}/tickets/count")
        #   return response.json().get("total", -1)
        
        return -1  # Unknown


if __name__ == "__main__":
    # Example usage
    scraper = Scraper(
        base_url="https://example.com/tickets",
        output_dir="data",
        cookies={"session_id": "your_session_cookie"},
        headers={"X-Custom-Header": "value"},
        rate_limit=1.0,
        timeout=30,
    )
    
    # Run scraper
    scraped_ids = scraper.run(max_tickets=161)
    print(f"Scraped {len(scraped_ids)} tickets")
