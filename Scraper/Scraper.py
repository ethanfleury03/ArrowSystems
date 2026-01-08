import json
import logging
import os
import re
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
        username: Optional[str] = "jung.gilee@memjet.partners",
        password: Optional[str] = "INK28Dm8",
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

    def proof_of_concept(self) -> None:
        "Make a funciton that will make a count of all the tickets to prove its working, then extract one ticket."
        pass
    
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
        
        Supports both JSON API endpoints and HTML pages with best-effort extraction.
        """
        page = 1
        consecutive_empty_pages = 0
        max_empty_pages = 3
        
        # Try multiple URL patterns in order
        # Zendesk Help Center uses /hc/en-us/requests with pagination
        url_patterns = [
            lambda p: f"{self.base_url}/hc/en-us/requests?page={p}",  # Zendesk Help Center requests
            lambda p: f"{self.base_url}/hc/requests?page={p}",  # Alternative format
            lambda p: f"{self.base_url}/api/tickets?page={p}",  # API endpoint (if available)
            lambda p: f"{self.base_url}/tickets?page={p}",  # Standard tickets
            lambda p: f"{self.base_url}/tickets/list?page={p}",  # Alternative tickets format
            lambda p: f"{self.base_url}/tickets?pageNumber={p}",  # Another alternative
        ]
        
        # Track which URL pattern works (try first page with all patterns)
        working_pattern = None
        if page == 1:
            for pattern_func in url_patterns:
                test_url = pattern_func(1)
                try:
                    test_response = self._make_request(test_url)
                    # Check if we get a valid response (not 404)
                    if test_response.status_code == 200:
                        working_pattern = pattern_func
                        self.logger.info(f"Using URL pattern: {test_url}")
                        break
                except requests.RequestException:
                    continue
            
            # Fallback to first pattern if none worked
            if working_pattern is None:
                working_pattern = url_patterns[0]
                self.logger.warning(f"No URL pattern worked, defaulting to: {working_pattern(1)}")
        
        while True:
            self.logger.info(f"Fetching ticket list page {page}...")
            
            # Build list page URL using working pattern
            if working_pattern:
                list_url = working_pattern(page)
            else:
                list_url = f"{self.base_url}/tickets?page={page}"
            
            try:
                response = self._make_request(list_url)
                
                ticket_refs = []
                content_type = response.headers.get('Content-Type', '').lower()
                is_json_response = 'application/json' in content_type
                
                # Try JSON first
                if is_json_response:
                    try:
                        data = response.json()
                        
                        # Look for tickets under common keys
                        tickets_list = None
                        for key in ["tickets", "results", "data", "items"]:
                            if key in data and isinstance(data[key], list):
                                tickets_list = data[key]
                                break
                        
                        if tickets_list:
                            for ticket_obj in tickets_list:
                                if not isinstance(ticket_obj, dict):
                                    continue
                                
                                # Extract ticket_id from common keys
                                ticket_id = None
                                for id_key in ["id", "ticket_id", "number", "ticket_number"]:
                                    if id_key in ticket_obj:
                                        ticket_id = str(ticket_obj[id_key])
                                        break
                                
                                if not ticket_id:
                                    continue
                                
                                # Extract URL from common keys
                                ticket_url = None
                                for url_key in ["url", "href", "link", "permalink"]:
                                    if url_key in ticket_obj:
                                        ticket_url = str(ticket_obj[url_key])
                                        break
                                
                                # Build URL if not present
                                if not ticket_url:
                                    ticket_url = f"{self.base_url}/tickets/{ticket_id}"
                                
                                ticket_refs.append({
                                    "ticket_id": ticket_id,
                                    "url": ticket_url
                                })
                    except (ValueError, KeyError, TypeError) as e:
                        self.logger.debug(f"JSON parsing failed, trying HTML: {e}")
                        is_json_response = False
                
                # Try HTML if JSON didn't work
                if not is_json_response or not ticket_refs:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Try multiple selectors for ticket links
                    selectors = [
                        "a[href*='ticket']",
                        "a[href*='/tickets/']",
                        "tr a[href]",
                        "a.ticket",
                        "a.ticket-link",
                        "a[data-ticket-id]",
                        ".ticket-row a",
                        ".ticket-item a",
                    ]
                    
                    links_found = set()
                    for selector in selectors:
                        for link in soup.select(selector):
                            href = link.get("href")
                            if not href:
                                continue
                            
                            # Extract ticket_id
                            ticket_id = None
                            
                            # Try data attribute first
                            ticket_id = link.get("data-ticket-id") or link.get("data-id")
                            
                            # Try regex from href
                            if not ticket_id:
                                import re
                                patterns = [
                                    r"/tickets/(\d+)",
                                    r"ticketId=(\d+)",
                                    r"id=(\d+)",
                                    r"ticket/(\d+)",
                                    r"tickets/([^/]+)",
                                ]
                                for pattern in patterns:
                                    match = re.search(pattern, href, re.IGNORECASE)
                                    if match:
                                        ticket_id = match.group(1)
                                        break
                            
                            if not ticket_id:
                                # Last resort: use href as-is if it looks like a ticket URL
                                if "ticket" in href.lower():
                                    ticket_id = href.split("/")[-1].split("?")[0]
                            
                            if ticket_id:
                                # Normalize URL
                                if not href.startswith("http"):
                                    href = urljoin(self.base_url, href)
                                
                                # Deduplicate by ticket_id
                                if ticket_id not in links_found:
                                    links_found.add(ticket_id)
                                    ticket_refs.append({
                                        "ticket_id": str(ticket_id),
                                        "url": href
                                    })
                    
                    # Also try extracting from embedded JSON in script tags
                    if not ticket_refs:
                        for script in soup.find_all("script", type="application/json"):
                            try:
                                embedded_data = json.loads(script.string)
                                # Recursively search for ticket-like structures
                                def find_tickets(obj, path=""):
                                    if isinstance(obj, dict):
                                        for key, val in obj.items():
                                            if "ticket" in key.lower() and isinstance(val, list):
                                                for item in val:
                                                    if isinstance(item, dict):
                                                        tid = item.get("id") or item.get("ticket_id")
                                                        url = item.get("url") or item.get("href")
                                                        if tid:
                                                            ticket_refs.append({
                                                                "ticket_id": str(tid),
                                                                "url": url or f"{self.base_url}/tickets/{tid}"
                                                            })
                                            find_tickets(val, f"{path}.{key}")
                                    elif isinstance(obj, list):
                                        for item in obj:
                                            find_tickets(item, path)
                                
                                find_tickets(embedded_data)
                            except (json.JSONDecodeError, AttributeError):
                                continue
                
                # Debug logging
                self.logger.info(f"Page {page} ({list_url}): extracted {len(ticket_refs)} ticket refs")
                
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
        
        metadata = {
            "source_url": raw_ticket["source_url"],
            "fetched_at": raw_ticket["fetched_at"],
            "format": format_type,
        }
        
        if format_type == "json":
            # Already JSON - extract messages if possible
            messages = []
            
            # Look for message list under common keys
            message_list = None
            for key in ["messages", "conversation", "thread", "comments", "events", "replies"]:
                if key in raw_data and isinstance(raw_data[key], list):
                    message_list = raw_data[key]
                    break
            
            if message_list:
                for idx, msg_obj in enumerate(message_list):
                    if not isinstance(msg_obj, dict):
                        continue
                    
                    # Extract id
                    msg_id = None
                    for id_key in ["id", "message_id", "comment_id", "event_id"]:
                        if id_key in msg_obj:
                            msg_id = str(msg_obj[id_key])
                            break
                    if not msg_id:
                        msg_id = str(idx)
                    
                    # Extract role
                    role = "unknown"
                    for role_key in ["role", "author_role", "authorType", "author", "sender", "user_type"]:
                        if role_key in msg_obj:
                            role_val = msg_obj[role_key]
                            if isinstance(role_val, str):
                                role = role_val.lower()
                            else:
                                role = str(role_val).lower()
                            break
                    
                    # Extract text
                    text = ""
                    for text_key in ["text", "body", "content", "comment", "message", "description"]:
                        if text_key in msg_obj:
                            text_val = msg_obj[text_key]
                            if isinstance(text_val, str):
                                text = text_val.strip()
                            elif isinstance(text_val, dict):
                                # Sometimes text is nested in a content object
                                text = str(text_val).strip()
                            break
                    
                    # Filter out empty messages
                    if text and len(text) >= 3:
                        messages.append({
                            "id": msg_id,
                            "role": role,
                            "text": text
                        })
            
            self.logger.info(f"Parsed ticket {ticket_id}: extracted {len(messages)} messages from JSON")
            
            return {
                "ticket_id": ticket_id,
                "messages": messages,
                "metadata": metadata,
                "raw_format": "json"
            }
        
        else:  # HTML
            soup = BeautifulSoup(raw_data, 'html.parser')
            messages = []
            
            # Try multiple selectors for message containers
            selectors = [
                ".message",
                ".comment",
                ".reply",
                ".ticket-comment",
                ".conversation-item",
                "article.message",
                "li.message",
                ".chat-message",
                ".thread-item",
                "[data-message-id]",
                "[data-comment-id]",
            ]
            
            found_containers = []
            for selector in selectors:
                containers = soup.select(selector)
                if containers:
                    found_containers.extend(containers)
                    break  # Use first selector that finds results
            
            # If no specific selectors worked, try broader patterns
            if not found_containers:
                # Look for common message-like structures
                found_containers = soup.find_all(["div", "article", "li"], class_=re.compile(r"message|comment|reply", re.I))
            
            for idx, container in enumerate(found_containers):
                # Extract text
                text = container.get_text(" ", strip=True)
                
                # Filter out tiny/empty blocks
                if not text or len(text) < 3:
                    continue
                
                # Extract role from data attributes or class names
                role = "unknown"
                role_attrs = ["data-author", "data-role", "data-author-type", "data-user-type"]
                for attr in role_attrs:
                    role_val = container.get(attr)
                    if role_val:
                        role = str(role_val).lower()
                        break
                
                # Check class names for role hints
                if role == "unknown":
                    classes = container.get("class", [])
                    class_str = " ".join(classes).lower()
                    if "agent" in class_str or "staff" in class_str or "admin" in class_str:
                        role = "agent"
                    elif "customer" in class_str or "user" in class_str or "client" in class_str:
                        role = "user"
                
                # Extract id
                msg_id = container.get("data-id") or container.get("data-message-id") or container.get("data-comment-id") or str(idx)
                
                messages.append({
                    "id": str(msg_id),
                    "role": role,
                    "text": text
                })
            
            # Safety fallback: if we found too many messages (likely wrong selector),
            # keep only the longest ones
            if len(messages) > 50:
                self.logger.warning(f"Found {len(messages)} messages, keeping top 50 longest")
                messages.sort(key=lambda m: len(m["text"]), reverse=True)
                messages = messages[:50]
            
            self.logger.info(f"Parsed ticket {ticket_id}: extracted {len(messages)} messages from HTML")
            
            if messages:
                return {
                    "ticket_id": ticket_id,
                    "messages": messages,
                    "metadata": metadata,
                    "raw_format": "html_extracted"
                }
            else:
                # Fallback: store HTML
                self.logger.warning(f"Could not extract messages from HTML for ticket {ticket_id}, storing raw HTML")
                return {
                    "ticket_id": ticket_id,
                    "messages": [],
                    "html": raw_data,
                    "metadata": metadata,
                    "raw_format": "html"
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
        """
        # Try count endpoint first
        count_endpoints = [
            f"{self.base_url}/api/tickets/count",
            f"{self.base_url}/tickets/count",
            f"{self.base_url}/api/tickets?count=true",
        ]
        
        for endpoint in count_endpoints:
            try:
                response = self._make_request(endpoint)
                if response.headers.get('Content-Type', '').startswith('application/json'):
                    data = response.json()
                    for key in ["total", "count", "totalCount", "total_items", "total_count"]:
                        if key in data:
                            count = data[key]
                            if isinstance(count, int):
                                return count
            except requests.RequestException:
                continue
        
        # Try to get count from first page metadata
        try:
            url_patterns = [
                f"{self.base_url}/api/tickets?page=1",
                f"{self.base_url}/tickets?page=1",
            ]
            
            for url in url_patterns:
                try:
                    response = self._make_request(url)
                    if response.headers.get('Content-Type', '').startswith('application/json'):
                        data = response.json()
                        # Look for total in pagination metadata
                        for key in ["total", "count", "totalCount", "total_items", "total_count"]:
                            if key in data:
                                count = data[key]
                                if isinstance(count, int):
                                    return count
                        # Check pagination object
                        if "pagination" in data:
                            pagination = data["pagination"]
                            for key in ["total", "total_count", "totalItems"]:
                                if key in pagination:
                                    count = pagination[key]
                                    if isinstance(count, int):
                                        return count
                except requests.RequestException:
                    continue
        except Exception:
            pass
        
        return -1  # Unknown
    
    def proof_of_concept(self, expected_count: Optional[int] = None) -> Dict[str, Any]:
        """
        Prove the scraper works by:
          1) Counting how many ticket refs we can discover (should match expected_count if provided)
          2) Fetching + saving exactly one ticket (the first one not already saved)
        
        Args:
            expected_count: Optional expected number of tickets to validate against
        
        Returns:
            Dict with summary info including:
                - discovered_ticket_refs: int
                - expected_count: Optional[int]
                - count_matches: Optional[bool]
                - extracted_ticket_id: Optional[str]
                - saved_path: Optional[str]
        """
        existing_ids = self.load_existing_ticket_ids()
        
        discovered = 0
        first_new_ref: Optional[Dict[str, Any]] = None
        
        for ref in self.iter_ticket_refs():
            discovered += 1
            tid = ref.get("ticket_id")
            if first_new_ref is None and tid and tid not in existing_ids:
                first_new_ref = ref
        
        if expected_count is not None:
            if discovered != expected_count:
                self.logger.warning(
                    f"Ticket count mismatch: discovered={discovered} expected={expected_count}"
                )
            else:
                self.logger.info(f"Ticket count matches expected_count={expected_count}")
        else:
            self.logger.info(f"Discovered ticket refs: {discovered}")
        
        result: Dict[str, Any] = {
            "discovered_ticket_refs": discovered,
            "expected_count": expected_count,
            "count_matches": (discovered == expected_count) if expected_count is not None else None,
            "extracted_ticket_id": None,
            "saved_path": None,
        }
        
        if not first_new_ref:
            self.logger.warning(
                "No new ticket refs found to extract (maybe all are already saved)."
            )
            return result
        
        # Fetch one ticket
        raw_ticket = self.fetch_ticket(first_new_ref)
        
        # Convert to JSON payload you want to persist (lossless if needed)
        payload = self.parse_ticket_to_json(raw_ticket)
        
        ticket_id = raw_ticket.get("ticket_id") or first_new_ref.get("ticket_id") or "unknown_ticket"
        saved_path = self.save_ticket(ticket_id, payload)
        
        self.logger.info(f"Extracted 1 ticket: ticket_id={ticket_id} saved_path={saved_path}")
        
        # Print ticket summary
        self._print_ticket_summary(payload)
        
        result["extracted_ticket_id"] = ticket_id
        result["saved_path"] = saved_path
        result["ticket_summary"] = payload  # Include full ticket data in result
        return result
    
    def _print_ticket_summary(self, ticket_data: Dict[str, Any]) -> None:
        """
        Print a formatted summary of the ticket data.
        
        Args:
            ticket_data: Parsed ticket JSON dict
        """
        print("\n" + "=" * 70)
        print("TICKET SUMMARY")
        print("=" * 70)
        
        ticket_id = ticket_data.get("ticket_id", "unknown")
        print(f"Ticket ID: {ticket_id}")
        
        # Metadata
        metadata = ticket_data.get("metadata", {})
        if metadata:
            print(f"Source URL: {metadata.get('source_url', 'N/A')}")
            print(f"Fetched At: {metadata.get('fetched_at', 'N/A')}")
            print(f"Format: {metadata.get('format', ticket_data.get('raw_format', 'N/A'))}")
        
        # Messages
        messages = ticket_data.get("messages", [])
        message_count = len(messages)
        print(f"\nMessage Count: {message_count}")
        
        if messages:
            print("\nMessages Preview:")
            print("-" * 70)
            # Show first 3 messages
            for idx, msg in enumerate(messages[:3], 1):
                role = msg.get("role", "unknown")
                text = msg.get("text", "")
                # Truncate long messages
                preview = text[:200] + "..." if len(text) > 200 else text
                print(f"\n[{idx}] Role: {role}")
                print(f"    Text: {preview}")
            
            if message_count > 3:
                print(f"\n... and {message_count - 3} more message(s)")
        else:
            print("\nNo messages extracted.")
            # Check if HTML fallback was used
            if "html" in ticket_data:
                html_preview = ticket_data["html"][:200] if isinstance(ticket_data["html"], str) else str(ticket_data["html"])[:200]
                print(f"HTML Content (preview): {html_preview}...")
        
        # Other fields
        other_fields = {k: v for k, v in ticket_data.items() 
                       if k not in ["ticket_id", "messages", "metadata", "html", "raw_format"]}
        if other_fields:
            print("\nOther Fields:")
            for key, value in other_fields.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    print(f"  {key}: {value}")
                elif isinstance(value, (list, dict)):
                    print(f"  {key}: {type(value).__name__} ({len(value) if isinstance(value, list) else 'dict'})")
        
        print("=" * 70 + "\n")


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
