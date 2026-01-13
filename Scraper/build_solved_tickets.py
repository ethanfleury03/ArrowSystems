#!/usr/bin/env python3
"""
Stage 2: Build detailed ticket conversation JSON only for solved/closed tickets (expensive operation).

This script:
- Reads solved ticket IDs from tickets_index
- Skips tickets already in tickets_detail
- Fetches detailed conversations via API
- Stores normalized conversation JSON in tickets_detail
"""

import html
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("ERROR: Selenium not installed. Install with: pip install selenium")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("WARNING: python-dotenv not installed. Install with: pip install python-dotenv")

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("WARNING: BeautifulSoup4 not installed. Install with: pip install beautifulsoup4")

from ticket_store import get_ticket_store


def setup_logging() -> logging.Logger:
    """Setup structured logging."""
    logger = logging.getLogger("build_solved_tickets")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def load_credentials() -> tuple[str, str]:
    """Load credentials from Scraper/.env file."""
    env_path = Path(__file__).parent / ".env"
    
    if DOTENV_AVAILABLE and env_path.exists():
        load_dotenv(env_path)
    
    email = os.getenv("ZENDESK_EMAIL")
    password = os.getenv("ZENDESK_PASSWORD")
    
    if not email or not password:
        raise ValueError(
            "ZENDESK_EMAIL and ZENDESK_PASSWORD must be set in Scraper/.env file"
        )
    
    return email, password


def get_driver(headless: bool = False) -> Any:
    """Create Selenium Chrome driver."""
    chrome_options = Options()
    
    if headless:
        chrome_options.add_argument('--headless')
    
    chrome_options.add_argument('--window-size=1600,1000')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    logger = logging.getLogger("build_solved_tickets")
    logger.info(f"Creating Chrome driver (headless={headless})")
    
    driver = webdriver.Chrome(options=chrome_options)
    driver.set_window_size(1600, 1000)
    
    return driver


def browser_fetch_json(driver: Any, url: str, max_retries: int = 3) -> Dict[str, Any]:
    """
    Fetch JSON from URL using browser context (preserves cookies/auth) with retry/backoff.
    
    Args:
        driver: Selenium WebDriver instance
        url: URL to fetch
        max_retries: Maximum retry attempts for 429/5xx errors
        
    Returns:
        Dict with: ok, status, content_type, text, data (if JSON)
    """
    logger = logging.getLogger("build_solved_tickets")
    
    for attempt in range(max_retries):
        try:
            result = driver.execute_async_script(r"""
                var url = arguments[0];
                var callback = arguments[arguments.length - 1];
                
                fetch(url, {
                    credentials: "include",
                    headers: {
                        "accept": "application/json"
                    }
                })
                .then(function(response) {
                    return response.text().then(function(text) {
                        callback({
                            ok: response.ok,
                            status: response.status,
                            content_type: response.headers.get("content-type") || "",
                            text: text
                        });
                    });
                })
                .catch(function(error) {
                    callback({
                        ok: false,
                        status: 0,
                        content_type: "",
                        text: error.toString()
                    });
                });
            """, url)
            
            status = result.get("status", 0)
            
            # Retry on 429 (rate limit) or 5xx errors
            if status == 429 or (500 <= status < 600):
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s, 6s
                    logger.warning(f"Status {status} on attempt {attempt + 1}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
            
            # Parse JSON if content type indicates JSON
            data = None
            content_type = result.get("content_type", "").lower()
            if "json" in content_type:
                try:
                    data = json.loads(result["text"])
                except json.JSONDecodeError:
                    pass
            
            return {
                "ok": result.get("ok", False),
                "status": status,
                "content_type": result.get("content_type", ""),
                "text": result.get("text", ""),
                "data": data
            }
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                logger.warning(f"Exception on attempt {attempt + 1}: {e}, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.warning(f"Browser fetch failed for {url}: {e}")
                return {
                    "ok": False,
                    "status": 0,
                    "content_type": "",
                    "text": str(e),
                    "data": None
                }
    
    return {
        "ok": False,
        "status": 0,
        "content_type": "",
        "text": "Max retries exceeded",
        "data": None
    }


def normalize_text(text: str) -> str:
    """Normalize message text: strip HTML entities, normalize whitespace, preserve paragraphs."""
    if not text:
        return ""
    
    # HTML unescape
    text = html.unescape(text)
    
    # Replace &nbsp; and \u00a0 with space
    text = text.replace("&nbsp;", " ")
    text = text.replace("\u00a0", " ")
    
    # Normalize newlines: CRLF -> LF
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    
    # Split into lines for processing
    lines = text.split("\n")
    
    # Strip trailing spaces per line
    lines = [line.rstrip() for line in lines]
    
    # Collapse multiple blank lines (max 2 consecutive)
    normalized_lines = []
    blank_count = 0
    for line in lines:
        if not line.strip():
            blank_count += 1
            if blank_count <= 2:
                normalized_lines.append("")
        else:
            blank_count = 0
            normalized_lines.append(line)
    
    # Join back and strip ends
    text = "\n".join(normalized_lines)
    text = text.strip()
    
    return text


def extract_attachments_from_html(html_content: str) -> List[Dict[str, Any]]:
    """Extract attachment/image URLs from HTML content."""
    attachments = []
    seen_urls = set()
    
    if not html_content or not BS4_AVAILABLE:
        return attachments
    
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find all images
        for img in soup.find_all('img'):
            src = img.get('src', '')
            if src and src not in seen_urls:
                seen_urls.add(src)
                name = img.get('alt') or img.get('name') or None
                attachments.append({
                    "type": "image",
                    "url": src,
                    "name": name
                })
        
        # Find all links to attachments
        for link in soup.find_all('a'):
            href = link.get('href', '')
            if href and '/attachments/token/' in href and href not in seen_urls:
                seen_urls.add(href)
                name = link.get_text(strip=True) or link.get('title') or None
                attachments.append({
                    "type": "link",
                    "url": href,
                    "name": name
                })
    
    except Exception as e:
        logger = logging.getLogger("build_solved_tickets")
        logger.debug(f"Error extracting attachments from HTML: {e}")
    
    return attachments


# In-memory cache for user lookups
_user_cache: Dict[str, Dict[str, Any]] = {}


def fetch_request_details_via_api(driver: Any, base_url: str, ticket_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch request details via Zendesk API to get requester_id, assignee_id, etc.
    
    Args:
        driver: Selenium WebDriver instance
        base_url: Base URL of Zendesk instance
        ticket_id: Request ID
        
    Returns:
        Dict with request details, or None if API fails
    """
    logger = logging.getLogger("build_solved_tickets")
    api_url = f"{base_url}/api/v2/requests/{ticket_id}.json"
    
    logger.debug(f"Fetching request details via API: {api_url}")
    
    result = browser_fetch_json(driver, api_url)
    
    if result["status"] != 200 or not result.get("data"):
        logger.debug(f"Request details API failed for {ticket_id}: status={result['status']}")
        return None
    
    request_data = result["data"].get("request", {})
    return request_data


def fetch_user_via_api(driver: Any, base_url: str, user_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch user details via Zendesk API (with caching).
    
    Args:
        driver: Selenium WebDriver instance
        base_url: Base URL of Zendesk instance
        user_id: User ID to fetch
        
    Returns:
        Dict with user details, or None if API fails
    """
    global _user_cache
    
    # Check cache first
    if user_id in _user_cache:
        return _user_cache[user_id]
    
    logger = logging.getLogger("build_solved_tickets")
    api_url = f"{base_url}/api/v2/users/{user_id}.json"
    
    logger.debug(f"Fetching user via API: {api_url}")
    
    result = browser_fetch_json(driver, api_url)
    
    if result["status"] != 200 or not result.get("data"):
        logger.debug(f"User API failed for {user_id}: status={result['status']}")
        return None
    
    user_data = result["data"].get("user", {})
    
    # Cache the result
    _user_cache[user_id] = user_data
    
    return user_data


def determine_role(
    author_id: str,
    requester_id: Optional[str],
    assignee_id: Optional[str],
    user_data: Optional[Dict[str, Any]]
) -> str:
    """
    Determine message role (requester, agent, system, unknown).
    
    Args:
        author_id: Author user ID
        requester_id: Requester user ID from request details
        assignee_id: Assignee user ID from request details
        user_data: User data from API lookup
        
    Returns:
        Role string: "requester", "agent", "system", or "unknown"
    """
    author_id_str = str(author_id) if author_id else ""
    
    # Check if author is requester
    if requester_id and author_id_str == str(requester_id):
        return "requester"
    
    # Check if author is assignee
    if assignee_id and author_id_str == str(assignee_id):
        return "agent"
    
    # Check user role from user data
    if user_data:
        user_role = user_data.get("role", "").lower()
        if user_role in ("agent", "admin"):
            return "agent"
        elif user_role == "end-user":
            return "requester"
    
    # System messages (usually have special author IDs or no author)
    if not author_id or author_id_str == "0" or author_id_str == "":
        return "system"
    
    return "unknown"


def fetch_request_comments_via_api(driver: Any, base_url: str, ticket_id: str) -> Optional[List[Dict[str, Any]]]:
    """Fetch request comments via Zendesk API."""
    logger = logging.getLogger("build_solved_tickets")
    api_url = f"{base_url}/api/v2/requests/{ticket_id}/comments.json"
    
    result = browser_fetch_json(driver, api_url)
    
    if result["status"] != 200 or not result.get("data"):
        logger.warning(f"Comments API failed for {ticket_id}: status={result['status']}")
        return None
    
    data = result["data"]
    comments = data.get("comments", [])
    
    if not isinstance(comments, list):
        logger.warning(f"API response missing 'comments' list for {ticket_id}")
        return None
    
    return comments


def build_conversation_payload(
    driver: Any,
    base_url: str,
    ticket_id: str,
    comments: List[Dict[str, Any]],
    request_details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Build normalized conversation payload from comments with proper role classification.
    
    Args:
        driver: Selenium WebDriver instance (for user lookups)
        base_url: Base URL of Zendesk instance
        ticket_id: Request ID
        comments: List of comment dicts
        request_details: Optional request details dict (if already fetched)
        
    Returns:
        Normalized conversation payload
    """
    logger = logging.getLogger("build_solved_tickets")
    
    # Fetch request details if not provided
    if request_details is None:
        request_details = fetch_request_details_via_api(driver, base_url, ticket_id)
    
    # Extract requester_id and assignee_id
    requester_id = None
    assignee_id = None
    request_metadata = {}
    
    if request_details:
        requester_id = request_details.get("requester_id")
        assignee_id = request_details.get("assignee_id")
        
        request_metadata = {
            "requester_id": str(requester_id) if requester_id else None,
            "assignee_id": str(assignee_id) if assignee_id else None,
            "status": request_details.get("status", ""),
            "subject": request_details.get("subject", ""),
            "created_at": request_details.get("created_at", ""),
            "updated_at": request_details.get("updated_at", "")
        }
    
    # Process comments into messages
    messages = []
    
    for comment in comments:
        comment_id = comment.get("id")
        author_id = comment.get("author_id")
        created_at = comment.get("created_at", "")
        
        # Get text (prefer plain_body, fallback to body)
        raw_text = comment.get("plain_body") or comment.get("body") or ""
        html_content = comment.get("html_body", "")
        
        # Normalize text
        normalized_text = normalize_text(raw_text)
        
        # Extract attachments
        attachments = []
        if html_content:
            attachments = extract_attachments_from_html(html_content)
        
        # Determine role using request details and user lookup
        user_data = None
        if author_id:
            # Try user lookup if not requester or assignee
            author_id_str = str(author_id)
            if requester_id and author_id_str != str(requester_id):
                if assignee_id and author_id_str != str(assignee_id):
                    # Not requester or assignee, try user lookup
                    user_data = fetch_user_via_api(driver, base_url, author_id_str)
        
        role = determine_role(
            author_id,
            requester_id,
            assignee_id,
            user_data
        )
        
        messages.append({
            "message_id": str(comment_id) if comment_id else "",
            "author_id": str(author_id) if author_id else "",
            "role": role,
            "created_at": created_at,
            "text": normalized_text,
            "html": html_content if html_content else None,
            "attachments": attachments
        })
    
    # Sort messages by created_at (ascending), then message_id
    messages.sort(key=lambda m: (
        m.get("created_at", ""),
        m.get("message_id", "")
    ))
    
    # Build final payload
    payload = {
        "ticket_id": str(ticket_id),
        "source_url": f"{base_url}/hc/en-us/requests/{ticket_id}",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "extraction_mode": "api",
        "request": request_metadata,
        "messages": messages
    }
    
    return payload


def login(driver: Any, base_url: str, max_retries: int = 2) -> None:
    """Login to Zendesk Help Center with retries."""
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    
    logger = logging.getLogger("build_solved_tickets")
    email, password = load_credentials()
    
    hub_url = f"{base_url}/hc/en-us/requests?query=&page=1&selected_tab_name=my-requests"
    
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Login attempt {attempt + 1}/{max_retries + 1}")
            logger.info("Navigating to requests page to check login status...")
            driver.get(hub_url)
            time.sleep(2)
            
            # Check if already logged in
            wait = WebDriverWait(driver, 10)
            try:
                wait.until(lambda d: d.execute_script("return document.readyState") == "complete")
                page_text = driver.find_element(By.TAG_NAME, "body").text.lower()
                current_url = driver.current_url.lower()
                
                if ("my requests" in page_text or "requests" in page_text) and \
                   "login" not in current_url and "sign-in" not in current_url:
                    if "/hc/" in current_url and "requests" in page_text:
                        logger.info("Already logged in")
                        return
            except:
                pass
            
            # Need to login
            logger.info("Not logged in, performing login...")
            login_url = f"{base_url}/access/login"
            driver.get(login_url)
            time.sleep(2)
            
            wait = WebDriverWait(driver, 15)
            
            # Enter email
            email_field = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='email']")))
            logger.info("Entering email...")
            email_field.clear()
            email_field.send_keys(email)
            time.sleep(1)
            
            # Click first submit
            submit_button = driver.find_element(By.CSS_SELECTOR, "input[type='submit']")
            logger.info("Clicking submit (email step)...")
            submit_button.click()
            time.sleep(3)
            
            # Enter password
            password_field = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='password']")))
            logger.info("Entering password...")
            password_field.clear()
            password_field.send_keys(password)
            time.sleep(1)
            
            # Click final submit
            submit_button = driver.find_element(By.CSS_SELECTOR, "input[type='submit']")
            logger.info("Clicking submit (password step)...")
            submit_button.click()
            time.sleep(5)
            
            # Wait for authenticated state
            wait.until(lambda d: d.execute_script("return document.readyState") == "complete")
            current_url = driver.current_url.lower()
            page_title = driver.title.lower()
            
            if "/hc/" in current_url and ("requests" in page_title or "memjet" in page_title):
                logger.info("Login successful")
                return
            else:
                raise Exception(f"Login may have failed: current_url={driver.current_url}, title={driver.title}")
                
        except Exception as e:
            logger.warning(f"Login attempt {attempt + 1} failed: {e}")
            if attempt < max_retries:
                logger.info("Retrying login...")
                time.sleep(2)
            else:
                raise Exception(f"Login failed after {max_retries + 1} attempts: {e}")
    
    raise Exception("Login failed after all retries")


def run_build_solved_tickets(
    ticket_ids: Optional[List[str]] = None,
    db_path: Optional[str] = None,
    headless: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Run the build solved tickets stage programmatically.
    
    Args:
        ticket_ids: Optional list of ticket IDs to process. If None, processes all solved tickets without detail.
        db_path: Optional path to database (defaults to Scraper/data/tickets.db)
        headless: Optional headless mode (defaults to env var or False)
        
    Returns:
        Dict with summary: total_processed, built, skipped, error_count
    """
    logger = setup_logging()
    
    # Determine headless mode
    if headless is None:
        env_path = Path(__file__).parent / ".env"
        if DOTENV_AVAILABLE and env_path.exists():
            load_dotenv(env_path)
        headless = os.getenv("ZENDESK_HEADLESS", "false").lower() == "true"
    
    base_url = "https://memjet.zendesk.com"
    
    # Initialize ticket store
    store = get_ticket_store(db_path=db_path)
    
    # Get ticket IDs to process
    if ticket_ids is None:
        ticket_ids = store.get_ticket_ids_without_detail()
    
    total_count = len(ticket_ids)
    
    if total_count == 0:
        logger.info("No solved tickets need detail building. All done!")
        return {"total_processed": 0, "built": 0, "skipped": 0, "error_count": 0}
    
    logger.info(f"Found {total_count} solved tickets to process")
    
    driver = None
    
    try:
        # Create driver and login once
        driver = get_driver(headless=headless)
        login(driver, base_url)
        
        # Process each ticket
        built_count = 0
        skipped_count = 0
        error_count = 0
        
        for i, ticket_id in enumerate(ticket_ids, 1):
            logger.info(f"[{i}/{total_count}] Processing ticket {ticket_id}...")
            
            try:
                # Fetch comments
                comments = fetch_request_comments_via_api(driver, base_url, ticket_id)
                
                if not comments:
                    logger.warning(f"  No comments fetched for {ticket_id}, skipping")
                    skipped_count += 1
                    continue
                
                # Build conversation payload
                conversation = build_conversation_payload(driver, base_url, ticket_id, comments)
                
                # Store in database
                store.set_ticket_detail(ticket_id, conversation)
                    
                    comments_count = len(conversation.get("messages", []))
                    attachments_count = sum(len(msg.get("attachments", [])) for msg in conversation.get("messages", []))
                    
                    logger.info(f"  Built: {comments_count} messages, {attachments_count} attachments")
                    built_count += 1
                    
                    # Small delay between tickets
                    time.sleep(0.3)
                
                except Exception as e:
                    logger.error(f"  Error processing {ticket_id}: {e}")
                    error_count += 1
                    continue
        
        summary = {
            "total_processed": total_count,
            "built": built_count,
            "skipped": skipped_count,
            "error_count": error_count
        }
        
        logger.info(f"BUILD SUMMARY: Processed={total_count}, Built={built_count}, Skipped={skipped_count}, Errors={error_count}")
        
        return summary
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass


def main():
    """Main entry point."""
    try:
        summary = run_build_solved_tickets()
        print("\n" + "="*60)
        print("BUILD SUMMARY")
        print("="*60)
        print(f"Total Processed: {summary['total_processed']}")
        print(f"Built: {summary['built']}")
        print(f"Skipped: {summary['skipped']}")
        print(f"Errors: {summary['error_count']}")
        print("="*60 + "\n")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\nERROR: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

