#!/usr/bin/env python3
"""
Stage 1: Index all Zendesk requests into SQLite (cheap operation).

This script:
- Logs in via Selenium
- Fetches all requests via search API
- Indexes them into tickets_index table
- Does NOT fetch detailed conversations
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

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

from ticket_store import get_ticket_store


def setup_logging() -> logging.Logger:
    """Setup structured logging."""
    logger = logging.getLogger("index_requests")
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
        chrome_options.add_argument('--headless=new')  # Use new headless mode
        chrome_options.add_argument('--disable-gpu')
    
    chrome_options.add_argument('--window-size=1600,1000')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    # Additional options for better compatibility and Cloudflare bypass
    chrome_options.add_argument('--disable-extensions')
    chrome_options.add_argument('--disable-software-rasterizer')
    # Set a realistic user agent to avoid Cloudflare detection
    chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    logger = logging.getLogger("index_requests")
    logger.info(f"Creating Chrome driver (headless={headless})")
    
    driver = webdriver.Chrome(options=chrome_options)
    if not headless:
        driver.set_window_size(1600, 1000)
    
    # Execute script to remove webdriver property (helps bypass Cloudflare)
    driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
        'source': '''
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            })
        '''
    })
    
    return driver


def browser_fetch_json(driver: Any, url: str) -> Dict[str, Any]:
    """
    Fetch JSON from URL using browser context (preserves cookies/auth).
    
    Args:
        driver: Selenium WebDriver instance
        url: URL to fetch
        
    Returns:
        Dict with: ok, status, content_type, text, data (if JSON)
    """
    logger = logging.getLogger("index_requests")
    
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
            "status": result.get("status", 0),
            "content_type": result.get("content_type", ""),
            "text": result.get("text", ""),
            "data": data
        }
    except Exception as e:
        logger.warning(f"Browser fetch failed for {url}: {e}")
        return {
            "ok": False,
            "status": 0,
            "content_type": "",
            "text": str(e),
            "data": None
        }


def login(driver: Any, base_url: str, max_retries: int = 2) -> None:
    """Login to Zendesk Help Center with retries."""
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    
    logger = logging.getLogger("index_requests")
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
            
            # Wait for Cloudflare challenge to complete (if present)
            wait = WebDriverWait(driver, 30)  # Increased timeout for Cloudflare
            logger.info("Waiting for page to load (may include Cloudflare challenge)...")
            
            # Wait for page to be ready
            wait.until(lambda d: d.execute_script("return document.readyState") == "complete")
            
            # Check if we're on a Cloudflare challenge page and wait for it
            max_cloudflare_wait = 25  # seconds (Cloudflare can take 10-20 seconds)
            cloudflare_wait_start = time.time()
            while ("Just a moment" in driver.title or 
                   "Checking your browser" in driver.page_source[:1000] or
                   "challenge-platform" in driver.page_source[:1000] or
                   "cf-browser-verification" in driver.page_source[:1000]):
                elapsed = time.time() - cloudflare_wait_start
                if elapsed > max_cloudflare_wait:
                    logger.warning(f"Cloudflare challenge taking too long ({elapsed:.1f}s), continuing anyway...")
                    break
                logger.info(f"Cloudflare challenge detected (waited {elapsed:.1f}s), waiting...")
                time.sleep(2)
                # Don't refresh - just wait for Cloudflare to complete automatically
                wait.until(lambda d: d.execute_script("return document.readyState") == "complete")
            
            # Additional wait for dynamic content after Cloudflare
            time.sleep(3)
            
            # Try multiple selectors for email field
            email_field = None
            selectors = [
                "input[type='email']",
                "input[name='email']",
                "input[id*='email']",
                "input[placeholder*='email' i]",
                "#email",
                "input.email"
            ]
            
            for selector in selectors:
                try:
                    email_field = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    logger.info(f"Found email field using selector: {selector}")
                    break
                except Exception:
                    continue
            
            if not email_field:
                # Log page source for debugging
                page_source_snippet = driver.page_source[:1000] if len(driver.page_source) > 1000 else driver.page_source
                logger.error(f"Could not find email field. Page title: {driver.title}, URL: {driver.current_url}")
                logger.debug(f"Page source snippet: {page_source_snippet}")
                raise Exception("Could not locate email input field on login page (may be blocked by Cloudflare)")
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


def list_requests_via_search_api(driver: Any, base_url: str) -> list[Dict[str, Any]]:
    """
    Fetch all ticket requests via paginated Zendesk search API using next_page links.
    
    Args:
        driver: Selenium WebDriver instance
        base_url: Base URL of Zendesk instance
        
    Returns:
        List of request dicts
    """
    logger = logging.getLogger("index_requests")
    
    # Start URL with proper encoding
    start_url = f"{base_url}/api/v2/requests/search.json?include=users&per_page=100&page=1&query=*+requester%3Ame"
    
    all_requests = []
    seen_ids = set()
    pages_fetched = 0
    max_pages = 50
    max_total_tickets = 5000
    url = start_url
    
    while pages_fetched < max_pages and len(all_requests) < max_total_tickets:
        pages_fetched += 1
        logger.info(f"Fetching API page {pages_fetched}")
        
        try:
            result = browser_fetch_json(driver, url)
            
            if result["status"] != 200:
                logger.error(f"API returned status {result['status']} on page {pages_fetched}")
                break
            
            # Parse JSON response
            data = result.get("data")
            if not data or not isinstance(data, dict):
                logger.warning(f"Invalid JSON response structure on page {pages_fetched}")
                break
            
            # Extract requests list
            requests_list = data.get("requests", [])
            if not isinstance(requests_list, list):
                logger.warning(f"No 'requests' list found in response on page {pages_fetched}")
                break
            
            # Add requests (deduplicate by ID)
            page_added = 0
            for req in requests_list:
                req_id = str(req.get("id", ""))
                if req_id and req_id not in seen_ids:
                    seen_ids.add(req_id)
                    all_requests.append(req)
                    page_added += 1
            
            logger.info(f"Page {pages_fetched}: Added {page_added} new requests (total: {len(all_requests)})")
            
            # Check for next page - authoritative pagination method
            next_page_url = data.get("next_page")
            if not next_page_url:
                # No next_page means we're done
                logger.info(f"Pagination complete: no next_page on page {pages_fetched}")
                break
            
            url = next_page_url
            
        except Exception as e:
            logger.error(f"Exception during API pagination on page {pages_fetched}: {e}")
            break
    
    if len(all_requests) > 0:
        logger.info(f"API pagination complete: {len(all_requests)} total requests from {pages_fetched} pages")
    else:
        logger.warning(f"API pagination failed: no requests returned")
    
    return all_requests


def run_index_requests(db_path: Optional[str] = None, headless: Optional[bool] = None) -> Dict[str, Any]:
    """
    Run the index requests stage programmatically.
    
    Args:
        db_path: Optional path to database (defaults to Scraper/data/tickets.db)
        headless: Optional headless mode (defaults to env var or False)
        
    Returns:
        Dict with summary: indexed, total_count, solved_count, open_count
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
    logger.info("Initializing ticket store...")
    store = get_ticket_store(db_path=db_path)
    store.ensure_scrape_runs_table()  # Ensure scrape_runs table exists
    
    driver = None
    
    try:
        # Create driver
        driver = get_driver(headless=headless)
        
        # Login
        login(driver, base_url)
        
        # Fetch all requests
        logger.info("Fetching all requests via search API...")
        requests_list = list_requests_via_search_api(driver, base_url)
        
        if not requests_list:
            logger.warning("No requests fetched")
            return {"indexed": 0, "total_count": 0, "solved_count": 0, "open_count": 0}
        
        # Index requests into database
        logger.info(f"Indexing {len(requests_list)} requests into database...")
        
        indexed = 0
        for req in requests_list:
            ticket_id = str(req.get("id", ""))
            if not ticket_id:
                continue
            
            raw_status = req.get("status", "")
            subject = req.get("subject", "")
            requester_id = req.get("requester_id")
            created_at = req.get("created_at", "")
            updated_at = req.get("updated_at", "")
            
            # Normalize status: "closed" -> "solved" for consistency
            status = "solved" if raw_status == "closed" else raw_status
            
            # Compute is_solved
            is_solved = 1 if status in ("solved", "closed") else 0
            
            row = {
                "ticket_id": ticket_id,
                "status": status,
                "subject": subject,
                "requester_id": str(requester_id) if requester_id else None,
                "created_at": created_at,
                "updated_at": updated_at,
                "is_solved": is_solved
            }
            
            store.upsert_ticket_index(row)
            indexed += 1
        
        logger.info(f"Indexed {indexed} requests")
        
        # Get summary
        total_count = store.count_index()
        solved_count = store.count_solved()
        open_count = store.count_open()
        
        summary = {
            "indexed": indexed,
            "total_count": total_count,
            "solved_count": solved_count,
            "open_count": open_count
        }
        
        logger.info(f"INDEXING SUMMARY: Total={total_count}, Solved={solved_count}, Open={open_count}")
        
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
        summary = run_index_requests()
        print("\n" + "="*60)
        print("INDEXING SUMMARY")
        print("="*60)
        print(f"Total Indexed: {summary['total_count']}")
        print(f"Solved: {summary['solved_count']}")
        print(f"Open: {summary['open_count']}")
        print("="*60 + "\n")
    except Exception as e:
        print(f"\nERROR: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

