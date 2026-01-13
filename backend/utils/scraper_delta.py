"""
Delta check logic for identifying new solved tickets from Zendesk.

This module compares ticket IDs from the Zendesk API with those in the local database
to identify new solved/closed tickets that need to be processed.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Set, Dict, Any

# Add Scraper to path to import Scraper modules
project_root = Path(__file__).parent.parent.parent
scraper_path = project_root / "Scraper"
if str(scraper_path) not in sys.path:
    sys.path.insert(0, str(scraper_path))

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

import sqlite3

from ..logging_config import get_logger

logger = get_logger(__name__)


def load_credentials() -> tuple[str, str]:
    """Load credentials from Scraper/.env file."""
    env_path = project_root / "Scraper" / ".env"
    
    if DOTENV_AVAILABLE and env_path.exists():
        load_dotenv(env_path)
    
    email = os.getenv("ZENDESK_EMAIL")
    password = os.getenv("ZENDESK_PASSWORD")
    
    if not email or not password:
        raise ValueError(
            "ZENDESK_EMAIL and ZENDESK_PASSWORD must be set in Scraper/.env file"
        )
    
    return email, password


def get_driver(headless: bool = True) -> any:
    """Create Selenium Chrome driver."""
    if not SELENIUM_AVAILABLE:
        raise RuntimeError("Selenium not available. Install with: pip install selenium")
    
    chrome_options = Options()
    
    if headless:
        chrome_options.add_argument('--headless')
    
    chrome_options.add_argument('--window-size=1600,1000')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    driver = webdriver.Chrome(options=chrome_options)
    driver.set_window_size(1600, 1000)
    
    return driver


def browser_fetch_json(driver: any, url: str) -> dict:
    """
    Fetch JSON from URL using browser context (preserves cookies/auth).
    
    Args:
        driver: Selenium WebDriver instance
        url: URL to fetch
        
    Returns:
        Dict with: ok, status, content_type, text, data (if JSON)
    """
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
        logger.error(f"Error fetching JSON from {url}: {e}")
        return {
            "ok": False,
            "status": 0,
            "content_type": "",
            "text": str(e),
            "data": None
        }


def check_login_status(driver: any, base_url: str) -> bool:
    """Check if already logged into Zendesk."""
    try:
        driver.get(f"{base_url}/hc/requests")
        driver.implicitly_wait(2)
        current_url = driver.current_url.lower()
        return "/hc/" in current_url and "login" not in current_url
    except Exception:
        return False


def login(driver: any, base_url: str, email: str, password: str, max_retries: int = 3) -> None:
    """Login to Zendesk via Selenium."""
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    import time
    
    for attempt in range(max_retries + 1):
        try:
            # Check if already logged in
            if check_login_status(driver, base_url):
                logger.info("Already logged in")
                return
            
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


def fetch_all_ticket_ids_from_api(driver: any, base_url: str) -> List[str]:
    """
    Fetch all ticket IDs from Zendesk API (lightweight, just IDs).
    
    Args:
        driver: Selenium WebDriver instance
        base_url: Base URL of Zendesk instance
        
    Returns:
        List of ticket ID strings
    """
    start_url = f"{base_url}/api/v2/requests/search.json?include=users&per_page=100&page=1&query=*+requester%3Ame"
    
    all_ids = []
    seen_ids: Set[str] = set()
    pages_fetched = 0
    max_pages = 50
    max_total_tickets = 5000
    url = start_url
    
    while pages_fetched < max_pages and len(all_ids) < max_total_tickets:
        pages_fetched += 1
        logger.info(f"Fetching API page {pages_fetched} for ticket IDs")
        
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
            
            # Add ticket IDs (deduplicate)
            page_added = 0
            for req in requests_list:
                req_id = str(req.get("id", ""))
                if req_id and req_id not in seen_ids:
                    seen_ids.add(req_id)
                    all_ids.append(req_id)
                    page_added += 1
            
            logger.info(f"Page {pages_fetched}: Added {page_added} new ticket IDs (total: {len(all_ids)})")
            
            # Check for next page
            next_page_url = data.get("next_page")
            if not next_page_url:
                logger.info(f"Pagination complete: no next_page on page {pages_fetched}")
                break
            
            url = next_page_url
            
        except Exception as e:
            logger.error(f"Exception during API pagination on page {pages_fetched}: {e}")
            break
    
    logger.info(f"Fetched {len(all_ids)} total ticket IDs from {pages_fetched} pages")
    return all_ids


def fetch_ticket_status(driver: any, base_url: str, ticket_id: str) -> Optional[str]:
    """
    Fetch status for a single ticket (lightweight check).
    
    Args:
        driver: Selenium WebDriver instance
        base_url: Base URL of Zendesk instance
        ticket_id: Ticket ID to check
        
    Returns:
        Status string or None if fetch failed
    """
    url = f"{base_url}/api/v2/requests/{ticket_id}.json"
    
    try:
        result = browser_fetch_json(driver, url)
        
        if result["status"] != 200:
            logger.debug(f"Failed to fetch ticket {ticket_id}: status {result['status']}")
            return None
        
        data = result.get("data")
        if not data or not isinstance(data, dict):
            return None
        
        request_data = data.get("request")
        if not request_data or not isinstance(request_data, dict):
            return None
        
        return request_data.get("status")
    except Exception as e:
        logger.debug(f"Error fetching status for ticket {ticket_id}: {e}")
        return None


def get_db_ticket_ids(db_path: str) -> Set[str]:
    """
    Get all ticket IDs from the database.
    
    Args:
        db_path: Path to SQLite database
        
    Returns:
        Set of ticket ID strings
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT ticket_id FROM tickets_index")
        rows = cursor.fetchall()
        return {row["ticket_id"] for row in rows}
    finally:
        conn.close()


def quick_count_check(db_path: str) -> dict:
    """
    Quick check to compare ticket counts without full scraping.
    
    This function:
    1. Gets DB count (fast, no API call needed)
    2. Returns that we should proceed with full scrape (since we can't get API count without login)
    
    Note: A true quick check would require login, which defeats the purpose.
    Instead, we'll do a lightweight Stage 1 that stops early if no new tickets.
    
    Args:
        db_path: Path to SQLite database
        
    Returns:
        Dict with: has_new_tickets (always True to proceed), message
    """
    logger.info("Performing quick count check (will do lightweight Stage 1 check)")
    
    # For now, we can't do a true quick check without login.
    # The optimization will happen in Stage 1 - it will index and then check if there are new solved tickets.
    # Return True to proceed, but Stage 1 will handle the actual check efficiently.
    
    import sqlite3
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM tickets_index")
            row = cursor.fetchone()
            db_count = row["count"] if row else 0
        finally:
            conn.close()
        
        return {
            "api_count": None,
            "db_count": db_count,
            "has_new_tickets": True,  # Always proceed - Stage 1 will do the real check
            "message": f"DB has {db_count} tickets. Proceeding with lightweight check..."
        }
    except Exception as e:
        logger.warning(f"Could not get DB count: {e}")
        return {
            "api_count": None,
            "db_count": None,
            "has_new_tickets": True,
            "message": "Proceeding with scrape..."
        }


def get_new_solved_ticket_ids(db_path: str) -> List[str]:
    """
    Identify new solved/closed ticket IDs that are not in the database.
    
    This function:
    1. Fetches all ticket IDs from Zendesk API
    2. Compares with database ticket IDs
    3. For new IDs, checks if they are solved/closed
    4. Returns list of new solved ticket IDs
    
    Args:
        db_path: Path to SQLite database
        
    Returns:
        List of new solved ticket ID strings
    """
    logger.info("Starting delta check for new solved tickets")
    
    # Load credentials
    email, password = load_credentials()
    base_url = "https://memjet.zendesk.com"
    
    # Determine headless mode
    env_path = project_root / "Scraper" / ".env"
    if DOTENV_AVAILABLE and env_path.exists():
        load_dotenv(env_path)
    headless = os.getenv("ZENDESK_HEADLESS", "true").lower() == "true"
    
    driver = None
    
    try:
        # Create driver and login
        logger.info("Creating Selenium driver and logging in...")
        driver = get_driver(headless=headless)
        login(driver, base_url, email, password)
        
        # Fetch all ticket IDs from API
        logger.info("Fetching all ticket IDs from Zendesk API...")
        api_ids = fetch_all_ticket_ids_from_api(driver, base_url)
        logger.info(f"Found {len(api_ids)} tickets in Zendesk")
        
        # Get database ticket IDs
        logger.info("Loading ticket IDs from database...")
        db_ids = get_db_ticket_ids(db_path)
        logger.info(f"Found {len(db_ids)} tickets in database")
        
        # Compute new IDs
        new_ids = [tid for tid in api_ids if tid not in db_ids]
        logger.info(f"Found {len(new_ids)} new ticket IDs")
        
        if not new_ids:
            logger.info("No new tickets found")
            return []
        
        # Check status for new tickets (only keep solved/closed)
        logger.info(f"Checking status for {len(new_ids)} new tickets...")
        solved_ids = []
        
        for i, ticket_id in enumerate(new_ids, 1):
            if i % 10 == 0:
                logger.info(f"Checked {i}/{len(new_ids)} tickets...")
            
            status = fetch_ticket_status(driver, base_url, ticket_id)
            if status and status.lower() in ("solved", "closed"):
                solved_ids.append(ticket_id)
        
        logger.info(f"Found {len(solved_ids)} new solved/closed tickets")
        return solved_ids
        
    except Exception as e:
        logger.error(f"Error during delta check: {e}", exc_info=True)
        raise
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass
