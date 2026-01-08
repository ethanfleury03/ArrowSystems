#!/usr/bin/env python3
"""
One-off helper script to dump Zendesk Help Center HTML for analysis.

Collects:
- List page HTML
- First ticket detail page HTML
- Summary JSON with metadata

Run: python Scraper/dump_zendesk_html.py
"""

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
except ImportError:
    print("ERROR: Selenium not installed. Install with: pip install selenium")
    exit(1)


def setup_driver(headless: bool = False):
    """Setup Chrome WebDriver."""
    chrome_options = Options()
    if headless:
        chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.maximize_window()
        return driver
    except Exception as e:
        print(f"ERROR: Failed to create Chrome driver: {e}")
        print("Make sure ChromeDriver is installed and in PATH")
        print("Or install webdriver-manager: pip install webdriver-manager")
        exit(1)


def login_to_zendesk(driver, username: str, password: str):
    """
    Login to Zendesk using Selenium.
    Returns True if successful, False otherwise.
    """
    print("=" * 70)
    print("STEP 1: Logging into Zendesk...")
    print("=" * 70)
    
    login_url = "https://memjet.zendesk.com/access/login"
    print(f"Navigating to: {login_url}")
    driver.get(login_url)
    time.sleep(2)
    
    wait = WebDriverWait(driver, 10)
    
    # Enter email
    email_selectors = [
        (By.CSS_SELECTOR, "input[type='email']"),
        (By.ID, "user_email"),
        (By.NAME, "user[email]"),
    ]
    
    email_field = None
    for selector_type, selector_value in email_selectors:
        try:
            email_field = driver.find_element(selector_type, selector_value)
            print(f"✓ Found email field")
            break
        except:
            continue
    
    if not email_field:
        print("ERROR: Could not find email field")
        return False
    
    print(f"Entering email: {username}")
    email_field.clear()
    email_field.send_keys(username)
    time.sleep(1)
    
    # Click first submit
    submit_button = driver.find_element(By.CSS_SELECTOR, "input[type='submit']")
    print("Clicking submit (email step)...")
    submit_button.click()
    time.sleep(3)
    
    # Enter password
    password_field = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='password']")))
    print("Entering password...")
    password_field.clear()
    password_field.send_keys(password)
    time.sleep(1)
    
    # Click second submit
    submit_button = driver.find_element(By.CSS_SELECTOR, "input[type='submit']")
    print("Clicking submit (password step)...")
    submit_button.click()
    time.sleep(5)
    
    # Check if login successful
    current_url = driver.current_url
    if 'login' not in current_url.lower():
        print(f"✓ Login successful! Current URL: {current_url}")
        return True
    else:
        print("ERROR: Login may have failed - still on login page")
        return False


def navigate_to_requests(driver):
    """Navigate to the Requests list page."""
    print("=" * 70)
    print("STEP 2: Navigating to Requests page...")
    print("=" * 70)
    
    # Try to click profile menu and Requests link
    wait = WebDriverWait(driver, 10)
    
    try:
        profile_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".user-menu")))
        print("Clicking profile menu...")
        driver.execute_script("arguments[0].click();", profile_button)
        time.sleep(2)
        
        requests_link = driver.find_element(By.CSS_SELECTOR, "a[href*='requests']")
        href = requests_link.get_attribute('href')
        if href and '/new' in href:
            # Navigate directly to list page
            requests_url = "https://memjet.zendesk.com/hc/en-us/requests"
            print(f"Requests link goes to /new, navigating to list page: {requests_url}")
            driver.get(requests_url)
        else:
            print("Clicking Requests link...")
            driver.execute_script("arguments[0].click();", requests_link)
        time.sleep(3)
    except Exception as e:
        print(f"Could not use menu navigation: {e}")
        print("Navigating directly to Requests page...")
        requests_url = "https://memjet.zendesk.com/hc/en-us/requests"
        driver.get(requests_url)
        time.sleep(3)
    
    # Ensure we're on the list page, not /new
    current_url = driver.current_url
    if '/new' in current_url:
        print("Still on /new page, navigating to list...")
        requests_url = "https://memjet.zendesk.com/hc/en-us/requests"
        driver.get(requests_url)
        time.sleep(2)
    
    print(f"✓ On Requests page: {driver.current_url}")


def extract_ticket_urls(driver):
    """
    Extract all ticket URLs from the list page.
    Returns list of (ticket_id, absolute_url) tuples.
    """
    print("=" * 70)
    print("STEP 3: Extracting ticket URLs...")
    print("=" * 70)
    
    # Wait for page to fully load and content to appear
    wait = WebDriverWait(driver, 20)
    
    # Try multiple selectors to find ticket links
    print("Waiting for ticket links to appear...")
    try:
        # Wait for any link that might be a ticket link
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='requests']")))
        print("✓ Found links containing 'requests'")
    except:
        print("WARNING: No links with 'requests' found, but continuing...")
    
    # Extra wait for dynamic content
    time.sleep(5)
    
    # Try multiple selectors
    selectors = [
        "a[href*='/hc/en-us/requests/']",
        "a[href*='/requests/']",
        "a[href*='requests']",
        "a[href*='request']",
    ]
    
    all_links = []
    for selector in selectors:
        try:
            links = driver.find_elements(By.CSS_SELECTOR, selector)
            print(f"  Selector '{selector}': found {len(links)} links")
            all_links.extend(links)
        except:
            continue
    
    # Deduplicate by href
    unique_links = {}
    for link in all_links:
        try:
            href = link.get_attribute('href')
            if href:
                unique_links[href] = link
        except:
            continue
    
    print(f"Total unique links found: {len(unique_links)}")
    
    # Debug: Print first 10 hrefs to see what we're getting
    print("\nFirst 10 hrefs found:")
    for i, (href, link) in enumerate(list(unique_links.items())[:10], 1):
        text = link.text[:50] if link.text else "(no text)"
        print(f"  {i}. {href} (text: '{text}')")
    
    # Multiple patterns to try
    patterns = [
        re.compile(r'/hc/en-us/requests/(\d+)'),  # Full path
        re.compile(r'/requests/(\d+)'),  # Short path
        re.compile(r'requests/(\d+)'),  # Anywhere in URL
        re.compile(r'/(\d+)(?:-|$)'),  # Just digits at end of path
    ]
    
    ticket_urls = []
    seen_ids = set()
    
    for href, link in unique_links.items():
        # Normalize to absolute URL
        if not href.startswith('http'):
            href = urljoin("https://memjet.zendesk.com", href)
        
        # Try each pattern
        for pattern in patterns:
            match = pattern.search(href)
            if match:
                ticket_id = match.group(1)
                
                # Validate it's actually a ticket ID (reasonable length)
                if ticket_id.isdigit() and len(ticket_id) >= 3:  # At least 3 digits
                    # Deduplicate by ticket_id
                    if ticket_id not in seen_ids:
                        seen_ids.add(ticket_id)
                        ticket_urls.append((ticket_id, href))
                        break
    
    # Sort by ticket_id (as integer)
    ticket_urls.sort(key=lambda x: int(x[0]))
    
    print(f"\n✓ Found {len(ticket_urls)} unique ticket URLs")
    if ticket_urls:
        print("First 5 ticket URLs:")
        for ticket_id, url in ticket_urls[:5]:
            print(f"  ID: {ticket_id} -> {url}")
    
    return ticket_urls


def save_list_page(driver, output_dir: Path):
    """Save the list page HTML."""
    print("=" * 70)
    print("STEP 4: Saving list page HTML...")
    print("=" * 70)
    
    # Wait for page to load and at least one ticket link to appear
    wait = WebDriverWait(driver, 15)
    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='/hc/en-us/requests/']")))
        print("✓ Ticket links detected on page")
    except:
        print("WARNING: No ticket links found, but saving HTML anyway...")
    
    # Wait for page to be fully loaded
    wait.until(lambda d: d.execute_script("return document.readyState") == "complete")
    time.sleep(2)  # Extra wait for dynamic content
    
    html_content = driver.page_source
    list_html_path = output_dir / "list_page.html"
    
    with open(list_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ Saved list page HTML ({len(html_content)} chars) to: {list_html_path}")
    
    return list_html_path


def save_first_ticket(driver, first_ticket_url: str, output_dir: Path):
    """Navigate to first ticket and save its HTML."""
    print("=" * 70)
    print("STEP 5: Saving first ticket HTML...")
    print("=" * 70)
    
    print(f"Navigating to: {first_ticket_url}")
    driver.get(first_ticket_url)
    
    # Wait for page to load
    wait = WebDriverWait(driver, 15)
    wait.until(lambda d: d.execute_script("return document.readyState") == "complete")
    
    # Wait for reasonable content size (indicates page loaded)
    for _ in range(10):
        html_length = len(driver.page_source)
        if html_length > 10000:  # At least 10k chars
            break
        time.sleep(1)
    else:
        print("WARNING: Page content seems small, but saving anyway...")
    
    time.sleep(2)  # Extra wait for dynamic content
    
    html_content = driver.page_source
    ticket_html_path = output_dir / "first_ticket.html"
    
    with open(ticket_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ Saved ticket HTML ({len(html_content)} chars) to: {ticket_html_path}")
    
    return ticket_html_path


def create_summary(list_url: str, list_html_path: Path, ticket_html_path: Path,
                   ticket_urls: list, first_ticket_id: str, first_ticket_url: str,
                   output_dir: Path):
    """Create summary JSON file."""
    print("=" * 70)
    print("STEP 6: Creating summary JSON...")
    print("=" * 70)
    
    summary = {
        "list_url": list_url,
        "list_html_path": str(list_html_path.relative_to(output_dir.parent)),
        "detail_html_path": str(ticket_html_path.relative_to(output_dir.parent)),
        "ticket_links_found": len(ticket_urls),
        "first_10_ticket_urls": [url for _, url in ticket_urls[:10]],
        "first_ticket_id": first_ticket_id,
        "first_ticket_url": first_ticket_url,
        "regex_used": r"/hc/en-us/requests/(\d+)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat()
    }
    
    summary_path = output_dir / "summary.json"
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved summary JSON to: {summary_path}")
    
    return summary


def main():
    """Main execution."""
    print("=" * 70)
    print("Zendesk HTML Dump Helper")
    print("=" * 70)
    
    # Configuration
    BASE_URL = "https://memjet.zendesk.com"
    LIST_URL = f"{BASE_URL}/hc/en-us/requests?query=&page=1&selected_tab_name=my-requests"
    
    # Credentials (you'll need to enter these or set as env vars)
    USERNAME = "jung.gilee@memjet.partners"
    PASSWORD = "INK28Dm8"
    
    # Output directory
    script_dir = Path(__file__).parent
    output_dir = script_dir / "debug_dump"
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nOutput directory: {output_dir.absolute()}")
    print(f"List URL: {LIST_URL}\n")
    
    driver = None
    try:
        # Setup driver
        driver = setup_driver(headless=False)
        
        # Login
        if not login_to_zendesk(driver, USERNAME, PASSWORD):
            print("ERROR: Login failed")
            return
        
        # Navigate to Requests
        navigate_to_requests(driver)
        
        # Extract ticket URLs
        ticket_urls = extract_ticket_urls(driver)
        
        if not ticket_urls:
            print("\nWARNING: No ticket URLs found with standard patterns")
            print("Current URL:", driver.current_url)
            print("Page title:", driver.title)
            print("\nSaving HTML anyway so you can inspect it manually...")
            print("The HTML file will contain all links - you can search for 'requests' in it.")
            
            # Save the HTML anyway
            list_html_path = save_list_page(driver, output_dir)
            
            # Try to find ANY link that might be a ticket
            print("\nTrying to find any link that might be a ticket...")
            all_links = driver.find_elements(By.TAG_NAME, "a")
            print(f"Total <a> tags on page: {len(all_links)}")
            
            # Look for links with numeric IDs in various formats
            potential_tickets = []
            for link in all_links[:50]:  # Check first 50 links
                href = link.get_attribute('href')
                text = link.text
                if href and ('request' in href.lower() or 'ticket' in href.lower()):
                    potential_tickets.append((href, text[:30] if text else ''))
            
            if potential_tickets:
                print(f"\nFound {len(potential_tickets)} potential ticket links:")
                for href, text in potential_tickets[:10]:
                    print(f"  {href} (text: '{text}')")
            
            print(f"\nHTML saved to: {list_html_path}")
            print("Please inspect the HTML file to find the correct selector pattern.")
            input("\nPress Enter to close browser...")
            return
        
        # Save list page
        list_html_path = save_list_page(driver, output_dir)
        
        # Get first ticket
        first_ticket_id, first_ticket_url = ticket_urls[0]
        print(f"\nFirst ticket ID: {first_ticket_id}")
        print(f"First ticket URL: {first_ticket_url}\n")
        
        # Save first ticket
        ticket_html_path = save_first_ticket(driver, first_ticket_url, output_dir)
        
        # Create summary
        summary = create_summary(
            LIST_URL,
            list_html_path,
            ticket_html_path,
            ticket_urls,
            first_ticket_id,
            first_ticket_url,
            output_dir
        )
        
        # Print summary (without HTML content)
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Ticket links found: {summary['ticket_links_found']}")
        print(f"First ticket ID: {summary['first_ticket_id']}")
        print(f"First ticket URL: {summary['first_ticket_url']}")
        print(f"\nFirst 10 ticket URLs:")
        for i, url in enumerate(summary['first_10_ticket_urls'], 1):
            print(f"  {i}. {url}")
        print(f"\nFiles saved to: {output_dir.absolute()}")
        print("=" * 70)
        
        # Keep browser open briefly
        print("\nKeeping browser open for 5 seconds for verification...")
        time.sleep(5)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        if driver:
            print(f"\nCurrent URL: {driver.current_url}")
            print(f"Page title: {driver.title}")
            input("Press Enter to close browser...")
    
    finally:
        if driver:
            print("Closing browser...")
            driver.quit()


if __name__ == "__main__":
    main()

