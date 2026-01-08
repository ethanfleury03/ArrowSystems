# Using this file to test the scraper and the ticket class
# Also using this to gather information to better design system

import Scraper
import requests
import time
from typing import Optional


def login_with_selenium(session: requests.Session, username: str, password: str, headless: bool = False) -> bool:
    """
    Login to Zendesk using Selenium (visible browser so you can watch).
    Transfers cookies from Selenium to requests session for subsequent scraping.
    
    Args:
        session: requests.Session to transfer cookies to
        username: Login email
        password: Login password
        headless: If False, browser will be visible (default: False for debugging)
        
    Returns:
        True if login successful, False otherwise
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
    except ImportError:
        print("ERROR: Selenium not installed. Install with: pip install selenium")
        print("Also install ChromeDriver: https://chromedriver.chromium.org/")
        return False
    
    driver = None
    try:
        # Setup Chrome options
        chrome_options = Options()
        if headless:
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # Create driver
        print("Starting Chrome browser...")
        driver = webdriver.Chrome(options=chrome_options)
        driver.maximize_window()
        
        # Navigate to login page
        login_url = "https://memjet.zendesk.com/access/login"
        print(f"Navigating to: {login_url}")
        driver.get(login_url)
        time.sleep(2)  # Wait for page to load
        
        print("Looking for login form...")
        wait = WebDriverWait(driver, 10)
        
        # STEP 1: Enter email and click sign in
        print("=" * 70)
        print("STEP 1: Entering email...")
        print("=" * 70)
        
        email_selectors = [
            (By.ID, "user_email"),
            (By.NAME, "user[email]"),
            (By.CSS_SELECTOR, "input[type='email']"),
            (By.CSS_SELECTOR, "input[name*='email']"),
            (By.CSS_SELECTOR, "input[name*='Email']"),
        ]
        
        email_field = None
        for selector_type, selector_value in email_selectors:
            try:
                email_field = driver.find_element(selector_type, selector_value)
                print(f"✓ Found email field using: {selector_type}={selector_value}")
                break
            except:
                continue
        
        if not email_field:
            print("ERROR: Could not find email field. Current page:")
            print(f"  URL: {driver.current_url}")
            print(f"  Title: {driver.title}")
            input("Press Enter after inspecting the page...")
            return False
        
        print(f"Entering email: {username}")
        email_field.clear()
        email_field.send_keys(username)
        time.sleep(1)
        
        # Find and click first submit button (for email)
        submit_selectors = [
            (By.CSS_SELECTOR, "button[type='submit']"),
            (By.CSS_SELECTOR, "input[type='submit']"),
            (By.CSS_SELECTOR, "button.login-button"),
            (By.CSS_SELECTOR, "input.login-button"),
            (By.XPATH, "//button[contains(text(), 'Sign in')]"),
            (By.XPATH, "//button[contains(text(), 'Continue')]"),
            (By.XPATH, "//button[contains(text(), 'Next')]"),
        ]
        
        submit_button = None
        for selector_type, selector_value in submit_selectors:
            try:
                submit_button = driver.find_element(selector_type, selector_value)
                print(f"✓ Found submit button using: {selector_type}={selector_value}")
                break
            except:
                continue
        
        if not submit_button:
            print("ERROR: Could not find submit button for email step")
            input("Press Enter after inspecting...")
            return False
        
        print("Clicking submit button (email step)...")
        submit_button.click()
        time.sleep(3)  # Wait for password page to load
        
        # STEP 2: Enter password and click sign in again
        print("=" * 70)
        print("STEP 2: Entering password...")
        print("=" * 70)
        
        password_selectors = [
            (By.ID, "user_password"),
            (By.NAME, "user[password]"),
            (By.CSS_SELECTOR, "input[type='password']"),
            (By.CSS_SELECTOR, "input[name*='password']"),
        ]
        
        password_field = None
        for selector_type, selector_value in password_selectors:
            try:
                password_field = wait.until(EC.presence_of_element_located((selector_type, selector_value)))
                print(f"✓ Found password field using: {selector_type}={selector_value}")
                break
            except:
                continue
        
        if not password_field:
            print("ERROR: Could not find password field")
            print(f"Current URL: {driver.current_url}")
            input("Press Enter after inspecting...")
            return False
        
        print("Entering password...")
        password_field.clear()
        password_field.send_keys(password)
        time.sleep(1)
        
        # Find and click submit button again (for password)
        submit_button = None
        for selector_type, selector_value in submit_selectors:
            try:
                submit_button = driver.find_element(selector_type, selector_value)
                print(f"✓ Found submit button using: {selector_type}={selector_value}")
                break
            except:
                continue
        
        if not submit_button:
            print("ERROR: Could not find submit button for password step")
            input("Press Enter after inspecting...")
            return False
        
        print("Clicking submit button (password step)...")
        submit_button.click()
        time.sleep(5)  # Wait for login to complete and redirect
        
        # Check if login was successful
        current_url = driver.current_url
        print(f"After login, current URL: {current_url}")
        
        # Check for error messages
        error_selectors = [
            (By.CSS_SELECTOR, ".error"),
            (By.CSS_SELECTOR, ".alert-error"),
            (By.CSS_SELECTOR, "[class*='error']"),
        ]
        
        for selector_type, selector_value in error_selectors:
            try:
                error_element = driver.find_element(selector_type, selector_value)
                if error_element.is_displayed():
                    error_text = error_element.text
                    print(f"ERROR: Login failed - {error_text}")
                    input("Press Enter to continue...")
                    return False
            except:
                continue
        
        # If we're not on login page, assume success
        if 'login' not in current_url.lower():
            print("✓ Login appears successful!")
            
            # STEP 3: Navigate to Requests page
            print("=" * 70)
            print("STEP 3: Navigating to Requests...")
            print("=" * 70)
            
            # Click on user profile name in top right
            profile_selectors = [
                (By.CSS_SELECTOR, "[data-testid='user-menu']"),
                (By.CSS_SELECTOR, ".user-menu"),
                (By.CSS_SELECTOR, "[aria-label*='user']"),
                (By.CSS_SELECTOR, "[aria-label*='User']"),
                (By.CSS_SELECTOR, "button[aria-label*='profile']"),
                (By.XPATH, "//button[contains(@aria-label, 'user')]"),
                (By.XPATH, "//button[contains(@aria-label, 'User')]"),
                (By.XPATH, "//div[contains(@class, 'user')]//button"),
                (By.CSS_SELECTOR, "nav button:last-child"),  # Often the rightmost button
            ]
            
            profile_button = None
            for selector_type, selector_value in profile_selectors:
                try:
                    profile_button = wait.until(EC.element_to_be_clickable((selector_type, selector_value)))
                    print(f"✓ Found profile menu using: {selector_type}={selector_value}")
                    break
                except:
                    continue
            
            if not profile_button:
                print("WARNING: Could not find profile menu button")
                print("Trying to find by text content...")
                # Try finding by text
                try:
                    username_part = username.split('@')[0]
                    profile_button = driver.find_element(By.XPATH, f"//button[contains(text(), '{username_part}')]")
                    print("✓ Found profile by username text")
                except:
                    print("Current page structure (looking for user menu):")
                    # Print some HTML to help debug
                    print(driver.page_source[:2000])
                    input("Press Enter to continue without clicking profile (will try direct URL)...")
                    profile_button = None
            
            if profile_button:
                print("Clicking on user profile menu...")
                driver.execute_script("arguments[0].click();", profile_button)  # Use JS click in case element is obscured
                time.sleep(2)  # Wait for menu to open
            else:
                print("Skipping profile menu click, will try direct navigation...")
            
            # Now click on "Requests" link/menu item
            requests_selectors = [
                (By.XPATH, "//a[contains(text(), 'Requests')]"),
                (By.XPATH, "//button[contains(text(), 'Requests')]"),
                (By.XPATH, "//span[contains(text(), 'Requests')]"),
                (By.CSS_SELECTOR, "a[href*='requests']"),
                (By.CSS_SELECTOR, "a[href*='Requests']"),
                (By.XPATH, "//a[contains(@href, '/requests')]"),
            ]
            
            requests_link = None
            for selector_type, selector_value in requests_selectors:
                try:
                    requests_link = wait.until(EC.element_to_be_clickable((selector_type, selector_value)))
                    print(f"✓ Found Requests link using: {selector_type}={selector_value}")
                    break
                except:
                    continue
            
            if requests_link:
                # Check if the link goes to /new - if so, we need the list page instead
                href = requests_link.get_attribute('href')
                if href and '/new' in href:
                    print("Requests link goes to 'new' page, navigating to list page instead...")
                    requests_url = "https://memjet.zendesk.com/hc/en-us/requests"
                    driver.get(requests_url)
                else:
                    print("Clicking on Requests...")
                    driver.execute_script("arguments[0].click();", requests_link)
                time.sleep(3)  # Wait for Requests page to load
                print(f"✓ Navigated to Requests page: {driver.current_url}")
            else:
                # Try direct URL navigation to list page (not /new)
                print("Could not find Requests link, trying direct URL to list page...")
                requests_url = "https://memjet.zendesk.com/hc/en-us/requests"
                driver.get(requests_url)
                time.sleep(3)
                print(f"✓ Navigated directly to: {driver.current_url}")
            
            # Verify we're on the list page, not the new request page
            final_url = driver.current_url
            if '/new' in final_url:
                print("WARNING: Still on /new page, navigating to list page...")
                requests_url = "https://memjet.zendesk.com/hc/en-us/requests"
                driver.get(requests_url)
                time.sleep(2)
                print(f"✓ Now on: {driver.current_url}")
            
            # Transfer cookies from Selenium to requests session
            print("=" * 70)
            print("Transferring cookies to requests session...")
            print("=" * 70)
            for cookie in driver.get_cookies():
                session.cookies.set(cookie['name'], cookie['value'], domain=cookie.get('domain'))
            
            print(f"✓ Transferred {len(driver.get_cookies())} cookies")
            
            # Keep browser open for a bit so you can see
            print("\nKeeping browser open for 5 seconds so you can verify...")
            time.sleep(5)
            
            return True
        else:
            print("WARNING: Still on login page - login may have failed")
            print("Current page title:", driver.title)
            input("Press Enter after inspecting the page...")
            return False
            
    except Exception as e:
        print(f"ERROR during Selenium login: {e}")
        import traceback
        traceback.print_exc()
        if driver:
            print("\nCurrent page URL:", driver.current_url)
            print("Current page title:", driver.title)
            input("Press Enter to close browser...")
        return False
    finally:
        if driver:
            print("Closing browser...")
            driver.quit()


def login_function(session: requests.Session) -> bool:
    """
    Login function that uses Selenium for visual debugging.
    """
    return login_with_selenium(
        session=session,
        username="jung.gilee@memjet.partners",
        password="INK28Dm8",
        headless=False  # Set to True to hide browser
    )


if __name__ == "__main__":
    # Base URL should be just the domain, not a full page URL
    # For Zendesk web scraping, use the base domain
    BASE_URL = "https://memjet.zendesk.com"
    
    # Alternative: If using Zendesk API, you'd use:
    # BASE_URL = "https://memjet.zendesk.com/api/v2"
    # And authenticate with API token in headers instead of login_callback
    
    # EASIEST OPTION: Use cookies from your browser session
    # 1. Log into Zendesk in your browser
    # 2. Open DevTools → Application/Storage → Cookies
    # 3. Copy the session cookie value
    # 4. Use it like this:
    # scraper = Scraper.Scraper(
    #     base_url=BASE_URL,
    #     cookies={"__cfruid": "your_cookie_value", "_zendesk_session": "your_session_value"},
    #     # ... other params
    # )
    
    # Initialize scraper with credentials
    scraper = Scraper.Scraper(
        base_url=BASE_URL,
        output_dir="data",
        username="jung.gilee@memjet.partners",
        password="INK28Dm8",
        login_callback=login_function,  # Use custom login function
        rate_limit=1.0,  # 1 second between requests
        timeout=30,
    )
    
    # Run proof of concept test
    print("Running proof of concept test...")
    print("=" * 70)
    
    summary = scraper.proof_of_concept(expected_count=161)
    
    print("\n" + "=" * 70)
    print("PROOF OF CONCEPT RESULTS")
    print("=" * 70)
    print(f"Discovered ticket refs: {summary['discovered_ticket_refs']}")
    print(f"Expected count: {summary['expected_count']}")
    print(f"Count matches: {summary['count_matches']}")
    print(f"Extracted ticket ID: {summary['extracted_ticket_id']}")
    print(f"Saved path: {summary['saved_path']}")
    print("=" * 70)
