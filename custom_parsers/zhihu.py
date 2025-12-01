import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

def main(bookmark: dict) -> dict:
    """
    Custom parser for Zhihu URLs.

    Detects Zhihu URLs and fetches content using Selenium with special handling
    for login pop-ups and content extraction.

    Parameters:
        bookmark (dict): Bookmark dictionary with 'url', 'title', etc.

    Returns:
        dict: Updated bookmark dictionary or original if not Zhihu.
    """
    url = bookmark.get('url', '')
    if not url or 'zhihu.com' not in url:
        return bookmark  # Not a Zhihu URL, return unchanged

    title = bookmark.get('name', 'No Title')

    # Use Selenium to fetch Zhihu content
    progress_info = ""  # No progress info in custom parser context

    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    # Add a more realistic user agent
    options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36')

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    try:
        print(f"{progress_info} Using dedicated method to crawl Zhihu content: {title} - {url}")
        driver.get(url)
        # Wait for page to load
        time.sleep(3)

        # Detect and close login pop-up
        try:
            # Note: find_element_by_css_selector is deprecated, but keeping the original logic structure
            login_close = driver.find_element("css selector", '.Modal-closeButton')
            login_close.click()
            print(f"{progress_info} Successfully closed Zhihu login pop-up")
            time.sleep(1)
        except Exception as e:
            print(f"{progress_info} Failed to close Zhihu login pop-up or no need to close: {title} - {str(e)}")

        # Get page content
        content = driver.page_source
        soup = BeautifulSoup(content, 'html.parser')

        # Extract main content
        article = soup.select_one('.Post-RichText') or soup.select_one('.RichText')
        if article:
            result = article.get_text()
            print(f"{progress_info} Successfully extracted Zhihu article content: {title}, length: {len(result)} characters")
            bookmark['content'] = result
        else:
            result = soup.get_text()
            print(f"{progress_info} Zhihu article body not found, using full text: {title}, length: {len(result)} characters")
            bookmark['content'] = result

    except Exception as e:
        print(f"{progress_info} Zhihu crawl exception: {title} - {url} - {str(e)}")
        # Return original bookmark if crawl fails
        return bookmark
    finally:
        driver.quit()

    return bookmark