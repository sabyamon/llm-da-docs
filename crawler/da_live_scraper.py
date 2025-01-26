import requests
from bs4 import BeautifulSoup
import json
import time

BASE_DOMAIN = "https://da.live"
BASE_URL = "https://da.live/docs"
OUTPUT_FILE = "data/da_live_docs.json"
VISITED = set()  # To track visited URLs
DOCS = []  # To store scraped data


def scrape_page(url):
    """Scrape the title, content, images, and YouTube links from a page."""
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch {url}: {response.status_code}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract the page title
    title = soup.find("h1").text.strip() if soup.find("h1") else "No Title"

    # Extract the main content
    content = soup.find("main").text.strip() if soup.find("main") else "No Content"

    # Extract all image URLs
    images = []
    for img in soup.find_all("img", src=True):
        img_url = img["src"]
        # Handle relative paths
        if img_url.startswith("/"):
            img_url = BASE_URL + img_url
        images.append(img_url)

    # Extract YouTube links and embedded videos
    youtube_links = []
    for iframe in soup.find_all("iframe", src=True):
        src = iframe["src"]
        if "youtube.com" in src or "youtu.be" in src:
            youtube_links.append(src)

    # Extract YouTube links from regular anchor tags
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "youtube.com" in href or "youtu.be" in href:
            youtube_links.append(href)

    return {"url": url, "title": title, "content": content, "images": images, "youtube_links": youtube_links}


def find_links(url):
    """Find all internal links on a given page."""
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch {url}: {response.status_code}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    links = []

    # Extract all anchor tags and filter for internal links
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if href.startswith("/") and BASE_DOMAIN + href not in VISITED:  # Internal link
            links.append(BASE_DOMAIN + href)
        elif href.startswith(BASE_URL) and href not in VISITED:  # Full URL within the site
            links.append(href)

    return links


def deep_crawl(url):
    """Recursively crawl all links starting from the base URL."""
    if url in VISITED:
        return

    print(f"Visiting: {url}")
    VISITED.add(url)

    # Scrape the page
    page_data = scrape_page(url)
    if page_data:
        DOCS.append(page_data)

    # Find and visit all internal links
    links = find_links(url)
    print(links)
    for link in links:
        time.sleep(1)  # Be polite and avoid overwhelming the server
        deep_crawl(link)


def start_crawl():
    """Start the deep crawl from the base URL."""
    deep_crawl(BASE_URL)
    print(f"Scraped {len(DOCS)} pages.")

    # Save the data to a JSON file
    with open(OUTPUT_FILE, "w") as f:
        json.dump(DOCS, f, indent=2)
    print(f"Scraped data saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    start_crawl()
