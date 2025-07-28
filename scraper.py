import requests
from bs4 import BeautifulSoup
import logging
from utils import save_to_file

# ---- Config ----
file_path = "/Users/trishika/Documents/My Projects/[1] HogRAG/urls.txt"

logging.basicConfig(
    level = logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("hograg.log"),
        logging.StreamHandler()]
)


# --------------------
# Read URLs from File
# --------------------
def read_urls_from_file(file_path):
    """
    Reads a list of URLs from a file.

    Args:
        file_path (str): Path to the text file containing URLs.

    Returns:
        list of str: A list of non-empty, stripped URLs.
    """
    try:
        with open(file_path, 'r') as file:
            urls = [line.strip() for line in file if line.strip()]
            logging.info(f"Successfully read {len(urls)} URLs from: {file_path}")
            return urls
    except FileNotFoundError:
        logging.error(f"File Not Found: {file_path}")
    except:
        logging.error(f"Unexpected error while handling file: {file_path}")
    return []


# -----------------------------
# Scrape and Save Multiple URLs
# -----------------------------
def scrape_urls(url_list):
    """
    Scrapes a list of URLs and saves their extracted content.

    Args:
        url_list (list of str): List of URLs to scrape.

    Returns:
        None
    """

    success_count = 0  # ADDED: Track how many URLs were successfully scraped
    for url in url_list:
        try:
            logging.info(f"Scrapping URL: {url}")
            content = scrape_content(url)

            if content: # ADDED: Check for non-empty content
                save_to_file(content)
                logging.info(f"Successfully scraped and saved: {url}")
                success_count += 1
            else:
                logging.warning(f"No content extracted from: {url}")
                
        except Exception as e:
            logging.exception(f"Failed to scrape: {url}")
    
    logging.info(f"Scraped {success_count} pages out of {len(url_list)} URLS")

# -----------------------------------
# Extract Main Content from a Web Page
# -----------------------------------
def scrape_content(URL):
    """
    Fetches and parses the main content of a Wikipedia-like web page.

    Args:
        URL (str): The URL to scrape.

    Returns:
        str: Extracted plain text content, or empty string on failure.
    """
    try:
        response = requests.get(URL, timeout = 10)
        response.raise_for_status()
        logging.info(f"Fetched content from: {URL}")

        soup = BeautifulSoup(response.text, "html.parser")
        main_content = soup.find("div",{"class": "mw-content-ltr mw-parser-output"})

        if not main_content:
            logging.warning(f"Main content not found in {URL}")
            return ""
            
        all_tags = main_content.find_all(["p"])

        # Exclude <p> tags with class "caption"- those are text under images 
        filtered_tags = [
            tag for tag in all_tags 
            if not(tag.name == "p" and "caption" in tag.get("class", []))]

        page_text = " ".join(tag.get_text(strip = True, separator = " ") 
                for tag in filtered_tags if tag.get_text(strip = True))
        logging.info(f"Extracted text content length: {len(page_text)} characters from {URL}")
        return page_text
    
    except requests.Timeout:
        logging.error(f"Request to {URL} timed out.")
    except requests.ConnectionError:
        logging.error(f"Connection error occurred while requesting {URL}.")
    except requests.HTTPError as http_err:
        logging.error(f"HTTP error occurred for {URL}: {http_err}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while scraping: {e}")

    return ""


if __name__ == "__main__":
    url_list = read_urls_from_file(file_path)
    if url_list:
        scrape_urls(url_list)



