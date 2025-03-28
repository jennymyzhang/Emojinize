import requests
from bs4 import BeautifulSoup
import os
import csv
from urllib.parse import urljoin
import time
import spacy
import re
from nltk.corpus import wordnet

# Load spaCy model
nlp = spacy.load("en_core_web_sm")
output_file = "emoji_dataset2.csv"
def load_existing_keywords(filename="emoji_dataset.csv"):
    existing_words = set()
    
    if not os.path.exists(filename):
        print("No existing CSV found. Starting fresh.")
        return existing_words

    try:
        with open(filename, mode="r", encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader, None)  # Skip header row
            for row in reader:
                if row:  # Ignore empty rows
                    existing_words.add(row[0].strip().lower())  # Lowercase for consistency
        print(f"Loaded {len(existing_words)} existing keywords from {filename}")
    except Exception as e:
        print(f"Error loading existing CSV: {e}")

    return existing_words

# Global variables
visited_urls = set()
words = load_existing_keywords()
print(words)
max_urls_to_visit = 500
base_url = "https://www.cnn.com/"
wiki_base_prefix = "https://www.cnn.com/"

# OpenRouter API setup
OPENROUTER_API_KEY = "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"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# ✅ Function to check if a URL is valid and should be crawled
def is_valid_url(url):
    return (
        url.startswith(wiki_base_prefix)
        and ":" not in url[len(wiki_base_prefix):]
        and url not in visited_urls
    )

# ✅ Function to check if a word is a valid English word using WordNet
def is_valid_english_word(word):
    return bool(wordnet.synsets(word))

# ✅ Function to scrape text from a webpage
def scrape_text(url):
    try:
        response = requests.get(url, timeout=20)
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove navbars, footers, sidebars
        for unwanted in soup.find_all(["nav", "footer", "aside"]):
            unwanted.decompose()

        headers = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
        paragraphs = soup.find_all("p")
        list_items = soup.find_all("li")
        table_cells = soup.find_all(["td", "th"])
        bold_text = soup.find_all("b")
        links = [a.get_text(strip=True) for a in soup.find_all("a") if a.get_text(strip=True)]

        elements = headers + paragraphs + list_items + table_cells + bold_text
        text = " ".join([element.get_text(separator=" ", strip=True) for element in elements])
        text += " " + " ".join(links)

        return text
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""

# ✅ Function to extract keywords using spaCy NER + Head Nouns
def extract_keywords(text, top_n=20):
    print("Extracting keywords with spaCy (Head Nouns + NER)...")

    doc = nlp(text)
    keywords = []

    # 1. Extract head nouns from noun chunks
    for chunk in doc.noun_chunks:
        head_noun = chunk.root.lemma_.lower().strip()

        if head_noun in {"its", "the", "this", "that", "these", "those", "head"}:
            continue
        if re.search(r"\d", head_noun) or len(head_noun) < 2:
            continue
        if re.search(r"[()]", head_noun):
            continue
        
        # ✅ Check if it's a valid English word
        if not is_valid_english_word(head_noun):
            continue

        keywords.append(head_noun)

    # 2. Extract Named Entities
    allowed_entity_types = {"PERSON", "ORG", "GPE", "LOC", "EVENT", "PRODUCT", "WORK_OF_ART"}
    for ent in doc.ents:
        if ent.label_ in allowed_entity_types:
            entity = ent.text.strip().lower()
            entity = re.sub(r"\s+", " ", entity)

            if len(entity) < 2:
                continue
            if not is_valid_english_word(entity.split()[0]):  # Basic check on the first word
                continue

            keywords.append(entity)

    unique_keywords = list(dict.fromkeys(keywords))
    print(f"Extracted terms (English head nouns + Entities): {unique_keywords[:top_n]}")

    return unique_keywords[:top_n]

# ✅ Function to get emoji representations using OpenRouter API (LLM)
def get_emoji_representation(keyword):
    prompt = (
        f"List the most accurate 3-5 emojis that directly relate to the concept of '{keyword}'. "
        "Return ONLY the emojis, make sure they are the most accurate and representative, separated by commas and wrapped by square brackets, like this: ['🌍','🔥','🌡️','🌪️','🌊']. "
        "Do not include any explanation or any text, just output the emojis. Don't include any duplicated or irrelevant emojis. Do not overuse emojis."
    )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": "google/gemini-2.0-flash-001",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.4,
        "max_tokens": 100
    }

    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=body, timeout=15)

        if response.status_code != 200:
            print("DEBUG RAW RESPONSE:", response.text)

        response.raise_for_status()
        json_response = response.json()

        message_content = json_response['choices'][0]['message']['content']
        emojis = message_content.strip().replace(" ", "")
        print(f"Emojis for '{keyword}': {emojis}")

        return emojis
    except Exception as e:
        print(f"Error getting emoji for '{keyword}': {e}")
        return ""

# ✅ Function to save results to a CSV file immediately after each entry
def save_to_csv_row(keyword, emojis, filename=output_file):
    try:
        with open(filename, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([keyword, emojis])
        print(f"Saved to CSV: {keyword} -> {emojis}")
    except Exception as e:
        print(f"Error saving {keyword} to CSV: {e}")

# ✅ Function to crawl the web
def crawl_web(start_url):
    urls_to_visit = [start_url]

    # ✅ Initialize CSV file with headers
    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Keyword", "Emojis"])

    while urls_to_visit and len(visited_urls) < max_urls_to_visit:
        url = urls_to_visit.pop(0)
        if url in visited_urls:
            continue

        print(f"\nCrawling: {url}")
        visited_urls.add(url)

        text = scrape_text(url)
        if not text:
            continue

        keywords = extract_keywords(text)
        for keyword in keywords:
            if keyword in words:
                continue
            words.add(keyword)

            emojis = get_emoji_representation(keyword)
            if emojis:
                save_to_csv_row(keyword, emojis)

            time.sleep(1)

        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")

            for link in soup.find_all("a", href=True):
                href = link["href"]
                full_url = urljoin("https://en.wikipedia.org/", href)

                if is_valid_url(full_url):
                    urls_to_visit.append(full_url)
        except Exception as e:
            print(f"Error crawling {url}: {e}")

    print(f"\nCrawling complete. Dataset saved to '{output_file}'.")

# ✅ Main function
def main():
    crawl_web(base_url)

# ✅ Run the script
if __name__ == "__main__":
    main()
