import pandas as pd
import re
import unicodedata
from urllib.parse import urlparse

def extract_date_from_url(url):
    """Extracts year, month, and day from a URL path."""
    try:
        path = urlparse(url).path
        match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', path)
        if match:
            return pd.Series([match.group(1), match.group(2), match.group(3)])
    except:
        pass
    return pd.Series([None, None, None])

def preprocess_text(text):
    """Removes extra whitespace and normalizes text."""
    if not isinstance(text, str):
        return ''
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_diacritics(text):
    """Removes diacritics/tonal marks from text for search queries."""
    if not isinstance(text, str):
        return ''
    normalized_text = unicodedata.normalize('NFD', text)
    return ''.join(char for char in normalized_text if unicodedata.category(char) != 'Mn')

def prepare_data(input_csv_path, output_csv_path):
    """
    Loads raw data, performs all preprocessing steps, and saves the cleaned data.
    """
    print("--- Step 1: Loading and preparing the data ---")

    try:
        df = pd.read_csv(input_csv_path)
        print(f"Dataset loaded successfully from '{input_csv_path}'.")
    except FileNotFoundError:
        print(f"Error: The file '{input_csv_path}' was not found.")
        return

    df.rename(columns={'text': 'original_text', 'source': 'source', 'url': 'url'}, inplace=True)

    df[['year', 'month', 'day']] = df['url'].apply(extract_date_from_url)

    df['cleaned_text'] = df['original_text'].apply(preprocess_text)

    df['diacriticless_text'] = df['cleaned_text'].apply(normalize_diacritics)

    print("\nFirst 5 rows of the prepared DataFrame:")
    print(df.head())

    df.to_csv(output_csv_path, index=False)
    print(f"\nPrepared data saved to '{output_csv_path}'.")
