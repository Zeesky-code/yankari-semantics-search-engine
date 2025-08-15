import pandas as pd
import re
import unicodedata
from urllib.parse import urlparse
import os

def prepare_data(input_csv_path, output_csv_path):
    """
    Loads raw data, performs preprocessing, and saves the cleaned data to a new CSV.
    
    Args:
        input_csv_path (str): The path to the raw dataset CSV file.
        output_csv_path (str): The path to save the prepared CSV file.
    """
    try:
        df = pd.read_csv(input_csv_path)
        print(f"Dataset loaded successfully from '{input_csv_path}'.")
    except FileNotFoundError:
        print(f"Error: The file '{input_csv_path}' was not found.")
        return

    df.rename(columns={'text': 'original_text'}, inplace=True)

    def extract_date_from_url(url):
        try:
            path = urlparse(url).path
            match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', path)
            if match:
                return pd.Series([match.group(1), match.group(2), match.group(3)])
        except:
            pass
        return pd.Series([None, None, None])

    df[['year', 'month', 'day']] = df['url'].apply(extract_date_from_url)

    def preprocess_text(text):
        if not isinstance(text, str):
            return ''
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df['cleaned_text'] = df['original_text'].apply(preprocess_text)

    def normalize_diacritics(text):
        if not isinstance(text, str):
            return ''
        normalized_text = unicodedata.normalize('NFD', text)
        return ''.join(char for char in normalized_text if unicodedata.category(char) != 'Mn')

    df['diacriticless_text'] = df['cleaned_text'].apply(normalize_diacritics)

    print("\nFirst 5 rows of the prepared DataFrame:")
    print(df.head())

    df.to_csv(output_csv_path, index=False)
    print(f"\nPrepared data saved to '{output_csv_path}'. Ready for embedding generation.")

if __name__ == "__main__":
    INPUT_FILE_PATH = os.path.join('data', 'raw', 'yankari_data.csv')
    OUTPUT_FILE_PATH = os.path.join('data', 'prepared', 'prepared_data.csv')

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
    
    # Run the data preparation function
    prepare_data(INPUT_FILE_PATH, OUTPUT_FILE_PATH)
