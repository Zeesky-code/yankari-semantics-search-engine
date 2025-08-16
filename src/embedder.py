import pandas as pd
import pickle
import os
import time
from sentence_transformers import SentenceTransformer

def generate_embeddings(input_csv_path, output_embeddings_path):
    """
    Generates embeddings for the cleaned text using a local sentence-transformer model.
    
    Args:
        input_csv_path (str): Path to the prepared CSV file.
        output_embeddings_path (str): Path to save the generated embeddings and metadata.
    """
    print("--- Step 2: Generating embeddings with a local model ---")
    
    # Load the prepared data
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Error: Prepared data file '{input_csv_path}' not found.")
        return
        
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    
    texts = df['diacriticless_text'].tolist()
    
    embeddings = model.encode(texts, show_progress_bar=True)
    
    embeddings_data = {
        'embeddings': embeddings,
        'metadata': df.to_dict('records')
    }
    
    with open(output_embeddings_path, 'wb') as f:
        pickle.dump(embeddings_data, f)
    
    print(f"\nEmbeddings for {len(embeddings)} articles saved to '{output_embeddings_path}'.")