import cohere
import pandas as pd
import pickle
import os
import time

def generate_embeddings(input_csv_path, output_embeddings_path, cohere_api_key):
    """
    Generates embeddings for the cleaned text using the Cohere API and saves them.
    
    Args:
        input_csv_path (str): Path to the prepared CSV file.
        output_embeddings_path (str): Path to save the generated embeddings and metadata.
        cohere_api_key (str): Your Cohere API key.
    """
    print("--- Step 2: Generating embeddings with Cohere API ---")
    
    # Load the prepared data
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Error: Prepared data file '{input_csv_path}' not found.")
        return
        
    co = cohere.Client(cohere_api_key)
    
    texts = df['diacriticless_text'].tolist()
    embeddings = []
    metadata = df.to_dict('records')
    
    # Batch size is set to 64 as per the plan
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Use exponential backoff to handle rate limits
        attempt = 0
        while attempt < 5:
            try:
                response = co.embed(
                    model='embed-multilingual-v3.0',
                    texts=batch_texts,
                    input_type="search_query"
                )
                embeddings.extend(response.embeddings)
                print(f"Processed batch {i // batch_size + 1} of {len(texts) // batch_size + 1}")
                break
            except cohere.core.api_error.ApiError as e:
                print(f"API Error: {e}. Retrying after a delay...")
                time.sleep(2 ** attempt)  # Exponential backoff
                attempt += 1
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                break
        if attempt == 5:
            print(f"Failed to process batch after multiple retries. Exiting.")
            return

    # Check if we got all embeddings
    if len(embeddings) != len(texts):
        print("Warning: Number of embeddings does not match number of texts.")
        return
        
    # Combine embeddings and metadata into a single object
    embeddings_data = {
        'embeddings': embeddings,
        'metadata': metadata
    }
    
    # Save the embeddings data to disk using pickle
    with open(output_embeddings_path, 'wb') as f:
        pickle.dump(embeddings_data, f)
    
    print(f"\nEmbeddings for {len(embeddings)} articles saved to '{output_embeddings_path}'.")
    