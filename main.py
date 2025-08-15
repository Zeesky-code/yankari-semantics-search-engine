import os
import dotenv
from src.data_processor import prepare_data
from src.embedder import generate_embeddings

# --- Project Configuration ---
INPUT_FILE_PATH = os.path.join('data', 'raw', 'yankari_data.csv')
OUTPUT_FILE_PATH = os.path.join('data', 'prepared', 'prepared_data.csv')
EMBEDDINGS_FILE_PATH = os.path.join('data', 'prepared', 'embeddings.pkl')
FAISS_INDEX_PATH = os.path.join('models', 'faiss_index.index')
def main():
    """
    The main function to run the entire semantic search pipeline.
    """
    print("Starting Yoruba Semantic Search Engine pipeline...")

    cohere_api_key = os.getenv('COHERE_API_KEY')
    if not cohere_api_key:
        print("Error: COHERE_API_KEY environment variable is not set.")
        sys.exit(1)

    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

    if not os.path.exists(OUTPUT_FILE_PATH):
        prepare_data(INPUT_FILE_PATH, OUTPUT_FILE_PATH)
    else:
        print(f"Prepared data found at '{OUTPUT_FILE_PATH}'. Skipping Step 1.")
    
    if not os.path.exists(EMBEDDINGS_FILE_PATH):
        generate_embeddings(OUTPUT_FILE_PATH, EMBEDDINGS_FILE_PATH, cohere_api_key)
    else:
        print(f"Embeddings found at '{EMBEDDINGS_FILE_PATH}'. Skipping Step 2.")

    if not os.path.exists(FAISS_INDEX_PATH):
        create_index(EMBEDDINGS_FILE_PATH, FAISS_INDEX_PATH)
    else:
        print(f"FAISS index found at '{FAISS_INDEX_PATH}'. Skipping Step 3.")
    
    sample_query = "Climate change in Africa"
    search_results = semantic_search(sample_query, cohere_api_key, EMBEDDINGS_FILE_PATH, FAISS_INDEX_PATH)

    print("\n--- Search Results ---")
    if search_results:
        for i, result in enumerate(search_results):
            print(f"{i+1}. Score: {result['score']:.4f}")
            print(f"   Title: {result.get('title', 'N/A')}")
            print(f"   Date: {result.get('year', 'N/A')}-{result.get('month', 'N/A')}-{result.get('day', 'N/A')}")
            print(f"   Source: {result.get('source', 'N/A')}")
            print("-" * 20)
    else:
        print("No results found or an error occurred.")

    print("Pipeline finished successfully. Ready for the next feature!")

if __name__ == "__main__":
    main()
