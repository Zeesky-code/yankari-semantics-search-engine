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
    dotenv.load_dotenv()
    cohere_api_key = os.getenv('COHERE_API_KEY')
    if not cohere_api_key:
        print("Error: COHERE_API_KEY environment variable is not set.")
        sys.exit(1)

    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

    # Step 1: Data Preparation
    prepare_data(INPUT_FILE_PATH, OUTPUT_FILE_PATH)
    
    # Step 2: Embedding Generation
    generate_embeddings(OUTPUT_FILE_PATH, EMBEDDINGS_FILE_PATH, cohere_api_key)

    # Step 3: FAISS Index Creation
    # TODO: Implement the FAISS indexing logic in src/search_index.py
    # search_index.create_index(EMBEDDINGS_FILE_PATH, FAISS_INDEX_PATH)

    # Step 4: Search Function & Evaluation
    # TODO: Implement the search function and evaluation logic.

    print("Pipeline finished successfully. Ready for the next feature!")

if __name__ == "__main__":
    main()