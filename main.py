import os
from src.data_processor import prepare_data

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

    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

    # Step 1: Data Preparation
    prepare_data(INPUT_FILE_PATH, OUTPUT_FILE_PATH)
    
    # Step 2: Embedding Generation
    # TODO: Implement the embedding generation logic in src/embedder.py
    # embedder.generate_embeddings(OUTPUT_FILE_PATH, EMBEDDINGS_FILE_PATH)

    # Step 3: FAISS Index Creation
    # TODO: Implement the FAISS indexing logic in src/search_index.py
    # search_index.create_index(EMBEDDINGS_FILE_PATH, FAISS_INDEX_PATH)

    # Step 4: Search Function & Evaluation
    # TODO: Implement the search function and evaluation logic.

    print("Pipeline finished successfully. Ready for the next feature!")

if __name__ == "__main__":
    main()