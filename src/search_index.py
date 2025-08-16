import numpy as np
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer

def create_index(embeddings_path, index_path):
    """
    Loads embeddings and builds a FAISS index.
    
    Args:
        embeddings_path (str): Path to the pickled embeddings file.
        index_path (str): Path to save the FAISS index.
    """
    print("--- Step 3: Building FAISS index ---")

    try:
        with open(embeddings_path, 'rb') as f:
            embeddings_data = pickle.load(f)
        embeddings = np.array(embeddings_data['embeddings']).astype('float32')
        print(f"Embeddings loaded successfully. Shape: {embeddings.shape}")
    except FileNotFoundError:
        print(f"Error: Embeddings file '{embeddings_path}' not found.")
        return
    
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    print(f"FAISS index built and saved to '{index_path}'.")


def semantic_search(query, embeddings_path, index_path, k=5):
    """
    Performs a semantic search on the FAISS index.
    
    Args:
        query (str): The search query.
        embeddings_path (str): Path to the pickled embeddings file.
        index_path (str): Path to the saved FAISS index.
        k (int): The number of top results to retrieve.
        
    Returns:
        list: A list of dictionaries with search results.
    """
    print(f"\n--- Step 4: Performing semantic search for query: '{query}' ---")
    
    try:
        with open(embeddings_path, 'rb') as f:
            embeddings_data = pickle.load(f)
        metadata = embeddings_data['metadata']
    except FileNotFoundError:
        print(f"Error: Embeddings file '{embeddings_path}' not found.")
        return []
    
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    
    query_embedding = model.encode(query)

    try:
        index = faiss.read_index(index_path)
    except RuntimeError:
        print(f"Error: Could not load FAISS index from '{index_path}'.")
        return []

    query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
    
    distances, indices = index.search(query_embedding, k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        result_metadata = metadata[idx]
        result_metadata['score'] = float(distances[0][i])
        results.append(result_metadata)
        
    return results