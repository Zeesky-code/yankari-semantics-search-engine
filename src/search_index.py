import numpy as np
import faiss
import pickle
import os

def create_index(embeddings_path, index_path):
    """
    Loads embeddings and builds a FAISS index.
    
    Args:
        embeddings_path (str): Path to the pickled embeddings file.
        index_path (str): Path to save the FAISS index.
    """
    print("--- Step 3: Building FAISS index ---")
    
    # Load the embeddings and metadata
    try:
        with open(embeddings_path, 'rb') as f:
            embeddings_data = pickle.load(f)
        embeddings = np.array(embeddings_data['embeddings']).astype('float32')
        print(f"Embeddings loaded successfully. Shape: {embeddings.shape}")
    except FileNotFoundError:
        print(f"Error: Embeddings file '{embeddings_path}' not found.")
        return
    
    d = embeddings.shape[1]
    
    # Choose an appropriate FAISS index
    # For a small to medium-sized dataset, an IndexFlatL2 is fast and simple.
    index = faiss.IndexFlatL2(d)
    
    # Add the embeddings to the index
    index.add(embeddings)
    
    # Save the index to disk
    faiss.write_index(index, index_path)
    print(f"FAISS index built and saved to '{index_path}'.")


def search_index(query_embedding, index_path, k=5):
    """
    Searches the FAISS index for the top k most similar vectors.
    
    Args:
        query_embedding (np.array): The embedding of the search query.
        index_path (str): Path to the saved FAISS index.
        k (int): The number of top results to retrieve.
        
    Returns:
        tuple: A tuple containing distances and indices of the top k results.
    """
    # Load the FAISS index
    try:
        index = faiss.read_index(index_path)
    except RuntimeError:
        print(f"Error: Could not load FAISS index from '{index_path}'.")
        return None, None

    # Reshape the query for FAISS search
    query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
    
    # Perform the search
    distances, indices = index.search(query_embedding, k)
    
    return distances, indices

def semantic_search(query, cohere_api_key, embeddings_path, index_path, k=5):
    """
    Performs a semantic search on the FAISS index.
    
    Args:
        query (str): The search query.
        cohere_api_key (str): Your Cohere API key.
        embeddings_path (str): Path to the pickled embeddings file.
        index_path (str): Path to the saved FAISS index.
        k (int): The number of top results to retrieve.
        
    Returns:
        list: A list of dictionaries with search results.
    """
    print(f"\n--- Step 4: Performing semantic search for query: '{query}' ---")
    
    # Load the embeddings metadata to get the article details
    try:
        with open(embeddings_path, 'rb') as f:
            embeddings_data = pickle.load(f)
        metadata = embeddings_data['metadata']
    except FileNotFoundError:
        print(f"Error: Embeddings file '{embeddings_path}' not found.")
        return []
    
    # Initialize Cohere client
    co = cohere.Client(cohere_api_key)
    
    # Get embedding for the query
    try:
        response = co.embed(
            model='embed-multilingual-v3.0',
            texts=[query],
            input_type="search_query"
        )
        query_embedding = response.embeddings[0]
    except Exception as e:
        print(f"Error embedding query: {e}")
        return []

    # Load the FAISS index
    try:
        index = faiss.read_index(index_path)
    except RuntimeError:
        print(f"Error: Could not load FAISS index from '{index_path}'.")
        return []

    # Reshape the query for FAISS search
    query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
    
    # Perform the search
    distances, indices = index.search(query_embedding, k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        result_metadata = metadata[idx]
        result_metadata['score'] = float(distances[0][i])
        results.append(result_metadata)
        
    return results