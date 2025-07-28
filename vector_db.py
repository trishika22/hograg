from embedding import load_embeddings, load_embedder
import faiss
import logging
import os


# ---- Config ----
faiss.omp_set_num_threads(1)
folder_path = "/Users/trishika/Documents/My Projects/[1] HogRAG/embeddings"

logging.basicConfig(
    level = logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("hograg.log"),
        logging.StreamHandler()]
)


# --------------------------
# Build and Save FAISS Index
# --------------------------
def build_faiss_index(embeddings, normalize = True, save_path = "faiss/faiss_index.index"):
    """
    Builds a FAISS index from the given embeddings, optionally normalizes them for cosine similarity,
    and saves the index to disk.

    Args:
        embeddings (np.ndarray): A 2D numpy array of embeddings to index.
        normalize (bool): Whether to normalize embeddings for cosine similarity. Defaults to True.
        save_path (str): Path where the FAISS index will be saved.

    Returns:
        faiss.Index: The built FAISS index.

    Raises:
        Exception: Logs and raises any exception encountered during index building.
    """
    try:
        # Normalize embeddings if specified (important for cosine similarity)
        if normalize:
            faiss.normalize_L2(embeddings)
            logging.info("Embeddings normalized for cosine similarity")
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Create a flat (brute-force) L2 distance index with dimensionality of embeddings
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)

        # Add embeddings to index in batches to handle large datasets efficiently
        batch_size = 1000
        for i in range(0, len(embeddings), batch_size):
            index.add(embeddings[i:i+batch_size])
        
        # Save the index to disk
        faiss.write_index(index, save_path)
        logging.info(f"FAISS index built and save to: {save_path}")

        return index
    
    except Exception as e:
        logging.error(f"Failed to build FAISS index: {e}", exc_info = True)
        raise  # Re-raise so caller knows something went wrong


# --------------------------
# Load FAISS Index from File
# --------------------------
def load_faiss_index(index_path = "faiss/faiss_index.index"):
    """
    Loads a FAISS index from a file on disk.

    Args:
        index_path (str): Path to the saved FAISS index file.

    Returns:
        faiss.Index: The loaded FAISS index.

    Raises:
        Exception: Logs and raises any exception encountered during index loading.
    """
    try:
        # Attempt to read the FAISS index file
        index = faiss.read_index(index_path)
        logging.info(f"FAISS index loaded from {index_path}")
        return index
    except Exception as e:
        logging.info(f"Could not load FAISS index: {e}", exc_info = True)
        raise  # Re-raise so caller can handle missing or corrupted index
        

# --------------------------
# Semantic Search Function
# --------------------------
def semantic_search(query, model, index, chunks, top_k = 5):
    """
    Performs a semantic similarity search to find top-k relevant chunks for a query.

    Args:
        query (str): The query string to search for.
        embedder (object): The embedding model used to vectorize the query.
        index (faiss.Index): The FAISS index containing vectorized document chunks.
        chunks (list): List of all document chunks (each should have text data).
        top_k (int): Number of top results to return.

    Returns:
        list of dict: Top-k context chunks most relevant to the query.
    """
    try:
        logging.info(f"Running semantic search for: '{query}'")

        logging.info(f"Index type: {type(index)}, Model: {type(model)}, Chunks: {len(chunks)}")

        # Embed the query into vector space
        query_vec = model.encode([query], normalize_embeddings = True).astype("float32")
        logging.info(f"Query embedding shape: {query_vec.shape}")
        logging.info(f"Index dimension: {index.d}")

        # Search the index for nearest neighbors to the query vector
        D, I = index.search(query_vec, top_k)

        # Collect corresponding chunks
        results = []
        for rank, idx in enumerate(I[0]):
            results.append({
                "rank": rank + 1,
                "score": float(D[0][rank]),
                "text": chunks[idx]
            })
        logging.info(f"Search for query '{query}' returned {len(results)} results")
        return results
    except Exception as e:
        logging.error(f"Semantic search failed: {e}")
        return []

# --------------------------
# Retrieve Semantic Context
# --------------------------
def retrieve_context(user_query):
    """
    Retrieves relevant context chunks for the given user query by performing semantic search.

    Args:
        user_query (str): The input question/query from the user.

    Returns:
        list of dict: List of relevant context chunks (dictionaries with at least a 'text' key).

    Raises:
        ValueError: If the user_query is empty or None.
    """
    try:
        # Validate user query input
        if not user_query or not user_query.strip():
            raise ValueError("User query must be a non-empty string.")
        
        logging.info("Loading embeddings and chunks from folder")
        embeddings, chunks = load_embeddings(folder_path)

        logging.info("Loading FAISS index for semantic search.")
        index = load_faiss_index()

        logging.info("Loading medding model.")
        embedder = load_embedder()

        logging.info("Performing semantic search for the user query.")
        context_chunks = semantic_search(user_query, embedder, index, chunks)

        logging.info(f"Retrieved {len(context_chunks)} relevant context chunks.")
        
        return context_chunks
    
    except ValueError as ve:
        logging.error(f"Invalid input error in retrieve_context: {ve}")
        raise

if __name__ == "__main__":
    # Testing ~
    # STEP-1: Load data
    embeddings, chunks = load_embeddings(folder_path)

    # STEP-2: Build index (first time only)
    index = build_faiss_index(embeddings)

    # Or: Load existing index
    # index = load_faiss_index()

    # STEP-3: Load embedding model
    model = load_embedder()

    # STEP-4: Query
    query = "What do we know about Harry Potter from Chapter 1?"
    results = semantic_search(query, model, index, chunks)

    for result in results:
        print(f"\nRank #{result['rank']} | Score: {result['score']:.4f}")
        print(result['text'][:300])
        print("-" * 60)
