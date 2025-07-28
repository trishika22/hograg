from sentence_transformers import SentenceTransformer
import os
import logging
import numpy as np
import json
from chunk_utils import chunk_folder

# ---- Config ----
folder_path = "Users/trishika/Documents/My Projects/[1] HogRAG/data"
model = SentenceTransformer("all-mpnet-base-v2")

logging.basicConfig(
    level = logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("hograg.log"),
        logging.StreamHandler()]
)


# -----------------------------------
# Generate Embeddings for Text Chunks
# -----------------------------------
def embed_chunks(chunks):
    """
    Embeds a list of text chunks using the global embedding model.

    Args:
        chunks (list of str): The text chunks to embed.

    Returns:
        np.ndarray or None: Array of vector embeddings, or None if embedding fails.
    """
    try:
        logging.info(f"Starting embedding for {len(chunks)} chunks")

        # Prefix format for instruction-tuned models like MPNet
        formatted_chunks = [f"passage: {chunk}" for chunk in chunks]

        # Perform embedding 
        embeddings = model.encode(formatted_chunks, normalize_embeddings = True)

        logging.info("Successfully generated embeddings")
        return embeddings
    except Exception as e:
        logging.error(f"Error during embedding chunks: {e}")
        return None


# ------------------------------------
# Save Embeddings and Metadata to Disk
# -------------------------------------
def save_embeddings(embeddings, chunks, output_folder = "embeddings"):
    """
    Saves the embeddings and corresponding metadata (text chunks) to disk.

    Args:
        embeddings (np.ndarray): The embedding matrix.
        chunks (list of str): The original text chunks.
        output_folder (str): Directory to save embedding files.

    Returns:
        None
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_folder, exist_ok = True)

        # Save embeddings as a .npy file
        np.save(f"{output_folder}/embedding.npy", embeddings)
        logging.info(f"Saved embeddings to {output_folder}/embeddings.py")

        # Save text chunks (metadata) as JSON
        with open(f"{output_folder}/metadata.json", "w", encoding = "utf-8") as f:
            json.dump(chunks, f, ensure_ascii = False, indent = 2)
        logging.info(f"Saved {len(chunks)} embeddings and metadata succcessfully.")
    
    except Exception as e:
        logging.error(f"Error saving embeddings or metadata: {e}")


# --------------------------------------
# Load Embeddings and Metadata from Disk
# --------------------------------------
def load_embeddings(folder_path):
    """
    Loads the saved embeddings and their corresponding text chunks from disk.

    Args:
        folder_path (str): Path to the folder containing the embedding files.

    Returns:
        tuple: A tuple (embeddings, chunks) where:
            - embeddings (np.ndarray)
            - chunks (list of str)

    Raises:
        ValueError: If the number of embeddings and metadata entries don't match.
    """
    try:
        # Load embedding vectors
        embedding_file_path = folder_path + "/embedding.npy"
        loaded_embeddings = np.load(embedding_file_path)

        # Load corresponding chunk metadata
        metadata_file_path = folder_path + "/metadata.json"
        with open(metadata_file_path, "r", encoding = "utf-8") as f:
            loaded_chunks = json.load(f)

        # Sanity check: match length
        if len(loaded_embeddings) != len(loaded_chunks):
            raise ValueError("Mismtach between the number of embeddings and chunks")
        
        logging.info(f"Loaded {len(loaded_embeddings)} embbeddings and {len(loaded_chunks)} chunks")
        return loaded_embeddings, loaded_chunks
    
    except Exception as e:
        logging.error(f"Error loading data: {e}")

# --------------------------
# Initialize Embedding Model
# --------------------------
def load_embedder(mode_name = "all-mpnet-base-v2"):
    """
    Loads a SentenceTransformer model for generating embeddings.

    Args:
        model_name (str): The name of the SentenceTransformer model.

    Returns:
        SentenceTransformer: Loaded embedding model instance.
    """
    try:
        model = SentenceTransformer(mode_name)
        logging.info(f"Embedding model {mode_name} loaded")
        return model
    except Exception as e:
        logging.error(f"Error logging embedding modelL {e}")


if __name__ == "__main__":
    all_chunks = chunk_folder(folder_path)
    embeddings = embed_chunks(all_chunks)
    save_embeddings(embeddings, all_chunks)
