from utils import read_from_file
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from preprocessing import clean_text


# ---- Config ----
folder_path = "/Users/trishika/Documents/My Projects/[1] HogRAG/data"

logging.basicConfig(
    level = logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("hograg.log"),
        logging.StreamHandler()]
)

# --------------------------
# Chunk Text from a Single File
# --------------------------
def chunk_text(file_path):
    """
    Reads a text file, splits it into overlapping chunks, and returns a list of cleaned chunks.

    Args:
        file_path (str): The path to the text file to be chunked.

    Returns:
        list of str: Cleaned and chunked text segments from the file.
    """
    try:
        # Read raw text from file
        paragraphs = read_from_file(file_path)

        # If file is empty or unreadable
        if not paragraphs:
                logging.warning(f"Content not found in {file_path}")
                return ""
        
        # Initialize text splitter with chunk size and overlap
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = 100,
            chunk_overlap = 50,
            # separators= ["\n\n", "\n", "  ", " "]
        )

        # Perform the text splitting
        chunks = splitter.split_text(paragraphs)
        # Clean each chunk using custom cleaner
        cleaned_chunks = [clean_text(chunk) for chunk in chunks]
        logging.info(f"Text from {file_path} split into {len(chunks)} chunks")

        # for i, chunk in enumerate(chunks):
        #     print(f"Chunk {i} ({len(chunk)} chars): \n {chunk} \n")
            
        return cleaned_chunks
    
    except Exception as e:
        logging.exception(f"An error occured while chunking the file {file_path}: {e}")
        return []

# --------------------------------
# Chunk All Text Files in a Folder
# --------------------------------
def chunk_folder(folder_path):
    """
    Iterates over all .txt files in the folder, chunks each, and returns a combined list of all chunks.

    Args:
        folder_path (str): Path to the folder containing text files.

    Returns:
        list of str: Combined list of cleaned chunks from all valid .txt files.
    """

    all_chunks = []
    try:
        # List all files in the given folder
        existing_files = os.listdir(folder_path)
        for file_name in existing_files:
            file_path = os.path.join(folder_path, file_name)
            
            # Skip non-txt files
            if not file_name.endswith(".txt"):
                logging.info(f"Skipping non-text file: {file_name}")
                continue

            logging.info(f"Chunking file: {file_path}")
            try:       
               chunks = chunk_text(file_path)
               all_chunks.extend(chunks)
            except Exception as e:
                logging.error(f"Failed to chink file: {file_path}: {e}")
            
        logging.info(f"Finished chunking {len(existing_files)} files. Total chunks: {len(all_chunks)}")
        return all_chunks
    
    except FileNotFoundError:
        logging.error(f"Folder not found: {folder_path}")

    except Exception as e:
        logging.exception(f"Unexpected error while processing folder {folder_path}: {e}")

    return []

if __name__ == "__main__":
    # Testing ~
    #file_path = "/Users/trishika/Documents/My Projects/[1] HogRAG/data/1.txt"
    #chunk_text(file_path)

    chunk_folder(folder_path)


         
    

