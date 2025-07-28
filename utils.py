import logging
import os

# ---- Config ----
logging.basicConfig(
    level = logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("hograg.log"),
        logging.StreamHandler()]
)


# --------------------------
# Read Text Content from File
# --------------------------
def read_from_file(file_path):
    """
    Reads the content of a UTF-8 encoded file.

    Args:
        file_path (str): Path to the file to read.

    Returns:
        str or None: The file's content, or None if the read fails.
    """
    try:
        with open(file_path, "r", encoding = "utf-8") as f:
            file_content = f.read()
        logging.info(f"Successfully read file contents: {file_path}")
        return file_content
    
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
    except Exception as e:
        logging.exception(f"An unexpected error occurred while reading {file_path}: {e}")
    return None


# --------------------------------------
# Save Scraped Content to a Numbered File
# ---------------------------------------
def save_to_file(content):
    """
    Saves content to a numbered text file inside a 'data/' directory.
    Automatically generates a new filename based on the next available index.

    Args:
        content (str): The text content to be saved.

    Returns:
        None
    """
    try:
        os.makedirs('data', exist_ok = True)

        existing_files = os.listdir('data')

        indices = []
        for fname in existing_files:
            if fname.endswith('.txt') and fname[: -4].isdigit():
                indices.append(int(fname[: -4]))

        next_index = max(indices, default = 0) + 1
        file_name = f"data/{next_index}.txt"

        with open(file_name, "w", encoding = 'utf-8') as f:
            f.write(content)
        logging.info(f"Content saved to {file_name}")

    except Exception as e:
        logging.error(f"Failed to save file {file_name}: {e}")
    

