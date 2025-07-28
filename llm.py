from gpt4all import GPT4All
from vector_db import semantic_search, load_embedder, load_faiss_index
from embedding import load_embeddings
import logging


# ---- Config ----
model_path = "/Users/trishika/Documents/My Projects/[1] HogRAG/model/GPT4All-13B-snoozy.ggmlv3.q4_0.bin"
folder_path = "/Users/trishika/Documents/My Projects/[1] HogRAG/embeddings"

logging.basicConfig(
    level = logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("hograg.log"),
        logging.StreamHandler()]
)


# --------------------------
# Build Prompt from Context
# --------------------------
def build_prompt(context_chunks, user_query):
    """
    Builds a prompt for the LLM using retrieved context and user query.

    Args:
        context_chunks (list of str): List of context passages retrieved based on semantic search.
        user_query (str): The question entered by the user.

    Returns:
        str: A formatted prompt string to be passed to the language model.
    """

    try:
        # Validate that context_chunks is a list
        if not isinstance(context_chunks, list):
            raise TypeError("Context chunks must be a list.")
        
        # Ensure all elements in context_chunks are strings
        if not all(isinstance(chunk, str) for chunk in context_chunks):
            raise ValueError("Each context chunk must be a string.")
        
        # Ensure all elements in context_chunks are strings
        if not isinstance(user_query, str):
            raise TypeError("user_query must be a string.")
        
        # Provide default messages if inputs are empty
        if not context_chunks:
            context_chunks = ["No context available."]
        if not user_query:
            user_query = "No question provided."
        
        # Join context chunks into a single block of text
        context = "\n\n".join(context_chunks)

        # Construct the full prompt with context and user question
        prompt = f""""
        You are a helpful assistant. Use the following context to answer the user's question.

        Context:
        {context}

        User Question:
        {user_query}

        Answer:
        """

        return prompt
    
    except Exception as e:
        # Handle any unexpected errors and return a fallback message
        logging.error(f"Failed to build prompt: {e}")
        return "You are a helpful assistant, but the prompt could not be generated."

# --------------------------
# Load GPT4All Language Model
# --------------------------
def load_llm():
    """
    Loads the GPT4All language model from a specified local path.

    Returns:
        GPT4All: An instance of the GPT4All model ready for inference.
    """
    try:
        # Initialize and return the GPT4All model
        return GPT4All(model_path)
    except Exception as e:
        # Log any loading error and re-raise
        logging.error("Failed to load GPT4All model: {e}")
        raise




if __name__ == "__main__":
    # Testing ~
    user_query = "Describe Diagon Alley."

    # STEP-1: Retrieve top chunks based on query
    embeddings, chunks = load_embeddings(folder_path)
    index = load_faiss_index()
    model = load_embedder()
    context_chunks = semantic_search(user_query, model, index, chunks)

    # STEP-2: Load LLM
    llm = load_llm()

     # STEP-3: Build the full prompt
    prompt = build_prompt([r["text"]for r in context_chunks], user_query)

    # STEP-4: Generate answer
    response = llm.generate(prompt, max_tokens = 512, temp = 0.7)
    print("\n--- Answer ---\n")
    print(response)

    # Testting ~
    # llm = GPT4All(model_path)
    # output = llm.generate("Who is Harry Potter?", max_tokens=50)
    # print(output)

