import streamlit as st
import base64
import os
from vector_db import retrieve_context
from llm import build_prompt, load_llm

# -------------------------------------
# Convert Image to Base64 for Background
# --------------------------------------
def get_base64_img(img_path):
    """
    Converts an image to base64 for embedding in HTML/CSS.

    Args:
        img_path (str): Path to the image file.

    Returns:
        str: Base64-encoded data URI for use in CSS.
    """
    with open(img_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return f"data:image/webp;base64,{encoded}"

# --------------------------------
# Set Background Image with Overlay
# ---------------------------------
def display_bg_img(bgimg_path):
    """
    Displays a background image with a white overlay in Streamlit UI.

    Args:
        bgimg_path (str): Path to the image file.

    Returns:
        None
    """

     # Check if image exists
    if not os.path.exists(bgimg_path):
        st.error(f"Background image not found: {bgimg_path}")
    else:
        img_url = get_base64_img(bgimg_path)

        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("{img_url}");
                background-size: cover;
                background-attachment: fixed;
                background-position: center;
            }}
            .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(255, 255, 255, 0.6);  /* White overlay with 60% opacity */
        z-index: 0;
        }}

        .block-container {{
        position: relative;
        z-index: 1;
        }}
            </style>
            """,
            unsafe_allow_html=True
        )

# --------------------------
# Cache the Language Model
# --------------------------
@st.cache_resource  
def get_model():
    """
    Loads and caches the LLM model.

    Returns:
        GPT4All model instance.
    """
    return load_llm()

# ------------------------
# Streamlit UI Application
# ------------------------
def main():
    """
    Main function to run the HogRAG Streamlit app.
    """

    st.set_page_config(page_title="HogRAG", layout="centered")
    st.title("🪄 Welcome to HogRAG")
    st.write("***This is your Harry Potter RAG-powered assistant.***")

    # Display background image
    bgimg_path = "/Users/trishika/Documents/My Projects/[1] HogRAG/ui_imgs/bg_pik.webp"
    display_bg_img(bgimg_path)

    # Initialize session state variables
    if "user_query" not in st.session_state:
        st.session_state.user_query = ""
    if "submitted" not in st.session_state:
        st.session_state.submitted = False

    # User input field
    user_query = st.text_area("Enter your query:", value=st.session_state.user_query, key = "user_query")

    # Create two columns: left for Submit and right for Clear
    col1, _, col3 = st.columns([1, 6, 1])


    with col1:
        if st.button("Submit"):
            st.session_state.submitted = True
    
    def clear():
        st.session_state.user_query = ""
        st.session_state.submitted = False
    
    with col3:
        # Clear Logic: Clear button resets query
        st.button("Clear", on_click = clear)
    

    # Submit Logic: Submit button to trigger processing
    if st.session_state.submitted and st.session_state.user_query.strip():
        
        with st.spinner("Processing your magical question..."):
            context_chunks = retrieve_context(st.session_state.user_query)
            prompt = build_prompt([chunk["text"] for chunk in context_chunks], st.session_state.user_query)
            llm = get_model()
            response = llm.generate(prompt, max_tokens=512, temp=0.7)
                
        st.subheader("📖 Answer:")
        st.write(response)

        # Feedback Buttons
        st.markdown("**Was this answer helpful?**")
        col1, col2, _ = st.columns([1, 1, 6])

        with col1:
            if st.button("👍 Yes"):
                st.success("Thanks for your feedback!")
        
        with col2:
            if st.button("👎 No"):
                st.info("Thanks! We’ll try to improve.")

            
    
# -----------
# Entry Point
# -----------
if __name__ == "__main__":
    main()
    
    
   

    

