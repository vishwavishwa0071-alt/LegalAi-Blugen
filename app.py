import streamlit as st
import asyncio
import time
from gemini_backend import GeminiRAGBackend as RAGBackend

# Page Configuration
st.set_page_config(
    page_title="BLUGEN AI - CPC Expert",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Aesthetics
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Inter', sans-serif;
    }
    
    /* Chat Container */
    .stChatMessage {
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .stChatMessage[data-testid="stChatMessageUser"] {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
    }
    
    .stChatMessage[data-testid="stChatMessageAssistant"] {
        background-color: #ffffff;
        border-left: 5px solid #4caf50;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #2c3e50;
        color: white;
    }
    
    section[data-testid="stSidebar"] h1, h2, h3 {
        color: #ecf0f1;
    }
    
    /* Header Styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #1a237e 0%, #0d47a1 100%);
        color: white;
        border-radius: 0 0 20px 20px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .main-header h1 {
        font-weight: 700;
        margin: 0;
        font-size: 2.5rem;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Input Box Styling */
    .stTextInput > div > div > input {
        border-radius: 25px;
        padding: 12px 20px;
        border: 2px solid #e0e0e0;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #2196f3;
        box-shadow: 0 0 0 2px rgba(33, 150, 243, 0.2);
    }
    /* Spinner */
    .stSpinner > div {
        border-top-color: #2196f3 !important;
    }
    
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>⚖️ BLUGEN AI</h1>
    <p>Your Expert Assistant for the Code of Civil Procedure, 1908</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.markdown("---")
    st.info("This AI assistant uses Gemini embeddings and LLM to provide accurate legal answers.")
    
    st.markdown("### 📚 Knowledge Base")
    st.success("Loaded: The Code of Civil Procedure, 1908")
    
    st.markdown("### 🛠️ Tech Stack")
    st.code("LLM: Gemini 2.5 Flash Lite\nEmbeddings: Gemini Embeddings\nVector Store: Numpy (In-Memory)\nFramework: Streamlit", language="yaml")
    
    st.markdown("---")
    if st.button("Clear Chat History", type="primary"):
        st.session_state.messages = []
        st.rerun()

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_backend" not in st.session_state:
    with st.spinner("Initializing Gemini AI Backend... This may take a moment."):
        try:
            st.session_state.rag_backend = RAGBackend()
            st.success("Gemini Backend Initialized Successfully!")
            time.sleep(1)
        except Exception as e:
            st.error(f"Failed to initialize backend: {e}")
            st.warning("Please ensure you have:\n- Set GEMINI_API_KEY in .env file\n- Run 'embedding_comparison.py' to generate chunks")
            st.stop()

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask a legal question (e.g., 'What is the procedure for filing a suit?')..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        status_placeholder.info("🤔 Thinking... Retrieving relevant information using Gemini...")
        
        try:
            # This is a simplified helper to run the async backend method
            async def get_responses():
                return await st.session_state.rag_backend.get_answer(prompt)

            # Run the async function
            combined_response = asyncio.run(get_responses())
            
            # Clear status
            status_placeholder.empty()

            # Display the response
            st.markdown(combined_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": combined_response})
            
        except Exception as e:
            status_placeholder.empty()
            st.error(f"An error occurred: {e}")
            st.warning("Please check:\n- Your GEMINI_API_KEY is valid\n- You have internet connection\n- API quota is not exceeded")

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 3rem; color: #888; font-size: 0.8rem;">
    <p>Disclaimer: This is an AI assistant. Verify all legal information with official sources.</p>
</div>
""", unsafe_allow_html=True)
