
#Add conversational memory, a chat-style UI, and dynamic streaming responses.
import sys
import os
import streamlit as st
import time

# Allow imports from src/
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

# --- Streamlit Page Config ---
st.set_page_config(page_title="TenaAI - Healthcare Chat", page_icon="🩺", layout="wide")

# --- Title and Description ---
st.title("🩺 TenaAI: Healthcare RAG Chat Assistant")
st.write("Ask clinical questions and get AI-generated answers from Ethiopian healthcare guidelines and official documents.")

# --- Initialize RAG System ---
@st.cache_resource
def init_rag(api_key):
    """Load FAISS store and RAG system once."""
    store = FaissVectorStore("faiss_store")
    store.load()
    rag = RAGSearch(api_key=api_key)
    return rag

DEEPSEEK_API_KEY = st.secrets["My_API_Key"]
rag_search = init_rag(DEEPSEEK_API_KEY)

# --- Initialize Chat Memory ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hello! I’m **TenaAI**, your healthcare knowledge assistant. How can I help you today?"}
    ]

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input ---
if prompt := st.chat_input("Ask a healthcare-related question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response with typing effect
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        with st.spinner("Retrieving and summarizing relevant medical information..."):
            try:
                # Get model response
                response = rag_search.search_and_summarize(prompt, top_k=3)
            except Exception as e:
                response = f"⚠️ Sorry, I encountered an error: {e}"

        # Simulate "typing" effect
        for chunk in response.split():
            full_response += chunk + " "
            time.sleep(0.03)
            message_placeholder.markdown(full_response + "▌")  # blinking cursor
        message_placeholder.markdown(full_response)

    # Save message
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- Footer ---
st.markdown("---")
st.caption("TenaAI © 2025 | AI-powered Retrieval-Augmented Generation System for Ethiopian Healthcare Professionals.")
