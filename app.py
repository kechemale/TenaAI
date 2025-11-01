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

# --- Function to Format LLM Response ---
def format_clinical_response(text: str) -> str:
    """
    Adds bold headers and bullets for better readability.
    Recognizes patterns like 'Cause:', 'Management:' and separates into sections.
    """
    lines = text.split("\n")
    formatted = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Detect sections
        if ":" in line and len(line.split(":")[0].split()) < 5:
            header, rest = line.split(":", 1)
            formatted += f"**{header.strip()}:** {rest.strip()}\n\n"
        else: 
            # Add as bullet point 
            formatted += f"- {line}\n"
    return formatted

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
                raw_response = rag_search.search_and_summarize(prompt, top_k=3)
                # Format response for readability
                response = format_clinical_response(raw_response)
            except Exception as e:
                response = f"⚠️ Sorry, I encountered an error: {e}"

        # Stream line by line for live effect
        for line in response.split("\n"):
            if line.strip() == "":
                continue
            full_response += line + "\n"
            message_placeholder.markdown(full_response + "▌")  # blinking cursor
            time.sleep(0.05)  # adjust typing speed
        message_placeholder.markdown(full_response)

    # Save message
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- Footer ---
st.markdown("---")
st.caption("TenaAI © 2025 | AI-powered Retrieval-Augmented Generation System for Ethiopian Healthcare Professionals.")

