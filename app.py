import sys
import os
import streamlit as st
import time
import re

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
        {"role": "assistant", "content": "👋 Hello! I'm **TenaAI**, your healthcare knowledge assistant. How can I help you today?"}
    ]

# --- Function to Format LLM Response ---
def format_clinical_response(text: str) -> str:
    """
    Formats medical text with proper structure for natural readability.
    Creates clear sections with headers and organized content.
    """
    # Remove excessive whitespace and normalize
    text = text.strip()
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Replace "Based on..." introduction
    text = re.sub(
        r'Based on the Ethiopian clinical guidelines provided in the context:?\s*',
        '📋 **Based on Ethiopian Clinical Guidelines:**\n\n',
        text,
        flags=re.IGNORECASE
    )
    
    # Split into sentences for better processing
    # Match section patterns like "Word:" or "Word Word:" followed by content
    section_pattern = r'([A-Z][a-zA-Z\s]{1,30}):\s+'
    
    # Split by section headers while keeping them
    parts = re.split(section_pattern, text)
    
    formatted = ""
    
    for i in range(len(parts)):
        part = parts[i].strip()
        
        if not part:
            continue
            
        # If this looks like a section header (short, capitalized)
        if i < len(parts) - 1 and len(part.split()) <= 5 and part[0].isupper() and not part.endswith('.'):
            # This is a header, next part is content
            formatted += f"\n\n**{part}:**\n\n"
        else:
            # This is content
            formatted += part
    
    # Clean up the result
    formatted = formatted.strip()
    
    # Ensure we don't have too many newlines
    formatted = re.sub(r'\n{4,}', '\n\n', formatted)
    
    # Make sure there's spacing after the intro
    formatted = re.sub(
        r'(Based on Ethiopian Clinical Guidelines:\*\*)\s*([A-Z])',
        r'\1\n\n\2',
        formatted
    )
    
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
        
        with st.spinner("🔍 Retrieving and summarizing relevant medical information..."):
            try:
                raw_response = rag_search.search_and_summarize(prompt, top_k=3)
                # Format response for readability
                response = format_clinical_response(raw_response)
            except Exception as e:
                response = f"⚠️ **Error**: Sorry, I encountered an issue while processing your request.\n\n*Details: {str(e)}*"
        
        # Stream with character-level animation for smoother effect
        for char in response:
            full_response += char
            # Update display at reasonable intervals
            if char in [' ', '\n', '.', ':', '!', '?'] and len(full_response) % 10 == 0:
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.01)
        
        # Final render without cursor
        message_placeholder.markdown(full_response)
    
    # Save message
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- Footer ---
st.markdown("---")
st.caption("💡 **TenaAI** © 2025 | AI-powered Retrieval-Augmented Generation System for Ethiopian Healthcare Professionals.")
