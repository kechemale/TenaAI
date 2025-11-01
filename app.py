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
    Formats medical text with proper structure, headers, and bullet points.
    Handles sections, numbered lists, and improves overall readability.
    """
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text.strip())
    
    lines = text.split("\n")
    formatted = ""
    in_list = False
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Skip empty lines but preserve paragraph breaks
        if not line:
            if formatted and not formatted.endswith("\n\n"):
                formatted += "\n"
            in_list = False
            continue
        
        # Detect section headers (e.g., "Diagnosis:", "Treatment:", "Symptoms:")
        if re.match(r'^[A-Z][a-zA-Z\s]{0,30}:', line):
            parts = line.split(":", 1)
            header = parts[0].strip()
            content = parts[1].strip() if len(parts) > 1 else ""
            
            # Add spacing before new section
            if formatted and not formatted.endswith("\n\n"):
                formatted += "\n\n"
            
            formatted += f"### {header}\n"
            if content:
                formatted += f"{content}\n"
            in_list = False
            
        # Detect numbered lists (1., 2., etc.)
        elif re.match(r'^\d+\.', line):
            formatted += f"{line}\n"
            in_list = True
            
        # Detect bullet points (-, *, •)
        elif re.match(r'^[-*•]\s', line):
            formatted += f"{line}\n"
            in_list = True
            
        # Regular paragraph text
        else:
            # If previous content was a list, add spacing
            if in_list and formatted:
                formatted += "\n"
                in_list = False
            
            # Check if this looks like a continuation of previous line
            if formatted and not formatted.endswith("\n\n") and not formatted.endswith(":\n"):
                # Don't add bullet if it's a continuation
                formatted += f"{line}\n"
            else:
                # Add as bullet point for standalone statements
                if not line.endswith(":"):
                    formatted += f"• {line}\n"
                else:
                    formatted += f"{line}\n"
    
    # Clean up any trailing whitespace
    formatted = formatted.strip()
    
    # Ensure proper spacing between sections
    formatted = re.sub(r'\n{3,}', '\n\n', formatted)
    
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
        
        # Stream word by word for smoother live effect
        words = response.split()
        for i, word in enumerate(words):
            full_response += word + " "
            # Update more frequently at paragraph breaks
            if word.endswith(("\n", ".", ":", "?", "!")) or i % 5 == 0:
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.03)
        
        # Final render without cursor
        message_placeholder.markdown(full_response.strip())
    
    # Save message
    st.session_state.messages.append({"role": "assistant", "content": full_response.strip()})

# --- Footer ---
st.markdown("---")
st.caption("💡 **TenaAI** © 2025 | AI-powered Retrieval-Augmented Generation System for Ethiopian Healthcare Professionals.")
