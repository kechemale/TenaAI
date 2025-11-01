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
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text.strip())
    
    # Remove leading bullets that might already exist
    text = re.sub(r'^\s*[•\-\*]\s+', '', text, flags=re.MULTILINE)
    
    lines = text.split("\n")
    formatted = ""
    current_section = ""
    
    for line in lines:
        line = line.strip()
        
        if not line:
            if formatted and not formatted.endswith("\n\n"):
                formatted += "\n\n"
            continue
        
        # Detect section headers with colons (e.g., "Cause:", "Management:", "Symptoms:")
        if re.match(r'^[A-Z][a-zA-Z\s]{0,30}:\s*.+', line):
            parts = line.split(":", 1)
            header = parts[0].strip()
            content = parts[1].strip() if len(parts) > 1 else ""
            
            # Add proper spacing before new section
            if formatted:
                formatted += "\n\n"
            
            # Format as a bold header with content on same line
            formatted += f"**{header}:**\n\n{content}"
            current_section = header
            
        # Detect numbered lists (1., 2., etc.)
        elif re.match(r'^\d+[\.)]\s+', line):
            formatted += f"\n{line}"
            
        # Detect existing bullet points
        elif re.match(r'^[•\-\*]\s+', line):
            formatted += f"\n{line}"
            
        # Regular text - append naturally
        else:
            # If we just started a new section, continue on same area
            if formatted.endswith(current_section):
                formatted += f" {line}"
            # If previous line ended with period, question mark, or exclamation, start new line
            elif formatted and formatted.rstrip().endswith(('.', '?', '!', ':')):
                formatted += f"\n\n{line}"
            # Otherwise continue on same line with space
            elif formatted:
                formatted += f" {line}"
            else:
                formatted += line
    
    # Clean up formatting
    formatted = formatted.strip()
    
    # Fix multiple spaces
    formatted = re.sub(r' {2,}', ' ', formatted)
    
    # Ensure consistent spacing (max 2 newlines between sections)
    formatted = re.sub(r'\n{3,}', '\n\n', formatted)
    
    # Add a summary introduction if response starts with "Based on"
    if formatted.startswith("Based on"):
        formatted = formatted.replace("Based on the Ethiopian clinical guidelines provided in the context:", 
                                      "📋 **Based on Ethiopian Clinical Guidelines:**\n\n", 1)
    
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
            # Update display periodically
            if i % 3 == 0 or word.endswith(('\n', '.')):
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.02)
        
        # Final render without cursor
        message_placeholder.markdown(full_response.strip())
    
    # Save message
    st.session_state.messages.append({"role": "assistant", "content": full_response.strip()})

# --- Footer ---
st.markdown("---")
st.caption("💡 **TenaAI** © 2025 | AI-powered Retrieval-Augmented Generation System for Ethiopian Healthcare Professionals.")
```
