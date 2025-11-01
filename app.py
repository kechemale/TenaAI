
# #Add conversational memory, a chat-style UI, and dynamic streaming responses.
# import sys
# import os
# import streamlit as st
# import time

# # Allow imports from src/
# sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# from src.vectorstore import FaissVectorStore
# from src.search import RAGSearch

# # --- Streamlit Page Config ---
# st.set_page_config(page_title="TenaAI - Healthcare Chat", page_icon="🩺", layout="wide")

# # --- Title and Description ---
# st.title("🩺 TenaAI: Healthcare RAG Chat Assistant")
# st.write("Ask clinical questions and get AI-generated answers from Ethiopian healthcare guidelines and official documents.")

# # --- Initialize RAG System ---
# @st.cache_resource
# def init_rag(api_key):
#     """Load FAISS store and RAG system once."""
#     store = FaissVectorStore("faiss_store")
#     store.load()
#     rag = RAGSearch(api_key=api_key)
#     return rag

# DEEPSEEK_API_KEY = st.secrets["My_API_Key"]
# rag_search = init_rag(DEEPSEEK_API_KEY)

# # --- Initialize Chat Memory ---
# if "messages" not in st.session_state:
#     st.session_state.messages = [
#         {"role": "assistant", "content": "👋 Hello! I’m **TenaAI**, your healthcare knowledge assistant. How can I help you today?"}
#     ]

# # --- Display Chat History ---
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # --- User Input ---
# if prompt := st.chat_input("Ask a healthcare-related question..."):
#     # Add user message
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # Generate assistant response with typing effect
#     with st.chat_message("assistant"):
#         message_placeholder = st.empty()
#         full_response = ""

#         with st.spinner("Retrieving and summarizing relevant medical information..."):
#             try:
#                 # Get model response
#                 response = rag_search.search_and_summarize(prompt, top_k=3)
#             except Exception as e:
#                 response = f"⚠️ Sorry, I encountered an error: {e}"

#         # Simulate "typing" effect
#         for chunk in response.split():
#             full_response += chunk + " "
#             time.sleep(0.03)
#             message_placeholder.markdown(full_response + "▌")  # blinking cursor
#         message_placeholder.markdown(full_response)

#     # Save message
#     st.session_state.messages.append({"role": "assistant", "content": full_response})

# # --- Footer ---
# st.markdown("---")
# st.caption("TenaAI © 2025 | AI-powered Retrieval-Augmented Generation System for Ethiopian Healthcare Professionals.")

import sys
import os
import streamlit as st
import time
import pandas as pd
from datetime import datetime
import uuid

# Allow imports from src/
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

# --- Query Logger Class ---
class QueryLogger:
    def __init__(self, log_file="user_queries.csv"):
        self.log_file = log_file
        self.setup_log_file()
    
    def setup_log_file(self):
        """Initialize the log file with headers if it doesn't exist"""
        if not os.path.exists(self.log_file):
            pd.DataFrame(columns=[
                'timestamp', 'session_id', 'query', 'response_preview', 
                'response_length', 'sources_count', 'response_time_seconds'
            ]).to_csv(self.log_file, index=False)
    
    def log_query(self, query, response, response_time, session_id):
        """Log a single query to CSV"""
        try:
            new_entry = pd.DataFrame([{
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'session_id': session_id,
                'query': query,
                'response_preview': response[:150] + "..." if len(response) > 150 else response,
                'response_length': len(response),
                'sources_count': 3,  # Since you're using top_k=3
                'response_time_seconds': round(response_time, 2)
            }])
            
            # Append to CSV without loading the entire file
            new_entry.to_csv(self.log_file, mode='a', header=False, index=False)
        except Exception as e:
            print(f"Error logging query: {e}")

# Initialize query logger
query_logger = QueryLogger()

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

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hello! I'm **TenaAI**, your healthcare knowledge assistant. How can I help you today?"}
    ]

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

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

        start_time = time.time()
        
        with st.spinner("Retrieving and summarizing relevant medical information..."):
            try:
                # Get model response
                response = rag_search.search_and_summarize(prompt, top_k=3)
            except Exception as e:
                response = f"⚠️ Sorry, I encountered an error: {e}"
        
        response_time = time.time() - start_time

        # Simulate "typing" effect
        for chunk in response.split():
            full_response += chunk + " "
            time.sleep(0.03)
            message_placeholder.markdown(full_response + "▌")  # blinking cursor
        message_placeholder.markdown(full_response)

    # Log the query to CSV
    query_logger.log_query(
        query=prompt,
        response=full_response,
        response_time=response_time,
        session_id=st.session_state.session_id
    )

    # Save message to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- Analytics Sidebar ---
# with st.sidebar:
#     st.subheader("📊 Usage Analytics")
    
#     try:
#         if os.path.exists("user_queries.csv"):
#             df = pd.read_csv("user_queries.csv")
            
#             if not df.empty:
#                 st.metric("Total Queries", len(df))
#                 st.metric("Your Session ID", st.session_state.session_id)
                
#                 # Show recent queries from current session
#                 session_queries = df[df['session_id'] == st.session_state.session_id]
#                 if not session_queries.empty:
#                     st.write("**Your recent queries:**")
#                     for _, row in session_queries.tail(3).iterrows():
#                         st.caption(f"• {row['query'][:40]}...")
                
#                 # Download button for admin (optional)
#                 if st.checkbox("Show raw data"):
#                     st.dataframe(df.tail(10))
                    
#                 with st.expander("Download Query Logs"):
#                     st.download_button(
#                         label="Download CSV",
#                         data=df.to_csv(index=False),
#                         file_name=f"tenaai_queries_{datetime.now().strftime('%Y%m%d')}.csv",
#                         mime="text/csv"
#                     )
#     except Exception as e:
#         st.info("No query data yet. Start asking questions!")

# --- Footer ---
st.markdown("---")
st.caption("TenaAI © 2025 | AI-powered Retrieval-Augmented Generation System for Ethiopian Healthcare Professionals.")



