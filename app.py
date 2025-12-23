# import sys
# import os
# import streamlit as st
# import time
# import re

# # Allow imports from src/
# sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
# from src.vectorstore import FaissVectorStore
# from src.search import RAGSearch

# # --- Streamlit Page Config ---
# st.set_page_config(page_title="TenaAI - Healthcare Clinician Assistant", page_icon="🩺", layout="wide")

# # --- Title and Description ---
# st.title("🩺 TenaAI: Clinician Assistant")
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
#         {"role": "assistant", "content": "👋 Hello! I'm **TenaAI**, your healthcare knowledge assistant. How can I help you today?"}
#     ]

# # --- Function to Format LLM Response ---
# def format_clinical_response(text: str) -> str:
#     """
#     Formats medical text with proper structure for natural readability.
#     Creates clear sections with headers and organized content.
#     """
#     # Remove excessive whitespace and normalize
#     text = text.strip()
#     text = re.sub(r'\n{3,}', '\n\n', text)
    
#     # Replace "Based on..." introduction
#     text = re.sub(
#         r'Based on the Ethiopian clinical guidelines provided in the context:?\s*',
#         '📋 **Based on Ethiopian Clinical Guidelines:**\n\n',
#         text,
#         flags=re.IGNORECASE
#     )
    
#     # Split into sentences for better processing
#     # Match section patterns like "Word:" or "Word Word:" followed by content
#     section_pattern = r'([A-Z][a-zA-Z\s]{1,30}):\s+'
    
#     # Split by section headers while keeping them
#     parts = re.split(section_pattern, text)
    
#     formatted = ""
    
#     for i in range(len(parts)):
#         part = parts[i].strip()
        
#         if not part:
#             continue
            
#         # If this looks like a section header (short, capitalized)
#         if i < len(parts) - 1 and len(part.split()) <= 5 and part[0].isupper() and not part.endswith('.'):
#             # This is a header, next part is content
#             formatted += f"\n\n**{part}:**\n\n"
#         else:
#             # This is content
#             formatted += part
    
#     # Clean up the result
#     formatted = formatted.strip()
    
#     # Ensure we don't have too many newlines
#     formatted = re.sub(r'\n{4,}', '\n\n', formatted)
    
#     # Make sure there's spacing after the intro
#     formatted = re.sub(
#         r'(Based on Ethiopian Clinical Guidelines:\*\*)\s*([A-Z])',
#         r'\1\n\n\2',
#         formatted
#     )
    
#     return formatted

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
        
#         with st.spinner("🔍 Retrieving and summarizing relevant medical information..."):
#             try:
#                 raw_response = rag_search.search_and_summarize(prompt, top_k=3)
#                 # Format response for readability
#                 response = format_clinical_response(raw_response)
#             except Exception as e:
#                 response = f"⚠️ **Error**: Sorry, I encountered an issue while processing your request.\n\n*Details: {str(e)}*"
        
#         # Stream with character-level animation for smoother effect
#         for char in response:
#             full_response += char
#             # Update display at reasonable intervals
#             if char in [' ', '\n', '.', ':', '!', '?'] and len(full_response) % 10 == 0:
#                 message_placeholder.markdown(full_response + "▌")
#                 time.sleep(0.01)
        
#         # Final render without cursor
#         message_placeholder.markdown(full_response)
    
#     # Save message
#     st.session_state.messages.append({"role": "assistant", "content": full_response})

# # --- Footer ---
# st.markdown("---")
# st.caption("💡 **TenaAI** © 2025 | AI-powered System for Ethiopian Healthcare Professionals.")


import sys
import os
import streamlit as st
import time
import re
import traceback
import json
import uuid
from datetime import datetime, timezone
from typing import Optional

# Allow imports from src/
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

# --- Streamlit Page Config ---
st.set_page_config(page_title="TenaAI - Healthcare Clinician Assistant", page_icon="🩺", layout="wide")

# --- Title and Description ---
st.title("🩺 TenaAI: Clinician Assistant")
st.write("Ask clinical questions and get AI-generated answers from Ethiopian healthcare guidelines and official documents.")

# --- Initialize RAG System ---
@st.cache_resource
def init_rag(api_key):
    """Load FAISS store and RAG system once."""
    store = FaissVectorStore("faiss_store")
    store.load()
    # Force the cheaper DeepSeek chat model (avoid accidentally using deepseek-reasoner).
    rag = RAGSearch(api_key=api_key, llm_model="deepseek-chat")
    return rag

DEEPSEEK_API_KEY = st.secrets["My_API_Key"]
rag_search = init_rag(DEEPSEEK_API_KEY)


_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)


def _ensure_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
    return st.session_state.session_id


def _log_path() -> str:
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return os.path.join(_LOG_DIR, f"rag_chat_{day}.jsonl")


def _read_log_text(*, session_id: Optional[str] = None) -> str:
    path = _log_path()
    if not os.path.exists(path):
        return ""
    try:
        if not session_id:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()

        # Filter to only this Streamlit session's entries.
        lines: list[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if session_id in line:
                    lines.append(line)
        return "".join(lines)
    except Exception:
        return ""


def _json_safe(value):
    # Convert common non-serializable values (numpy scalars, bytes, etc.)
    # without adding a hard dependency on numpy.
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return repr(value)
    # numpy scalar types usually have `.item()`
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _append_jsonl(record: dict) -> None:
    record = dict(record)
    record.setdefault("ts", datetime.now(timezone.utc).isoformat())
    record.setdefault("session_id", _ensure_session_id())
    try:
        with open(_log_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(_json_safe(record), ensure_ascii=False) + "\n")
    except Exception:
        # Logging must never break the user experience.
        print("[WARN] Failed to write JSONL log")
        print(traceback.format_exc())


# --- Manual log download (kept out of git via .gitignore) ---
with st.sidebar:
    st.markdown("### Logs")
    sid = _ensure_session_id()

    session_log_text = _read_log_text(session_id=sid)
    if session_log_text:
        st.download_button(
            label="Download my session log (today)",
            data=session_log_text.encode("utf-8"),
            file_name=f"rag_chat_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}_session_{sid}.jsonl",
            mime="application/jsonl",
        )
    else:
        st.caption("No session logs yet for today.")

    full_log_text = _read_log_text(session_id=None)
    if full_log_text:
        st.download_button(
            label="Download full log (today)",
            data=full_log_text.encode("utf-8"),
            file_name=f"rag_chat_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl",
            mime="application/jsonl",
        )


def _friendly_rag_failure_message() -> str:
    return (
        "⚠️ I can’t generate an answer right now because the AI service is unavailable. "
        "Please try again later, or ask the administrator."
    )


def _looks_like_provider_or_billing_error(text: str) -> bool:
    t = (text or "").lower()
    # Common patterns from OpenAI-compatible SDKs / DeepSeek responses
    return any(
        needle in t
        for needle in [
            "insufficient balance",
            "invalid_request_error",
            "status code: 402",
            "'code': 'invalid_request_error'",
            "\"code\": \"invalid_request_error\"",
            "error generating summary:",
            "rate limit",
            "quota",
            "payment",
            "billing",
            "402",
        ]
    )


def _sanitize_rag_output(text: Optional[str]) -> str:
    if not text:
        return _friendly_rag_failure_message()
    if _looks_like_provider_or_billing_error(text):
        return _friendly_rag_failure_message()
    return text


def _candidates_for_log(results) -> list:
    candidates = []
    for r in results or []:
        meta = r.get("metadata") if isinstance(r, dict) else None
        text = ""
        if isinstance(meta, dict):
            text = meta.get("text", "") or ""
        candidates.append(
            {
                "index": r.get("index") if isinstance(r, dict) else None,
                "distance": r.get("distance") if isinstance(r, dict) else None,
                "text": text[:4000],
                "text_length": len(text),
            }
        )
    return candidates

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
                top_k = 3
                results = rag_search.vectorstore.query(prompt, top_k=top_k)
                debug_fn = getattr(rag_search, "search_and_summarize_with_debug", None)
                if callable(debug_fn):
                    payload = debug_fn(prompt, top_k=top_k)
                    raw_answer = payload.get("answer")
                    # Prefer the exact results used by the RAG if provided.
                    results = payload.get("results") or results
                    # Optional: cost/debug metadata from the LLM call (if available).
                    llm_model_used = payload.get("llm_model") or getattr(rag_search, "last_model", None)
                    llm_usage = payload.get("llm_usage") or getattr(rag_search, "last_usage", None)
                else:
                    raw_answer = rag_search.search_and_summarize(prompt, top_k=top_k)
                    llm_model_used = getattr(rag_search, "last_model", None)
                    llm_usage = getattr(rag_search, "last_usage", None)

                safe_answer = _sanitize_rag_output(raw_answer)
                response = format_clinical_response(safe_answer)

                _append_jsonl(
                    {
                        "event": "chat_turn",
                        "prompt": prompt,
                        "top_k": top_k,
                        "candidates": _candidates_for_log(results),
                        "answer_raw": raw_answer,
                        "answer_user": response,
                        "sanitized": safe_answer != (raw_answer or ""),
                        "llm_model": llm_model_used,
                        "llm_usage": _json_safe(llm_usage),
                    }
                )
            except Exception as e:
                # Do not leak provider errors (e.g., 402 Insufficient Balance) to end users.
                # Keep full details in server logs for debugging.
                print("[ERROR] RAG failure in app.py")
                err_tb = traceback.format_exc()
                print(err_tb)
                _append_jsonl(
                    {
                        "event": "chat_turn_error",
                        "prompt": prompt,
                        "top_k": 3,
                        "candidates": _candidates_for_log(locals().get("results")),
                        "error": str(e),
                        "traceback": err_tb,
                    }
                )
                response = _friendly_rag_failure_message()
        
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
st.caption("💡 **TenaAI** © 2025 | AI-powered System for Ethiopian Healthcare Professionals.")



