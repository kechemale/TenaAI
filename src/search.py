# import os
# from dotenv import load_dotenv
# from src.vectorstore import FaissVectorStore

# load_dotenv()
# from openai import OpenAI

# class RAGSearch:
#     def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "deepseek-chat"):
#         self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
#         # Load or build vectorstore
#         faiss_path = os.path.join(persist_dir, "faiss.index")
#         meta_path = os.path.join(persist_dir, "metadata.pkl")
#         if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
#             from data_loader import load_all_documents
#             docs = load_all_documents("data/core_documents_and_guidelines/test")
#             self.vectorstore.build_from_documents(docs)
#         else:
#             self.vectorstore.load()
        
#         # Initialize DeepSeek client
#         deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "sk-6d940aacef3f4a86ba5392943993c249")
#         if not deepseek_api_key:
#             raise ValueError("DEEPSEEK_API_KEY environment variable is required")
        
#         self.client = OpenAI(
#             api_key=deepseek_api_key,
#             base_url="https://api.deepseek.com/v1"  # DeepSeek API endpoint
#         )
#         self.llm_model = llm_model
#         print(f"[INFO] DeepSeek LLM initialized: {llm_model}")

#     def search_and_summarize(self, query: str, top_k: int = 5) -> str:
#         results = self.vectorstore.query(query, top_k=top_k)
#         texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
#         context = "\n\n".join(texts)
#         if not context:
#             return "No relevant documents found."
        
#         prompt = f"""Based on the following context, provide a comprehensive summary that answers the query: '{query}'

# Context:
# {context}

# Please provide a clear and concise answer that directly addresses the query:"""
        
#         try:
#             response = self.client.chat.completions.create(
#                 model=self.llm_model,
#                 messages=[
#                     {"role": "system", "content": "You are a helpful assistant that provides accurate summaries based on the given context."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 temperature=0.1  # Lower temperature for more deterministic responses
#             )
#             return response.choices[0].message.content
#         except Exception as e:
#             return f"Error generating summary: {str(e)}"

# # Example usage
# if __name__ == "__main__":
#     rag_search = RAGSearch()
#     query = "What are the registration requirements for new and repeat candidates in the EHPLE system?"
#     summary = rag_search.search_and_summarize(query, top_k=3)
#     print("Summary:", summary)



#Version 2

# import sys
# import os
# from pathlib import Path

# # Ensure project root is on sys.path so `import src.*` works when running this file directly
# project_root = Path(__file__).resolve().parents[1]
# if str(project_root) not in sys.path:
#     sys.path.insert(0, str(project_root))

# from dotenv import load_dotenv
# from src.vectorstore import FaissVectorStore
# from openai import OpenAI

# load_dotenv()

# class RAGSearch:
#     def __init__(
#         self,
#         persist_dir: str = "faiss_store",
#         embedding_model: str = "all-MiniLM-L6-v2",
#         llm_model: str = "deepseek-chat",
#         api_key: str | None = None,
#     ):
#         self.vectorstore = FaissVectorStore(persist_dir, embedding_model)

#         # Load or build vectorstore
#         faiss_path = os.path.join(persist_dir, "faiss.index")
#         meta_path = os.path.join(persist_dir, "metadata.pkl")
#         if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
#             from data_loader import load_all_documents
#             docs = load_all_documents("data/core_documents_and_guidelines/test")
#             self.vectorstore.build_from_documents(docs)
#         else:
#             self.vectorstore.load()

#         # ✅ Securely initialize DeepSeek API key
#         deepseek_api_key = (
#             api_key or os.getenv("DEEPSEEK_API_KEY")
#         )
#         if not deepseek_api_key:
#             raise ValueError("❌ DeepSeek API key missing! Provide it via Streamlit secrets or .env file.")

#         self.client = OpenAI(
#             api_key=deepseek_api_key,
#             base_url="https://api.deepseek.com/v1"
#         )
#         self.llm_model = llm_model
#         print(f"[INFO] ✅ DeepSeek LLM initialized with model: {llm_model}")

#     def search_and_summarize(self, query: str, top_k: int = 5) -> str:
#         results = self.vectorstore.query(query, top_k=top_k)
#         texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
#         context = "\n\n".join(texts)
#         if not context:
#             return "⚠️ No relevant documents found."

#         prompt = f"""You are a healthcare assistant specialized in Ethiopian medical guidelines.

# Based on the following context, answer the question comprehensively:

# Question:
# {query}

# Context:
# {context}

# Please provide a clear and medically accurate summary directly addressing the query.
# """

#         try:
#             response = self.client.chat.completions.create(
#                 model=self.llm_model,
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": "You are a helpful assistant that summarizes Ethiopian clinical guidelines accurately."
#                     },
#                     {"role": "user", "content": prompt},
#                 ],
#                 temperature=0.1,  # Deterministic output
#             )
#             return response.choices[0].message.content
#         except Exception as e:
#             return f"❌ Error generating summary: {str(e)}"


# # Example local usage
# if __name__ == "__main__":
#     rag_search = RAGSearch()
#     query = "What are the registration requirements for new and repeat candidates in the EHPLE system?"
#     summary = rag_search.search_and_summarize(query, top_k=3)
#     print("Summary:", summary)


#Version 3





# import os
# from dotenv import load_dotenv
# from src.vectorstore import FaissVectorStore

# load_dotenv()
# from openai import OpenAI

# class RAGSearch:
#     def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "deepseek-chat"):
#         self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
#         # Load or build vectorstore
#         faiss_path = os.path.join(persist_dir, "faiss.index")
#         meta_path = os.path.join(persist_dir, "metadata.pkl")
#         if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
#             from data_loader import load_all_documents
#             docs = load_all_documents("data/core_documents_and_guidelines/test")
#             self.vectorstore.build_from_documents(docs)
#         else:
#             self.vectorstore.load()
        
#         # Initialize DeepSeek client
#         deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "sk-6d940aacef3f4a86ba5392943993c249")
#         if not deepseek_api_key:
#             raise ValueError("DEEPSEEK_API_KEY environment variable is required")
        
#         self.client = OpenAI(
#             api_key=deepseek_api_key,
#             base_url="https://api.deepseek.com/v1"  # DeepSeek API endpoint
#         )
#         self.llm_model = llm_model
#         print(f"[INFO] DeepSeek LLM initialized: {llm_model}")

#     def search_and_summarize(self, query: str, top_k: int = 5) -> str:
#         results = self.vectorstore.query(query, top_k=top_k)
#         texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
#         context = "\n\n".join(texts)
#         if not context:
#             return "No relevant documents found."
        
#         prompt = f"""Based on the following context, provide a comprehensive summary that answers the query: '{query}'

# Context:
# {context}

# Please provide a clear and concise answer that directly addresses the query:"""
        
#         try:
#             response = self.client.chat.completions.create(
#                 model=self.llm_model,
#                 messages=[
#                     {"role": "system", "content": "You are a helpful assistant that provides accurate summaries based on the given context."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 temperature=0.1  # Lower temperature for more deterministic responses
#             )
#             return response.choices[0].message.content
#         except Exception as e:
#             return f"Error generating summary: {str(e)}"

# # Example usage
# if __name__ == "__main__":
#     rag_search = RAGSearch()
#     query = "What are the registration requirements for new and repeat candidates in the EHPLE system?"
#     summary = rag_search.search_and_summarize(query, top_k=3)
#     print("Summary:", summary)


# import os
# from dotenv import load_dotenv
# from src.vectorstore import FaissVectorStore

# load_dotenv()
# from openai import OpenAI

# class RAGSearch:
#     def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "deepseek-chat"):
#         self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
#         # Load or build vectorstore
#         faiss_path = os.path.join(persist_dir, "faiss.index")
#         meta_path = os.path.join(persist_dir, "metadata.pkl")
#         if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
#             from data_loader import load_all_documents
#             docs = load_all_documents("data/core_documents_and_guidelines/test")
#             self.vectorstore.build_from_documents(docs)
#         else:
#             self.vectorstore.load()
        
#         # Initialize DeepSeek client
#         deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "sk-6d940aacef3f4a86ba5392943993c249")
#         if not deepseek_api_key:
#             raise ValueError("DEEPSEEK_API_KEY environment variable is required")
        
#         self.client = OpenAI(
#             api_key=deepseek_api_key,
#             base_url="https://api.deepseek.com/v1"  # DeepSeek API endpoint
#         )
#         self.llm_model = llm_model
#         print(f"[INFO] DeepSeek LLM initialized: {llm_model}")

#     def search_and_summarize(self, query: str, top_k: int = 5) -> str:
#         results = self.vectorstore.query(query, top_k=top_k)
#         texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
#         context = "\n\n".join(texts)
#         if not context:
#             return "No relevant documents found."
        
#         prompt = f"""Based on the following context, provide a comprehensive summary that answers the query: '{query}'

# Context:
# {context}

# Please provide a clear and concise answer that directly addresses the query:"""
        
#         try:
#             response = self.client.chat.completions.create(
#                 model=self.llm_model,
#                 messages=[
#                     {"role": "system", "content": "You are a helpful assistant that provides accurate summaries based on the given context."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 temperature=0.1  # Lower temperature for more deterministic responses
#             )
#             return response.choices[0].message.content
#         except Exception as e:
#             return f"Error generating summary: {str(e)}"

# # Example usage
# if __name__ == "__main__":
#     rag_search = RAGSearch()
#     query = "What are the registration requirements for new and repeat candidates in the EHPLE system?"
#     summary = rag_search.search_and_summarize(query, top_k=3)
#     print("Summary:", summary)



import sys
import os
from pathlib import Path
import csv
import json
from datetime import datetime

import gspread

import streamlit as st
from google.oauth2.service_account import Credentials

# Ensure project root is on sys.path so `import src.*` works when running this file directly
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from openai import OpenAI

load_dotenv()
def _append_eval_to_gsheet(
        query: str,
        top_k: int,
        contexts: list,
        response: str,
        sheet_name: str = "TenaAI_Logs"
    ):
        """
        Appends query, context, and response data to a Google Sheet.

        Args:
            query (str): User query.
            top_k (int): Number of retrieved contexts.
            contexts (list): Contexts or retrieved passages.
            response (str): Model response.
            sheet_name (str): Google Sheet name.
        """
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "top_k": top_k,
            "contexts": json.dumps(contexts, ensure_ascii=False),
            "response": response,
        }

        try:
            # Google Sheets authentication
            scopes = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"
            ]
            creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
            client = gspread.authorize(creds)

            # Open the sheet
            sheet = client.open(sheet_name).sheet1  # first tab

            # Append the row
            sheet.append_row(
                [
                    row["timestamp"],
                    row["query"],
                    row["top_k"],
                    row["contexts"],
                    row["response"]
                ],
                value_input_option="USER_ENTERED"
            )
            print(f"✅ Appended row to Google Sheet '{sheet_name}' successfully.")

        except Exception as e:
            print(f"⚠️ Failed to append to Google Sheet '{sheet_name}': {e}")
                
class RAGSearch:

    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "deepseek-chat",
        api_key: str | None = None,
    ):
        self.eval_log_default = Path("evaluation_logs/queries.csv")
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)

        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from data_loader import load_all_documents
            docs = load_all_documents("data/core_documents_and_guidelines/test")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()

        # ✅ Securely initialize DeepSeek API key
        deepseek_api_key = (
            api_key or os.getenv("DEEPSEEK_API_KEY")
        )
        if not deepseek_api_key:
            raise ValueError("❌ DeepSeek API key missing! Provide it via Streamlit secrets or .env file.")

        self.client = OpenAI(
            api_key=deepseek_api_key,
            base_url="https://api.deepseek.com/v1"
        )
        self.llm_model = llm_model
        print(f"[INFO] ✅ DeepSeek LLM initialized with model: {llm_model}")
    
    def search_and_summarize(
        self,
        query: str,
        top_k: int = 5,
        save_path: str | None = None,
        return_contexts: bool = False  # optional
) -> str | tuple[str, list]:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r.get("metadata")]
        #   Capture document IDs for evaluation/logging
        contexts_meta = []
        for r in results:
            meta = r.get("metadata") or {}
            contexts_meta.append({
                "source": meta.get("source") or meta.get("doc_id") or None,
                "text": meta.get("text", "")[:1000]  # trim for CSV/Sheets if desired
        })

        context = "\n\n".join(texts)
        if not context:
            answer = "⚠️ No relevant documents found."
            if return_contexts:
                return answer, contexts_meta
            return answer

        prompt = f"""You are a healthcare assistant specialized in Ethiopian medical guidelines.

    Based on the following context, answer the question comprehensively:

    Question:
    {query}

    Context:
    {context}

    Please provide a clear and medically accurate summary directly addressing the query.
    """

        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that summarizes Ethiopian clinical guidelines accurately."
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"❌ Error generating summary: {str(e)}"

        # save evaluation row if requested
        csv_target = Path(save_path) if save_path else self.eval_log_default
        try:
            self._append_eval_csv(csv_target, query, top_k, contexts_meta, answer)
        except Exception as e:
            print(f"[WARN] Failed to write eval CSV: {e}")

        if return_contexts:
            return answer, contexts_meta
        return answer
# Example local usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What are the registration requirements for new and repeat candidates in the EHPLE system?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
