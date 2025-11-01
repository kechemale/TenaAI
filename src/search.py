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

# Ensure project root is on sys.path so `import src.*` works when running this file directly
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from openai import OpenAI

load_dotenv()

class RAGSearch:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "deepseek-chat",
        api_key: str | None = None,
    ):
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

def search_and_summarize(self, query: str, top_k: int = 5) -> str:
    results = self.vectorstore.query(query, top_k=top_k)
    texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
    context = "\n\n".join(texts)

    # If no relevant context found
    if not context.strip():
        return (
            "⚠️ No relevant Ethiopian healthcare document found in the database for your query.\n\n"
            "The assistant is restricted to answer **only** from the official Ethiopian clinical guidelines, "
            "and cannot generate answers from general knowledge. Please rephrase or try another clinical topic."
        )

    # Construct the RAG-grounded prompt
    prompt = f"""
You are a healthcare assistant specialized in Ethiopian clinical and public health guidelines.
Answer the question **only** using the provided context below. 
If the information is not clearly available, explicitly say so — do not generate or guess an answer.

Question:
{query}

Context (official Ethiopian healthcare documents):
{context}

Your response should:
- Be medically accurate and concise.
- Use Ethiopian healthcare terminology when possible.
- If unclear or missing, say: "The Ethiopian clinical guideline does not specify this information."
"""

    try:
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a cautious healthcare assistant that responds strictly based on Ethiopian "
                        "clinical guidelines and NEVER uses general knowledge or external data. "
                        "If context is missing or incomplete, clearly state that."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,  # deterministic output
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error generating summary: {str(e)}"



# Example local usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What are the registration requirements for new and repeat candidates in the EHPLE system?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)


