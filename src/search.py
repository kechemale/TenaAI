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

import sys
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

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
        api_key: Optional[str] = None,
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
        # Ensure we use the cheaper chat model even if misconfigured.
        if (llm_model or "").strip().lower() == "deepseek-reasoner":
            llm_model = "deepseek-chat"
        self.llm_model = llm_model
        print(f"[INFO] ✅ DeepSeek LLM initialized with model: {llm_model}")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        return self.vectorstore.query(query, top_k=top_k)

    @staticmethod
    def _basename(path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        try:
            return os.path.basename(path)
        except Exception:
            return str(path)

    @staticmethod
    def _infer_page_from_text(text: str) -> Optional[int]:
        """Fallback when page isn't present in metadata.

        Some of your chunks start with a line like "62\n4.2.5 ...". If so, treat it as page.
        """
        if not text:
            return None
        first = text.lstrip().splitlines()[0].strip() if text.strip() else ""
        if re.fullmatch(r"\d{1,4}", first):
            try:
                return int(first)
            except Exception:
                return None
        return None

    @staticmethod
    def _infer_section_or_chapter(text: str) -> Optional[str]:
        """Best-effort extraction of a chapter/section label from the chunk text."""
        if not text:
            return None

        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        if not lines:
            return None

        # Drop leading page number line like "62"
        if re.fullmatch(r"\d{1,4}", lines[0]):
            lines = lines[1:]
        if not lines:
            return None

        head = lines[0][:120]
        # Examples: "Chapter 4 ..."
        if re.match(r"^(chapter|chap)\s+\d+\b", head, flags=re.IGNORECASE):
            return head
        # Examples: "4.2.5 Follow-up and Clinical Monitoring"
        if re.match(r"^\d+(?:\.\d+)+\b", head):
            return head
        return None

    def format_references(self, results: List[Dict[str, Any]]) -> str:
        """Create a stable numbered reference list for the retrieved chunks."""
        lines: list[str] = []
        for i, r in enumerate(results or [], start=1):
            meta = r.get("metadata") if isinstance(r, dict) else None
            meta = meta if isinstance(meta, dict) else {}
            text = (meta.get("text") or "") if isinstance(meta, dict) else ""

            src = self._basename(meta.get("source"))
            page = meta.get("page")
            if isinstance(page, int):
                # PyPDFLoader pages are typically 0-based.
                page_label = str(page + 1)
            elif isinstance(page, str) and page.isdigit():
                page_label = page
            else:
                fallback = self._infer_page_from_text(text)
                page_label = str(fallback) if isinstance(fallback, int) else "N/A"

            section = meta.get("section")
            if not section:
                section = self._infer_section_or_chapter(text) or "N/A"

            src_label = src or "Unknown document"
            lines.append(f"[{i}] {src_label} — page {page_label} — {section}")
        return "\n".join(lines) if lines else "(No references available)"

    @staticmethod
    def build_context(results: List[Dict[str, Any]]) -> str:
        # Provide numbered context blocks so the model can cite them like [1], [2], ...
        blocks: list[str] = []
        for i, r in enumerate(results or [], start=1):
            meta = r.get("metadata") if isinstance(r, dict) else None
            meta = meta if isinstance(meta, dict) else {}
            text = meta.get("text", "") if isinstance(meta, dict) else ""
            if not text:
                continue
            blocks.append(f"[{i}]\n{text}")
        return "\n\n".join(blocks)

    @staticmethod
    def build_prompt(query: str, context: str) -> str:
        return f"""You are a healthcare assistant specialized in Ethiopian medical guidelines.

Based on the following context, answer the question comprehensively:

- When you use information from a context block, add an inline citation like [1] or [2].
- At the end of your answer, include a short 'References' section listing the sources you used.

Question:
{query}

Context:
{context}

Please provide a clear and medically accurate summary directly addressing the query.
"""

    def generate_answer(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that summarizes Ethiopian clinical guidelines accurately.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        # Store last-call metadata for debugging/cost attribution.
        # Most OpenAI-compatible APIs include `model` and `usage`.
        self.last_model = getattr(response, "model", None)
        self.last_usage = getattr(response, "usage", None)
        return response.choices[0].message.content

    def search_and_summarize_with_debug(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        results = self.retrieve(query, top_k=top_k)
        context = self.build_context(results)
        if not context:
            return {
                "answer": "⚠️ No relevant documents found.",
                "results": results,
                "context": context,
                "prompt": None,
                "llm_model": None,
                "llm_usage": None,
            }

        prompt = self.build_prompt(query, context)
        answer = self.generate_answer(prompt)
        # Ensure references are always present, even if the model forgets.
        refs = self.format_references(results)
        answer = f"{answer}\n\n---\n\nReferences:\n{refs}"
        return {
            "answer": answer,
            "results": results,
            "context": context,
            "prompt": prompt,
            "llm_model": getattr(self, "last_model", None),
            "llm_usage": getattr(self, "last_usage", None),
        }

    def search_and_summarize(self, query: str, top_k: int = 3) -> str:
        try:
            payload = self.search_and_summarize_with_debug(query, top_k=top_k)
            return payload["answer"]
        except Exception as e:
            return f"❌ Error generating summary: {str(e)}"


# Example local usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What are the registration requirements for new and repeat candidates in the EHPLE system?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
