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
        if not context:
            return "⚠️ No relevant documents found."

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
                temperature=0.1,  # Deterministic output
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ Error generating summary: {str(e)}"


# Example local usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What are the registration requirements for new and repeat candidates in the EHPLE system?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)

