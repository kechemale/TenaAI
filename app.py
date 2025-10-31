from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

# Example usage
if __name__ == "__main__":
    
    docs = load_all_documents("data/core_documents_and_guidelines/test")
    store = FaissVectorStore("faiss_store")
    store.load()
    rag_search = RAGSearch()
    query = "How is the passing score (cut-off point) determined for the EHPLE medicine exam?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
