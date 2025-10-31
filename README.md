
````markdown
# 🩺 TenaAI: RAG Healthcare Knowledge System

**TenaAI** is a **Retrieval-Augmented Generation (RAG)** system designed to empower **healthcare professionals in Ethiopia** with intelligent search and summarization capabilities.  
It retrieves relevant information from official healthcare guidelines and documents to support evidence-based decision-making.

---

## 🚀 Quick Start

### 🧩 Installation
```bash
git clone https://github.com/kechemale/TenaAI.git
cd TenaAI
pip install -r requirements.txt
````

---

### ⚙️ Setup

1. **Get an API Key**

   * Obtain a [DeepSeek API key](https://platform.deepseek.com) or use another LLM of your choice.
   * If using a different model, update the retrieval function accordingly.

2. **Create a `.env` file** in the project root:

   ```env
   DEEPSEEK_API_KEY=your_api_key_here
   ```

3. **Add your healthcare documents**

   * Place all source documents (PDFs, guidelines, etc.) inside the `data/` folder.

---

### 💡 Usage Example

```python
from rag_system import RAGSearch

rag = RAGSearch()

query = "What are the registration requirements for healthcare professionals?"
summary = rag.search_and_summarize(query)

print(summary)
```

---

## 🔍 Features

* **Smart Document Search** – FAISS-powered vector database for fast and relevant retrieval
* **AI Summarization** – DeepSeek (or other LLMs) provides concise, accurate summaries
* **Healthcare Focus** – Tailored to Ethiopian healthcare guidelines and policies
* **Flexible Formats** – Supports PDF; easily extendable to Word, Text, and CSV

---

## 🧠 Use Cases

* Quick access to FMOH and clinical guidelines
* Decision support for medical practitioners
* Policy and research reference for healthcare administrators
* Integration into healthcare knowledge portals

---

## 🤝 Contributing

Contributions are welcome!
Feel free to open issues or submit pull requests to enhance functionality, add new LLM integrations, or improve document processing.

---

## 📜 License

This project is licensed under the **MIT License**.
See the [LICENSE](LICENSE) file for details.

```

---

✅ Just copy the entire block above into your `README.md` file — it’s already Markdown-formatted for GitHub.  

Would you like me to add an **“Architecture Overview”** section (with a short RAG pipeline diagram using Mermaid) for better documentation?
```




