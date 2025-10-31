````markdown
# 🩺 TenaAI: RAG Healthcare Knowledge System

**TenaAI** is a **Retrieval-Augmented Generation (RAG)** system designed to empower **healthcare professionals in Ethiopia** with intelligent search and summarization capabilities.  
It retrieves relevant information from official healthcare guidelines and documents to support evidence-based decision-making.

---

## Quick Start

###  Installation

```bash
git clone https://github.com/kechemale/TenaAI.git
cd TenaAI
pip install -r requirements.txt
````

---

### Setup

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

##  Usage Example

```python
from rag_system import RAGSearch

rag = RAGSearch()

query = "How is the passing score (cut-off point) determined for the EHPLE medicine exam?"
summary = rag.search_and_summarize(query)

print(summary)
```

**Summary Example:**

> The passing score (cut-off point) for the EHPLE medicine exam is determined using the Modified-Angoff method.
> This process involves a panel of subject matter experts who evaluate each test question and estimate the probability that a minimally competent examinee would answer it correctly.
> The average of these predictions across all test items establishes the recommended cut-off point, which represents the minimum level of knowledge and skill required to pass the exam.

---

###  Example 2

```python
query = "How can postpartum hemorrhage due to uterine inversion be recognized and managed?"
```

**Summary Example:**

> **Recognition:**

> * Signs of postpartum hemorrhage: vaginal bleeding (>500ml after vaginal delivery), tachycardia, hypotension, and pale conjunctivae
> * Key diagnostic finding: a pinkish mass protruding from the cervix with an absent palpable uterus on abdominal examination

> **Management:**

> * Immediate manual replacement of the inverted uterus
> * Administration of uterotonics to control bleeding
> * Fluid resuscitation and hemodynamic support for shock
> * Surgical intervention if manual replacement fails

> The condition typically occurs during or after placental delivery.
> Risk factors include mismanagement of the third stage of labor, adherent placenta, and short umbilical cord.

---

##  Features

* **Smart Document Search** – FAISS-powered vector database for fast and relevant retrieval
* **AI Summarization** – DeepSeek (or other LLMs) provides concise, accurate summaries
* **Healthcare Focus** – Tailored to Ethiopian healthcare guidelines and policies
* **Flexible Formats** – Supports PDF; easily extendable to Word, Text, and CSV

---

## Use Cases

* Quick access to FMOH and clinical guidelines
* Decision support for medical practitioners
* Policy and research reference for healthcare administrators
* Integration into healthcare knowledge portals

---

##  License

This project is licensed under the **MIT License**.
See the [LICENSE](LICENSE) file for details.





