# TenaAI: RAG Healthcare System

A Retrieval-Augmented Generation system for intelligent search and summarization of healthcare documents in Ethiopia for healthcare professionals.

## Quick Start

### Installation
```bash
git clone https://github.com/kechemale/TenaAI.git
cd TenaAI
pip install -r requirements.txt

Setup

Get DeepSeek API key from platform.deepseek.com

Create .env file:

env
DEEPSEEK_API_KEY=your_api_key_here or any other LLM, but you need to modify your retreival function
Add healthcare documents to data/ folder

Usage

from rag_system import RAGSearch

rag = RAGSearch()
query = "What are the registration requirements for healthcare professionals?"
summary = rag.search_and_summarize(query)
print(summary)

Features
Smart Document Search - FAISS-powered vector search

AI Summarization - DeepSeek LLM for accurate summaries

Healthcare Focus - Optimized for medical documents

Formats - PDF, You can also extend to Word, Text, CSV support
