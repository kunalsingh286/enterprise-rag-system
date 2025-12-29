# Enterprise RAG System (Fully Local & Offline)

## Overview
This project implements a **fully local, hallucination-resistant Retrieval-Augmented Generation (RAG) system** designed for querying enterprise internal documents.

The system runs **completely offline** using local embeddings, FAISS vector search, and a lightweight local LLM (TinyLlama via Ollama).  
The design intentionally prioritizes **retrieval correctness over model size**.

---

## Why Retrieval-Augmented Generation (RAG)?
Large Language Models can hallucinate when answering questions outside their training data.

RAG mitigates this by:
- Retrieving **relevant enterprise documents**
- Injecting them into the LLM prompt as **grounded context**
- Forcing the model to answer **only from retrieved information**

This project focuses on **retrieval quality**, not prompt engineering.

---

## Key Design Principles
- **Retrieval is the source of truth**
- **LLM is a constrained generation layer**
- **Fail safely when information is missing**
- **Model-agnostic RAG pipeline**
- **Cost-aware and offline-first**

---

## Architecture (High-Level)

Enterprise Documents
↓
Document Ingestion
↓
Semantic Chunking
↓
Local Embeddings (Sentence Transformers)
↓
FAISS Vector Store
↓
Top-K Semantic Retrieval
↓
Context Injection
↓
Local LLM (TinyLlama via Ollama)
↓
Grounded Answer / Safe Refusal


---

## Tech Stack

| Layer | Technology |
|-----|-----------|
| Language | Python |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector Store | FAISS |
| LLM | TinyLlama (via Ollama) |
| Interface | Command Line (CLI) |

---

## Project Structure

enterprise-rag-system/
├── data/ # Enterprise documents
├── ingestion/ # Loading, chunking, embeddings
├── vectorstore/ # FAISS vector index
├── retrieval/ # Semantic retrieval logic
├── llm/ # Local LLM wrapper
├── rag/ # End-to-end RAG pipeline
├── app.py # CLI entry point
├── requirements.txt
└── README.md


---

## How Hallucinations Are Prevented
- The LLM **never sees raw documents**
- Only **retrieved chunks** are passed as context
- The prompt enforces:
  > “If the answer is not present, say *Information not available*”

If retrieval fails, the system **fails safely** instead of hallucinating.

---

## Setup Instructions

### 1️⃣ Clone Repository
```bash
git clone https://github.com/<your-username>/enterprise-rag-system.git
cd enterprise-rag-system


2️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate

3️⃣ Install Dependencies
python -m pip install -r requirements.txt

4️⃣ Install Ollama & TinyLlama
ollama pull tinyllama
Verify:
ollama run tinyllama

##Running the Application
python app.py

You can then ask questions like:
What is the remote work policy?
Who approves remote work requests?

If the information is not present:
Information not available.

## Example Use Case

This system can be used for:
HR policy Q&A
Internal onboarding assistance
Compliance documentation lookup
Offline enterprise knowledge systems

##Why a Small Local LLM?

TinyLlama was intentionally chosen to:
Validate retrieval quality
Avoid masking poor retrieval with large model capabilities
Keep the system fully local and cost-free
If the RAG pipeline works with a small model, it will work with larger models as well.


##Future Improvements

Hybrid retrieval (BM25 + FAISS)
Retrieval evaluation metrics (Precision@K, Recall@K)
Document versioning and metadata filtering
Web or Streamlit-based UI
Authentication and access control

