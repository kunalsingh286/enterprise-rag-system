# Enterprise RAG System (Fully Local)

## Overview
This project implements a fully local, hallucination-resistant Retrieval-Augmented Generation (RAG) system for enterprise internal documents.

The system is designed to work completely offline using:
- Local embeddings
- FAISS vector search
- Local LLM (TinyLlama via Ollama)

## Why This Project
Enterprise AI systems prioritize **grounded answers, safety, and cost control** over creative generation.
This project focuses on retrieval quality rather than model size.

## Architecture (High-Level)
Enterprise Documents → Chunking → Embeddings → FAISS → Retriever → Local LLM → Grounded Answer

## Tech Stack
- Python
- FAISS (Vector Database)
- Sentence Transformers (Local Embeddings)
- TinyLlama via Ollama (Local LLM)

## Key Features (Planned)
- Offline document ingestion
- Semantic chunking and retrieval
- Hallucination-resistant prompting
- Model-agnostic RAG pipeline

## Status
🚧 Phase 0: Project initialization
