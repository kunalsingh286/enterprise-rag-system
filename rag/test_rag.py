from ingestion.loader import load_documents
from ingestion.chunker import chunk_documents
from ingestion.embedder import LocalEmbedder
from vectorstore.faiss_store import FAISSVectorStore
from retrieval.retriever import SemanticRetriever
from llm.tinyllama import TinyLlamaLLM
from rag.pipeline import RAGPipeline

# Load and prepare documents
docs = load_documents("data")
chunks = chunk_documents(docs)

# Embeddings
embedder = LocalEmbedder()
embeddings = embedder.embed_documents(chunks)

# Vector store
store = FAISSVectorStore(embedding_dim=len(embeddings[0]))
store.add_embeddings(embeddings)

# Retriever
retriever = SemanticRetriever(embedder, store, chunks)

# LLM
llm = TinyLlamaLLM()

# RAG pipeline
rag = RAGPipeline(retriever, llm)

# Test query
query = "What is the remote work policy?"
response = rag.answer(query, top_k=3)

print("Question:", query)
print("\nAnswer:\n")
print(response)
