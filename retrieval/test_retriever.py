from ingestion.loader import load_documents
from ingestion.chunker import chunk_documents
from ingestion.embedder import LocalEmbedder
from vectorstore.faiss_store import FAISSVectorStore
from retrieval.retriever import SemanticRetriever

docs = load_documents("data")
chunks = chunk_documents(docs)

embedder = LocalEmbedder()
embeddings = embedder.embed_documents(chunks)

store = FAISSVectorStore(embedding_dim=len(embeddings[0]))
store.add_embeddings(embeddings)

retriever = SemanticRetriever(embedder, store, chunks)

query = "What is the remote work policy?"
results = retriever.retrieve(query, top_k=2)

print("Retrieved chunks:\n")
for i, doc in enumerate(results):
    print(f"--- Result {i+1} ---")
    print(doc.page_content)
    print()
