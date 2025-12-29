from ingestion.loader import load_documents
from ingestion.chunker import chunk_documents
from ingestion.embedder import LocalEmbedder
from vectorstore.faiss_store import FAISSVectorStore

docs = load_documents("data")
chunks = chunk_documents(docs)

embedder = LocalEmbedder()
doc_embeddings = embedder.embed_documents(chunks)

store = FAISSVectorStore(embedding_dim=len(doc_embeddings[0]))
store.add_embeddings(doc_embeddings)

query = "What is the remote work policy?"
query_embedding = embedder.embed_query(query)

distances, indices = store.search(query_embedding, top_k=3)

print("Top retrieved chunk indices:", indices)
print("\nTop retrieved chunk content:\n")
print(chunks[indices[0]].page_content)
