from ingestion.loader import load_documents
from ingestion.chunker import chunk_documents
from ingestion.embedder import LocalEmbedder

docs = load_documents("data")
chunks = chunk_documents(docs)

embedder = LocalEmbedder()
embeddings = embedder.embed_documents(chunks)

print(f"Total chunks: {len(chunks)}")
print(f"Embedding dimension: {len(embeddings[0])}")
