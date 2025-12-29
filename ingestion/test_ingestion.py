from ingestion.loader import load_documents
from ingestion.chunker import chunk_documents

docs = load_documents("data")
chunks = chunk_documents(docs)

print(f"Loaded documents: {len(docs)}")
print(f"Generated chunks: {len(chunks)}")
print("\nSample chunk:\n")
print(chunks[0].page_content)

