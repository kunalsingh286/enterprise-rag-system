from ingestion.loader import load_documents
from ingestion.chunker import chunk_documents
from ingestion.embedder import LocalEmbedder
from vectorstore.faiss_store import FAISSVectorStore
from retrieval.retriever import SemanticRetriever
from llm.tinyllama import TinyLlamaLLM
from rag.pipeline import RAGPipeline


def build_rag_system(data_dir="data"):
    """
    Build and return a ready-to-use RAG pipeline.
    """
    print("🔹 Loading enterprise documents...")
    docs = load_documents(data_dir)

    print("🔹 Chunking documents...")
    chunks = chunk_documents(docs)

    print("🔹 Generating embeddings (local)...")
    embedder = LocalEmbedder()
    embeddings = embedder.embed_documents(chunks)

    print("🔹 Building FAISS index...")
    store = FAISSVectorStore(embedding_dim=len(embeddings[0]))
    store.add_embeddings(embeddings)

    print("🔹 Initializing retriever and LLM...")
    retriever = SemanticRetriever(embedder, store, chunks)
    llm = TinyLlamaLLM()

    print("✅ RAG system ready.\n")
    return RAGPipeline(retriever, llm)


def main():
    rag = build_rag_system()

    print("Enterprise RAG System (Local)")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Ask a question: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Goodbye 👋")
            break

        answer = rag.answer(query)
        print("\nAnswer:\n")
        print(answer)
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()
