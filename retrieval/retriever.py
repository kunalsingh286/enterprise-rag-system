class SemanticRetriever:
    """
    Handles semantic retrieval using embeddings and FAISS.
    """

    def __init__(self, embedder, vector_store, documents):
        self.embedder = embedder
        self.vector_store = vector_store
        self.documents = documents

    def retrieve(self, query: str, top_k=5):
        """
        Retrieve top_k most relevant document chunks for a query.
        """
        query_embedding = self.embedder.embed_query(query)
        distances, indices = self.vector_store.search(
            query_embedding, top_k=top_k
        )

        retrieved_docs = []
        for idx in indices:
            if idx < len(self.documents):
                retrieved_docs.append(self.documents[idx])

        return retrieved_docs
