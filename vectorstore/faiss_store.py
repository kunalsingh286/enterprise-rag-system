import faiss
import numpy as np

class FAISSVectorStore:
    """
    FAISS-based vector store for semantic search.
    """

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)

    def add_embeddings(self, embeddings):
        """
        Add document embeddings to FAISS index.
        """
        vectors = np.array(embeddings).astype("float32")
        self.index.add(vectors)

    def search(self, query_embedding, top_k=5):
        """
        Search for top_k most similar vectors.
        Returns distances and indices.
        """
        query_vector = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_vector, top_k)
        return distances[0], indices[0]
