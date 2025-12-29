from sentence_transformers import SentenceTransformer

class LocalEmbedder:
    """
    Local embedding model for converting text chunks into vectors.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents):
        """
        Takes a list of LangChain Documents and returns embeddings.
        """
        texts = [doc.page_content for doc in documents]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings

    def embed_query(self, query: str):
        """
        Embed a single query string.
        """
        return self.model.encode(query)
