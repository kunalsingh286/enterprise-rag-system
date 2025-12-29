class RAGPipeline:
    """
    End-to-end Retrieval Augmented Generation pipeline.
    """

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def answer(self, query: str, top_k=5) -> str:
        """
        Generate an answer grounded only in retrieved documents.
        """
        retrieved_docs = self.retriever.retrieve(query, top_k=top_k)

        if not retrieved_docs:
            return "Information not available."

        context = "\n\n".join(
            [doc.page_content for doc in retrieved_docs]
        )

        prompt = f"""
You are an enterprise internal knowledge assistant.
Answer the question ONLY using the context below.
If the answer is not present, say "Information not available."

Context:
{context}

Question:
{query}

Answer:
"""

        return self.llm.generate(prompt)
