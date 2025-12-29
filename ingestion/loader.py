from langchain_community.document_loaders import PyPDFLoader, TextLoader
from pathlib import Path

def load_documents(data_dir: str):
    """
    Load all PDF and TXT documents from a directory.
    Returns a list of LangChain Document objects.
    """
    documents = []
    data_path = Path(data_dir)

    for file in data_path.iterdir():
        if file.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file))
            documents.extend(loader.load())

        elif file.suffix.lower() == ".txt":
            loader = TextLoader(str(file), encoding="utf-8")
            documents.extend(loader.load())

    return documents
