import os
from langchain_community.vectorstores import Chroma
from langchain.schema import Document as LCDocument
from .embedding_utils import get_embedding_model

class VectorStoreManager:
    def __init__(self, persist_dir: str = "./vectordb"):
        """
        Manages Chroma vector store with Hugging Face embeddings.
        """
        # Initialize embedding model (from embedding_utils)
        self.embedding_model = get_embedding_model()

        # Initialize Chroma with this embedding model
        self.vs = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embedding_model
        )

    def add_documents(self, docs):
        """
        Add documents (with automatic embedding).
        docs: List[Dict] with keys 'page_content' and 'metadata'
        """
        lc_docs = [LCDocument(page_content=d["page_content"], metadata=d["metadata"]) for d in docs]
        self.vs.add_documents(lc_docs)
        print(f"✅ Added {len(lc_docs)} documents to vector store.")

    def query(self, query: str, k: int):
        """
        Perform similarity search on the vector store.
        """
        results = self.vs.similarity_search(query, k=k)
        print(f"🔍 Retrieved {len(results)} documents for query: '{query}'")
        return results
